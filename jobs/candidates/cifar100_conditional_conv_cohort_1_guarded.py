"""GUARDED CAMPAIGN DRAFT: requires independent calibration audit, review and freeze before execution."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['HF_HOME'] = '/output/hf-cache'
os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
import copy
import hashlib
import json
import math
from pathlib import Path
import time
import traceback
import platform
import shutil
import sys
import resource
import subprocess
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from neural_lab.artifacts import ArtifactWriter, BestCheckpoint, load_json_state

# Deliberate submission gate: this is a proposed implementation, not a frozen protocol.
REVIEWED_AND_FROZEN = False
CFG = dict(epochs=100, seeds=[4801,4802,4803,4804,4805,4806], split_seed=20261101,
           batch=256, lr=.001, weight_decay=.01, threshold=-.01)
ARMS = ('dynamic','static','single')
OUT = Path('/output/campaign')
COHORTS = {'1': (4801,4802), '2': (4803,4804), '3': (4805,4806)}
COHORT = '1'
OWNED_OUTPUT = False
REV = 'aadb3af77e9048adbea6b47c21a81e47dd092ae5'
FILE = 'cifar100/train-00000-of-00001.parquet'
SHA = '694865d6b990e234804f01268586c41e88bcbbb75e20858432c05ad4081aca23'
START = time.monotonic()

class ConditionalConv(nn.Module):
    def __init__(self, static):
        super().__init__()
        self.static = static
        self.experts = nn.ModuleList([nn.Conv2d(64,64,3,padding=1) for _ in range(2)])
        self.gate = nn.Sequential(nn.Linear(64,16),nn.GELU(),nn.Linear(16,2))
    def forward(self, x):
        z = x.mean((2,3))
        if self.static:
            z = torch.zeros_like(z)
        p = self.gate(z).softmax(dim=1)
        e = torch.stack([conv(x) for conv in self.experts],dim=1)
        return (e * p[:,:,None,None,None]).sum(dim=1)

class Model(nn.Module):
    def __init__(self, arm):
        super().__init__()
        if arm not in ARMS: raise ValueError('unknown arm')
        self.stem = nn.Sequential(nn.Conv2d(3,64,2,stride=2),nn.GELU(),nn.BatchNorm2d(64))
        self.blocks = nn.ModuleList([nn.Sequential(nn.Conv2d(64,64,3,padding=1),nn.GELU(),nn.BatchNorm2d(64)) for _ in range(3)])
        self.last = nn.Sequential(nn.Conv2d(64,64,3,padding=1) if arm == 'single' else ConditionalConv(arm == 'static'),nn.GELU(),nn.BatchNorm2d(64))
        self.head = nn.Linear(64,100)
    def forward(self,x):
        x = self.stem(x)
        for block in self.blocks: x = block(x)
        return self.head(self.last(x).mean((2,3)))

def paired_initial_states(seed):
    torch.manual_seed(seed)
    single=Model('single'); dynamic=Model('dynamic')
    with torch.no_grad():
        reference=single.last[0]
        perturb=torch.randn_like(reference.weight)*(.01*reference.weight.std(unbiased=False))
        dynamic.last[0].experts[0].weight.copy_(reference.weight+perturb)
        dynamic.last[0].experts[1].weight.copy_(reference.weight-perturb)
        for expert in dynamic.last[0].experts: expert.bias.copy_(reference.bias)
        dynamic.last[0].gate[-1].weight.zero_(); dynamic.last[0].gate[-1].bias.zero_()
        dynamic.stem.load_state_dict(single.stem.state_dict()); dynamic.blocks.load_state_dict(single.blocks.state_dict())
        dynamic.last[2].load_state_dict(single.last[2].state_dict()); dynamic.head.load_state_dict(single.head.state_dict())
    return copy.deepcopy(single.state_dict()), copy.deepcopy(dynamic.state_dict())

def paired_summary(values):
    if len(values)!=len(CFG['seeds']) or not all(math.isfinite(v) for v in values):
        raise ValueError('all finite paired values required')
    mean=sum(values)/len(values)
    sd=math.sqrt(sum((x-mean)**2 for x in values)/(len(values)-1))
    margin=2.570581835636314*sd/math.sqrt(len(values))
    return dict(values=values,mean=mean,sample_sd=sd,descriptive_t95=[mean-margin,mean+margin])

@torch.no_grad()
def routing_diagnostics(model,raw,ids,mu,sd):
    if not isinstance(model.last[0], ConditionalConv): return None
    model.eval(); layer=model.last[0]; weights=[]
    def capture(module,args):
        x=args[0]; z=x.mean((2,3))
        if layer.static: z=torch.zeros_like(z)
        weights.append(layer.gate(z).softmax(1).cpu())
    hook=layer.register_forward_pre_hook(capture)
    try:
        for start in range(0,len(ids),CFG['batch']): model(inputs(raw,ids[start:start+CFG['batch']],mu,sd))
    finally: hook.remove()
    p=torch.cat(weights,0); entropy=-(p*p.clamp_min(1e-30).log()).sum(1)
    return dict(examples=len(p),mean_weights=p.mean(0).tolist(),weights_population_std=p.std(0,unbiased=False).tolist(),mean_entropy=float(entropy.mean()),expert_weight_l2=float(torch.linalg.vector_norm(layer.experts[0].weight-layer.experts[1].weight)),expert_bias_l2=float(torch.linalg.vector_norm(layer.experts[0].bias-layer.experts[1].bias)))

def schedule(seed, epochs, n):
    order, crop, flip = [], [], []
    for epoch in range(epochs):
        rng = np.random.default_rng(seed+100000+epoch*1000003)
        order.append(rng.permutation(n).astype(np.int32))
        crop.append(rng.integers(0,9,(n,2),dtype=np.int16))
        flip.append(rng.integers(0,2,n,dtype=np.uint8))
    return np.stack(order),np.stack(crop),np.stack(flip)

def inputs(raw, ids, mean, std, crop=None, flip=None):
    x = raw[ids].float().div_(255)
    if crop is not None:
        x = F.pad(x,(4,4,4,4),mode='reflect')
        rows = torch.arange(32,device=x.device)[None,:]+crop[:,0,None]
        cols = torch.arange(32,device=x.device)[None,:]+crop[:,1,None]
        x = x[torch.arange(len(ids),device=x.device)[:,None,None],:,rows[:,:,None],cols[:,None,:]].permute(0,3,1,2)
        x = torch.where(flip[:,None,None,None].bool(),x.flip(-1),x)
    return (x-mean)/std

@torch.no_grad()
def evaluate(model,raw,y,ids,mean,std):
    model.eval(); loss=0.; correct=0; outputs=[]
    for start in range(0,len(ids),CFG['batch']):
        ix=ids[start:start+CFG['batch']]; logits=model(inputs(raw,ix,mean,std))
        outputs.append(logits.cpu().numpy()); loss += F.cross_entropy(logits,y[ix],reduction='sum').item()
        correct += (logits.argmax(1)==y[ix]).sum().item()
    return dict(ce=loss/len(ids),accuracy=correct/len(ids)), np.concatenate(outputs)

def event(kind,**kw):
    row=dict(event=kind,elapsed_seconds=time.monotonic()-START,**kw)
    with (OUT/'progress.jsonl').open('a') as f: f.write(json.dumps(row,allow_nan=False)+'\n')
    print(json.dumps(row),flush=True)

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def publish(path, obj):
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('x') as f:
        json.dump(obj,f,indent=2,allow_nan=False); f.flush(); os.fsync(f.fileno())
    os.link(tmp,path); tmp.unlink()

def main():
    global OWNED_OUTPUT
    if not REVIEWED_AND_FROZEN: raise RuntimeError('Proposed candidate must be independently reviewed and frozen before execution')
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required; no CPU fallback')
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False; torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    cohort=COHORT
    if cohort not in COHORTS: raise ValueError('invalid embedded cohort')
    seeds=COHORTS[cohort]
    # Compare this snapshot to the independently frozen submission digest at audit.
    expected_source=digest(__file__)
    OUT.mkdir(parents=False,exist_ok=False); OWNED_OUTPUT=True
    disk_before=shutil.disk_usage(OUT).free
    if disk_before < 5_000_000_000: raise RuntimeError('insufficient output filesystem free space')
    writer=ArtifactWriter(OUT)
    import neural_lab.artifacts as artifacts
    if hashlib.sha256(Path(artifacts.__file__).read_bytes()).hexdigest() != '6ef5863486d3f9cfa4fac8b06823ff399986c78e898130e979a9035c6f062baf': raise ValueError('artifact helper hash mismatch')
    (OUT/'provenance').mkdir()
    for src,name in ((Path(__file__),'experiment.py'),(Path(artifacts.__file__),'artifacts.py')):
        dst=OUT/'provenance'/name; shutil.copyfile(src,dst); writer.register_file('datasets',dst,name,format='python',source=dict(role='executed source snapshot'))
    from huggingface_hub import HfApi,hf_hub_download
    import huggingface_hub, datasets
    info=HfApi(token=False).dataset_info('uoft-cs/cifar100',revision=REV,timeout=20)
    if info.sha != REV: raise ValueError('dataset revision mismatch')
    path=hf_hub_download(repo_id='uoft-cs/cifar100',repo_type='dataset',filename=FILE,revision=REV,token=False,local_dir=str(OUT/'hf-source'))
    if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=SHA: raise ValueError('dataset hash mismatch')
    source=writer.register_file('datasets',path,'cifar100-official-training-source',format='parquet',source=dict(repo='uoft-cs/cifar100',revision=REV,file=FILE,sha256=SHA,creator='https://www.cs.toronto.edu/~kriz/cifar.html',license=info.card_data.get('license') if info.card_data else None,huggingface_hub_version=huggingface_hub.__version__,datasets_version=datasets.__version__))
    ds=datasets.load_dataset('parquet',data_files={'train':path},split='train',cache_dir=str(OUT/'hf-cache'/'decoded'))
    images=np.stack([np.asarray(row['img'].convert('RGB'),dtype=np.uint8) for row in ds]); labels=np.asarray(ds['fine_label'],dtype=np.int64)
    if images.shape!=(50000,32,32,3) or not np.array_equal(np.bincount(labels,minlength=100),np.full(100,500)): raise ValueError('unexpected CIFAR-100 layout or fine-label counts')
    rng=np.random.default_rng(CFG['split_seed']); train=[]; dev=[]
    for c in range(100):
        ids=rng.permutation(np.flatnonzero(labels==c)); train.extend(ids[:450]); dev.extend(ids[450:])
    train=np.asarray(sorted(train)); dev=np.asarray(sorted(dev))
    if len(train)!=45000 or len(dev)!=5000 or len(np.unique(np.concatenate([train,dev])))!=50000 or np.intersect1d(train,dev).size or not np.array_equal(np.bincount(labels[train],minlength=100),np.full(100,450)) or not np.array_equal(np.bincount(labels[dev],minlength=100),np.full(100,50)): raise ValueError('invalid split')
    decoded_sha=hashlib.sha256(images.tobytes()+labels.tobytes()).hexdigest()
    pixels=images[train].astype(np.float64)/255; mean=pixels.mean((0,1,2)); std=pixels.std((0,1,2)); del pixels
    data=writer.save_dataset('cifar100-training-split',dict(images=images,labels=labels,train_ids=train,dev_ids=dev,mean=mean,std=std),provenance=dict(source=source,split_seed=CFG['split_seed'],generator='numpy PCG64, class order 0..99, first 450 per class train, rest development, sorted IDs',numpy_version=np.__version__,official_test='never requested',decoded_sha256_images_then_labels=decoded_sha,normalization='training RGB population mean and standard deviation'))
    raw=torch.from_numpy(images.transpose(0,3,1,2).copy()).to('cuda'); y=torch.from_numpy(labels).to('cuda')
    mu=torch.tensor(mean,dtype=torch.float32,device='cuda')[None,:,None,None]; sd=torch.tensor(std,dtype=torch.float32,device='cuda')[None,:,None,None]
    train_gpu=torch.tensor(train,device='cuda'); dev_gpu=torch.tensor(dev,device='cuda')
    source_sha=digest(__file__); records=[]
    if source_sha!=expected_source or digest(OUT/'provenance'/'experiment.py')!=expected_source: raise ValueError('source snapshot mismatch')
    if digest(OUT/'provenance'/'artifacts.py')!='6ef5863486d3f9cfa4fac8b06823ff399986c78e898130e979a9035c6f062baf': raise ValueError('helper snapshot mismatch')
    config=dict(**CFG,architecture='patch2 stem; three conv3 blocks; fourth block single or two expert conv3s with input-conditioned scalar softmax or zero-input static gate; all GELU/BN; global average and linear100; same W plus/minus independent normal perturbation at 0.01 W population std; equal biases; zero final gate weight and bias; gate Linear64-16/GELU/Linear16-2, temperature 1; other tensors shared',precision='deterministic float32, TF32 disabled',selection='minimum observed augmented minibatch CE before optimizer step after BN update and final canonical train CE; mixed scopes',source_sha256=source_sha)
    for seed in seeds:
        idx=CFG['seeds'].index(seed)
        order,crop,flip=schedule(seed,CFG['epochs'],len(train))
        sched=writer.save_dataset(f'schedule-{seed}',dict(order=order,crop=crop,flip=flip),provenance=dict(seed=seed,generator='PCG64 seed+100000+epoch*1000003; permutation,crop,flip',numpy_version=np.__version__,indexing='order indexes sorted train IDs; crop and flip indexed by order position'))
        torch.cuda.manual_seed_all(seed)
        single_state,dynamic_state=paired_initial_states(seed)
        refs=[source['path'],data['path'],sched['path']]
        arms=ARMS[idx%3:]+ARMS[:idx%3]
        for arm in arms:
            initial=single_state if arm=='single' else dynamic_state
            model=Model(arm); model.load_state_dict(initial,strict=True); assert all(torch.equal(v,initial[k]) for k,v in model.state_dict().items())
            initial_ref=writer.save_checkpoint(f'{seed}-{arm}-initial',initial,arm=arm,seed=seed,selection=dict(metric='paired initialization'),config=config,datasets=refs)
            model.to('cuda'); assert all(t.is_cuda for t in list(model.parameters())+list(model.buffers()))
            optimizer=torch.optim.AdamW(model.parameters(),lr=CFG['lr'],weight_decay=CFG['weight_decay'])
            best=BestCheckpoint(scope=config['selection']); traces=[]; epoch_times=[]; step=0; presentations=0
            torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); started=time.monotonic()
            for epoch in range(CFG['epochs']):
                model.train(); tick=time.monotonic()
                for start in range(0,len(train),CFG['batch']):
                    stop=min(start+CFG['batch'],len(train)); ix=train_gpu[torch.as_tensor(order[epoch,start:stop].astype(np.int64),device='cuda')]
                    c=torch.as_tensor(crop[epoch,start:stop].astype(np.int64),device='cuda'); f=torch.as_tensor(flip[epoch,start:stop],device='cuda')
                    optimizer.zero_grad(set_to_none=True); loss=F.cross_entropy(model(inputs(raw,ix,mu,sd,c,f)),y[ix]); value=float(loss.detach())
                    if not math.isfinite(value): raise FloatingPointError('nonfinite loss')
                    best.observe(model,value,step); traces.append([step,epoch,start,stop-start,value,0]); loss.backward()
                    for p in model.parameters():
                        if p.grad is not None: torch._assert_async(torch.isfinite(p.grad).all(),'nonfinite gradient')
                    optimizer.step(); step+=1; presentations+=stop-start
                torch.cuda.synchronize(); epoch_times.append(time.monotonic()-tick); event('epoch_complete',seed=seed,arm=arm,epoch=epoch+1,updates=step,seconds=epoch_times[-1])
            torch.cuda.synchronize(); training_seconds=time.monotonic()-started
            if not all(torch.isfinite(v).all().item() for v in model.state_dict().values()): raise FloatingPointError('nonfinite state')
            if step!=17600 or presentations!=4500000 or traces[-1][3]!=200: raise ValueError('invalid counts or partial batch')
            final_train,_=evaluate(model,raw,y,train_gpu,mu,sd); best.observe(model,final_train['ce'],step); traces.append([step,CFG['epochs'],-1,len(train),final_train['ce'],1])
            final_dev,dev_logits=evaluate(model,raw,y,dev_gpu,mu,sd)
            if dev_logits.shape!=(len(dev),100) or not np.isfinite(dev_logits).all() or not all(math.isfinite(v) for v in (*final_train.values(),*final_dev.values())): raise FloatingPointError('invalid final endpoint')
            routing=routing_diagnostics(model,raw,dev_gpu,mu,sd)
            logits_ref=writer.save_dataset(f'{seed}-{arm}-dev-logits',dict(ids=dev,labels=labels[dev],logits=dev_logits),provenance=dict(checkpoint='fixed final state',ordered='sorted development IDs',split=data['path']))
            final=writer.save_checkpoint(f'{seed}-{arm}-final',model,arm=arm,seed=seed,selection=dict(metric='fixed final update',step=step),config=config,datasets=refs)
            assert best.observations==len(traces)==step+1 and best.value==min(x[4] for x in traces)
            selected=best.save(writer,f'{seed}-{arm}-training-selected',arm=arm,seed=seed,config=config,datasets=refs)
            trace_ref=writer.save_dataset(f'{seed}-{arm}-trace',dict(observations=np.asarray(traces,dtype=np.float64)),provenance=dict(columns=['step','epoch','order_start','count','ce','scope_0_augmented_minibatch_1_final_canonical_training']))
            if best.step != next(row[0] for row in traces if row[4]==best.value): raise ValueError('selection is not earliest minimum')
            restored=Model(arm); restored.load_state_dict(load_json_state(OUT/final['path']),strict=True); restored.to('cuda').eval()
            replay_metric,replay_logits=evaluate(restored,raw,y,dev_gpu,mu,sd)
            if not np.isfinite(replay_logits).all() or not np.allclose(dev_logits,replay_logits,rtol=1e-5,atol=1e-6): raise ValueError('final dev checkpoint replay mismatch')
            selected_state=load_json_state(OUT/selected['path'])
            if selected_state.keys()!=best.state.keys() or any(not torch.equal(selected_state[k],best.state[k]) for k in best.state): raise ValueError('selected checkpoint state mismatch')
            replay_max_abs=float(np.max(np.abs(dev_logits-replay_logits)))
            del restored,replay_logits
            record=dict(seed=seed,arm=arm,arm_order=list(arms),initial=initial_ref['path'],final=final['path'],selected=selected['path'],trace=trace_ref['path'],dev_logits=logits_ref['path'],presentations=presentations,partial_batch=200,replay_max_abs=replay_max_abs,updates=step,training_seconds=training_seconds,epoch_seconds=epoch_times,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),parameters=sum(p.numel() for p in model.parameters()),final_train=final_train,final_dev=final_dev,routing=routing)
            records.append(record); event('fit_complete',seed=seed,arm=arm,updates=step); del model,optimizer,best
    if len(records)!=len(seeds)*len(ARMS) or {(r['seed'],r['arm']) for r in records}!={(seed,arm) for seed in seeds for arm in ARMS}: raise ValueError('incomplete or duplicate fits')
    if any(r['updates']!=CFG['epochs']*math.ceil(len(train)/CFG['batch']) for r in records): raise ValueError('incorrect update count')
    try: driver=subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True,timeout=10).strip()
    except Exception as e: driver='unavailable: '+repr(e)
    details=dict(config=config,cohort=cohort,cohort_seeds=list(seeds),records=records,artifact_helper_sha256=digest(artifacts.__file__),python_version=sys.version,platform=platform.platform(),numpy_version=np.__version__,datasets_version=datasets.__version__,huggingface_hub_version=huggingface_hub.__version__,cudnn_version=torch.backends.cudnn.version(),device=torch.cuda.get_device_name(),driver=driver,torch_version=torch.__version__,cuda_version=torch.version.cuda,elapsed_seconds=time.monotonic()-START,process_peak_rss_kib_linux=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,disk_free_before=disk_before,disk_free_before_publication=shutil.disk_usage(OUT).free,measurement_scope='GPU peaks include train, evaluation and replay; reserved allocator may carry over across fits; disk sampled at two points, not high-water; publication and teardown excluded')
    publish(OUT/'details.json',details)
    inventory={}
    for path in sorted(OUT.rglob('*')):
        if path.is_file() and path.name not in ('integrity.json','result.json'):
            h=hashlib.sha256()
            with path.open('rb') as f:
                for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
            inventory[path.relative_to(OUT).as_posix()]={'sha256':h.hexdigest(),'bytes':path.stat().st_size}
    publish(OUT/'integrity.json',inventory)
    publish(OUT.parent/'result.json',dict(metrics=dict(training_seconds=sum(r['training_seconds'] for r in records),fits=len(records),peak_cuda_memory_bytes=max(r['peak_allocated_bytes'] for r in records)),summary='Cohort completed; independently audit all three cohorts before interpreting campaign.'))

if __name__=='__main__':
    try: main()
    except Exception:
        if OWNED_OUTPUT:
            try: (OUT/'failure-traceback.txt').write_text(traceback.format_exc())
            except Exception: pass
        raise
