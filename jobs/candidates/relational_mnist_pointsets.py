"""Frozen CUDA relational development campaign; run only through queue.

No test selection, novelty claim, local training, adaptive budget, or fallback.
"""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['HF_HOME'] = '/output/hf-cache'
os.environ['HF_DATASETS_CACHE'] = '/output/hf-cache/datasets'
os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
import copy
import hashlib
import json
import math
import platform
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from neural_lab.artifacts import ArtifactWriter, BestCheckpoint

MODE = 'mnist'
ARMS = ('relational', 'constant_pair', 'centered_DS', 'fixed_RBF', 'centered_aug_dense')
N = 64
BATCH = 256
EPOCHS = 8 if MODE == 'synthetic' else 24
CLASSES = 2 if MODE == 'synthetic' else 10
BANDWIDTHS = torch.exp(torch.linspace(math.log(.0001), math.log(8.), 32))
PAIRS = torch.triu_indices(N, N, offset=1)
OUT = Path('/output')


def digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def finite(tensor, name):
    if tensor is None:
        raise FloatingPointError('missing ' + name)
    check=torch.isfinite(tensor).all()
    if tensor.is_cuda:
        torch._assert_async(check,'nonfinite '+name)
    elif not check.item():
        raise FloatingPointError('nonfinite '+name)


def event(kind, **data):
    row = {'event': kind, 'elapsed_wall_seconds': time.monotonic()-STARTED, **data}
    line = json.dumps(row, allow_nan=False)
    with (OUT/'progress.jsonl').open('a') as handle:
        handle.write(line+'\n')
    print(line, flush=True)


def write_json(name, value):
    target = OUT/name
    temporary = target.with_suffix(target.suffix+'.tmp')
    temporary.write_text(json.dumps(value, allow_nan=False, indent=2))
    temporary.replace(target)


def center(x):
    return torch.cat((x[..., :2]-x[..., :2].mean(1, keepdim=True), x[..., 2:]), -1)


class Model(nn.Module):
    def __init__(self, arm, classes=CLASSES):
        super().__init__()
        self.arm = arm
        self.register_buffer('pair_indices', PAIRS.clone(), persistent=False)
        self.register_buffer('bandwidths', BANDWIDTHS.clone())
        if arm == 'centered_DS':
            self.item = nn.Sequential(nn.Linear(3,96), nn.GELU(), nn.Linear(96,96), nn.GELU())
            self.readout = nn.Sequential(nn.Linear(96,96), nn.GELU(), nn.Linear(96,classes))
        elif arm == 'centered_aug_dense':
            self.input = nn.Linear(192,192)
            self.blocks = nn.ModuleList([nn.Linear(192,192) for _ in range(2)])
            self.output = nn.Linear(192,classes)
        else:
            self.item = nn.Sequential(nn.Linear(3,64), nn.GELU(), nn.Linear(64,64), nn.GELU())
            if arm != 'fixed_RBF':
                self.pair = nn.Sequential(nn.Linear(1,32), nn.GELU(), nn.Linear(32,32), nn.GELU())
            self.readout = nn.Sequential(nn.Linear(96,96), nn.GELU(), nn.Linear(96,classes))

    def forward(self, x):
        if self.arm == 'centered_aug_dense':
            h = F.gelu(self.input(x.flatten(1)))
            for block in self.blocks:
                h = h+F.gelu(block(h))
            return self.output(h)
        item = self.item(x).mean(1)
        if self.arm == 'centered_DS':
            return self.readout(item)
        d = (x[:,self.pair_indices[0],:2]-x[:,self.pair_indices[1],:2]).square().sum(-1,keepdim=True)
        if self.arm == 'constant_pair':
            d = torch.zeros_like(d)
        if self.arm == 'fixed_RBF':
            pair = torch.exp(-d/self.bandwidths).mean(1)
        else:
            pair = self.pair(d).mean(1)
        return self.readout(torch.cat((item,pair),-1))


def make_models(init):
    torch.manual_seed(65101+init)
    reference = Model('relational')
    models = {'relational': reference, 'constant_pair': copy.deepcopy(reference)}
    models['constant_pair'].arm = 'constant_pair'
    torch.manual_seed(65101+init)
    fixed = Model('fixed_RBF')
    fixed.item.load_state_dict(reference.item.state_dict())
    fixed.readout.load_state_dict(reference.readout.state_dict())
    models['fixed_RBF'] = fixed
    for arm, base in [('centered_DS',66101),('centered_aug_dense',67101)]:
        torch.manual_seed(base+init)
        models[arm] = Model(arm)
    return models


def schedule(size, dataset_id):
    generator = torch.Generator().manual_seed(74100+dataset_id)
    before = generator.get_state()
    orders = torch.stack([torch.randperm(size, generator=generator) for _ in range(EPOCHS)])
    after = generator.get_state()
    item_generator = torch.Generator().manual_seed(75100+dataset_id)
    item_before = item_generator.get_state()
    items = torch.stack([torch.randperm(N, generator=item_generator) for _ in range(EPOCHS)])
    probe_generator = torch.Generator().manual_seed(76100+dataset_id)
    probe = torch.stack([torch.randperm(N,generator=probe_generator) for _ in range(8192 if MODE=='synthetic' else 10000)])
    return {'epoch_train_indices': orders, 'epoch_dense_item_permutation': items,
            'schedule_generator_state_initial': before, 'schedule_generator_state_final': after,
            'item_generator_state_initial':item_before,'item_generator_state_final':item_generator.get_state(),
            'dev_probe_indices':probe,'probe_generator_state_final':probe_generator.get_state()}


def synthetic_dataset(regime, draw):
    arrays = {'pair_indices': PAIRS, 'rbf_bandwidths': BANDWIDTHS}
    seen=set()
    for split, size, offset in [('train',32768,0),('dev',8192,1)]:
        seed = 62100+10*draw+offset
        rng = torch.Generator().manual_seed(seed)
        arrays[split+'_generator_state_initial'] = rng.get_state()
        raw = torch.cat((2*torch.rand((size,N,2),generator=rng)-1,torch.ones(size,N,1)), -1)
        x = center(raw)
        for example in raw:
            xy=example[:,:2].numpy()
            ordered=xy[np.lexsort((xy[:,1],xy[:,0]))]
            key=hashlib.sha256(ordered.tobytes()).digest()
            if key in seen:
                raise AssertionError('exact unordered synthetic duplicate across/within splits')
            seen.add(key)
        # Chunk analytic geometry to avoid a full Bx2016x2 temporary on host.
        kth, radii, pair_means = [], [], []
        for chunk in x.split(256):
            d = (chunk[:,PAIRS[0],:2]-chunk[:,PAIRS[1],:2]).square().sum(-1)
            pair_means.append(d.mean(1))
            kth.append(d.kthvalue(8,dim=1).values)
            radii.append(chunk[:,:,:2].square().sum(-1).mean(1))
        kth, radius = torch.cat(kth),torch.cat(radii)
        pair_mean=torch.cat(pair_means)
        if not torch.allclose(pair_mean,(128/63)*radius,atol=1e-5,rtol=0):
            raise AssertionError('pair mean/radius identity failed')
        score = .005-kth if regime=='local' else radius-63/96
        arrays.update({split+'_raw_inputs':raw,split+'_inputs':x,split+'_labels':(score>0).long(),
                       split+'_kth8_squared_distance':kth,split+'_radius':radius,split+'_score':score,
                       split+'_source_row_id':torch.arange(size),split+'_generator_state_final':rng.get_state(),
                       split+'_centroids':raw[:,:,:2].mean(1),split+'_mean_pair_distance_squared':pair_mean,
                       split+'_local_labels':(.005-kth>0).long(),split+'_radius_labels':(radius-63/96>0).long(),
                       split+'_local_score':.005-kth,split+'_radius_score':radius-63/96,
                       split+'_moments':torch.stack((x[:,:,0].square().mean(1),x[:,:,1].square().mean(1),(x[:,:,0]*x[:,:,1]).mean(1)),1)})
    arrays.update(schedule(32768,draw))
    provenance = {'kind':'analytic synthetic', 'regime':regime,'draw':draw,
                  'split_seeds':{sp:62100+10*draw+off for sp,off in [('train',0),('dev',1)]},
                  'shuffle_seed':74100+draw,'dense_permutation_seed':75100+draw,'probe_seed':76100+draw,
                  'generator':'CPU torch.rand train then dev; float32 uniform[-1,1]^2, intensity1; independent split generators',
                  'teacher':'local label=(8th smallest unordered off-diagonal squared distance<.005); radius label=(mean squared centered radius>63/96)',
                  'split':{'train':32768,'dev':8192,'test':'not generated'},
                  'duplicates':'canonical xy row-sort SHA256 exact duplicate check within/across splits, fail on duplicate; no label filtering, balancing, redraw or calibration',
                  'exact_unordered_duplicate_sets':0,'dataset_id':draw}
    return arrays,provenance


def hf_dataset():
    # All network I/O occurs in the queued worker; no executable dataset scripts.
    from datasets import load_dataset
    import datasets
    from huggingface_hub import HfApi
    metadata = HfApi(token=False).dataset_info('ylecun/mnist',revision='main',timeout=30)
    resolved_at=datetime.now(timezone.utc).isoformat()
    revision = metadata.sha
    if not isinstance(revision,str) or len(revision)!=40 or any(c not in '0123456789abcdef' for c in revision):
        raise ValueError('immutable Hugging Face revision unavailable')
    train_file='mnist/train-00000-of-00001.parquet'
    if train_file not in {row.rfilename for row in metadata.siblings}:
        raise ValueError('frozen MNIST train Parquet path absent; no fallback')
    # The built-in builder avoids repository card expectations for the excluded test split.
    train_uri='hf://datasets/ylecun/mnist@'+revision+'/'+train_file
    data = load_dataset('parquet',data_files={'train':train_uri},split='train',token=False,
                        cache_dir='/output/hf-cache/datasets')
    if len(data)!=60000:
        raise ValueError('unexpected MNIST official train length')
    generator = torch.Generator().manual_seed(72100)
    before = generator.get_state()
    row_order = torch.randperm(60000,generator=generator)
    arrays = {'official_train_row_order':row_order,'split_generator_state_initial':before,
              'split_generator_state_final':generator.get_state(),'pair_indices':PAIRS,'rbf_bandwidths':BANDWIDTHS}
    for split, ids in [('train',row_order[:40000]),('dev',row_order[40000:50000])]:
        images = torch.from_numpy(np.stack([np.asarray(data[int(i)]['image'],dtype=np.uint8) for i in ids]))
        labels = torch.tensor([data[int(i)]['label'] for i in ids],dtype=torch.long)
        flat = images.flatten(1)
        # stable descending sort keeps row-major order for exact intensity ties.
        pixels = torch.argsort(flat,dim=1,descending=True,stable=True)[:,:N]
        xy = torch.stack(((pixels%28).float()/27*2-1,(pixels//28).float()/27*2-1),-1)
        intensity = flat.gather(1,pixels).float().unsqueeze(-1)/255
        raw = torch.cat((xy,intensity),-1)
        arrays.update({split+'_raw_images_uint8':images,split+'_source_row_id':ids,
                       split+'_top64_pixel_indices':pixels,split+'_raw_inputs':raw,
                       split+'_inputs':center(raw),split+'_labels':labels,
                       split+'_centroids':raw[:,:,:2].mean(1)})
    arrays.update(schedule(40000,2))
    train_image_hashes={}
    for image,label in zip(arrays['train_raw_images_uint8'],arrays['train_labels']):
        train_image_hashes.setdefault(digest(image),set()).add(int(label))
    duplicate_dev_rows=conflict_dev_rows=0
    for image,label in zip(arrays['dev_raw_images_uint8'],arrays['dev_labels']):
        labels=train_image_hashes.get(digest(image))
        if labels is not None:
            duplicate_dev_rows+=1
            conflict_dev_rows+=int(any(other!=int(label) for other in labels))
    cache_files=[]
    for row in data.cache_files:
        filename=Path(row['filename'])
        filehash=hashlib.sha256()
        with filename.open('rb') as handle:
            for chunk in iter(lambda:handle.read(1024*1024),b''):
                filehash.update(chunk)
        cache_files.append({'filename':str(filename),'bytes':filename.stat().st_size,'sha256':filehash.hexdigest()})
    card = metadata.card_data
    provenance = {'kind':'huggingface','repo_id':'ylecun/mnist','requested_revision':'main','resolved_revision':revision,'resolved_at_utc':resolved_at,
                  'explicit_data_files':{'train':train_file},
                  'loader':'built-in parquet; default verification retained','immutable_train_uri':train_uri,
                  'config':data.info.config_name,'loaded_split':'train','official_test':'never loaded or evaluated',
                  'url':'https://huggingface.co/datasets/ylecun/mnist/tree/'+revision,
                  'license':card.get('license') if card else None,'authentication':'public token=False',
                  'dataset_fingerprint':data._fingerprint,'builder_name':data.info.builder_name,
                  'cache_files':cache_files,'source_download_checksums':data.info.download_checksums,
                  'source_hash_note':'builder download checksums recorded where supplied; saved image-array hashes are authoritative',
                  'exact_image_duplicate_dev_rows_with_train_match':duplicate_dev_rows,
                  'duplicate_dev_rows_with_label_conflict':conflict_dev_rows,'datasets_version':datasets.__version__,
                  'cache':'/output/hf-cache/datasets','split_seed':72100,'shuffle_seed':74102,'dense_permutation_seed':75102,'probe_seed':76102,'dataset_id':2,
                  'split':'seed72100 CPU randperm(60000), first40000 train,next10000 dev,last10000 unused',
                  'preprocessing':'stable descending intensity top64 pixels, rowmajor ties; xy=[2col/27-1,2row/27-1]; intensity=pixel/255; center xy across64, retain intensities',
                  'limits':'top64 intensity sets may discard low-intensity stroke pixels; zero pixels retained when fewer than64 nonzero; fixed deterministic representation, no fitted feature extraction',
                  'features':data.features.to_dict()}
    return arrays,provenance


def save_data(writer,label,arrays,provenance,source_sha):
    for split in ('train','dev'):
        spot=arrays[split+'_inputs'][:4]
        d=(spot[:,PAIRS[0],:2]-spot[:,PAIRS[1],:2]).square().sum(-1)
        independent=torch.cdist(spot[:,:,:2],spot[:,:,:2]).square()[:,PAIRS[0],PAIRS[1]]
        assert torch.allclose(d,independent,atol=1e-5,rtol=0)
        arrays[split+'_geometry_spot_distances']=d
        arrays[split+'_geometry_spot_rbf_mean']=torch.exp(-d[:,:,None]/BANDWIDTHS).mean(1)
    for name,value in arrays.items():
        finite(value,name)
    provenance.update({'source_sha256':source_sha,'array_sha256':{k:digest(v) for k,v in arrays.items()},
                       'class_counts':{sp:arrays[sp+'_labels'].bincount(minlength=CLASSES).tolist() for sp in ('train','dev')},
                       'schedule':'independent CPU generators: shuffle74100+dataset_id, dense item75100+id, dev probe76100+id; exact saved schedules reused arms/inits and synthetic regimes',
                       'pair_geometry':'2016 unordered offdiagonal pairs i<j lexicographic; squared xy distance only; uniform item/pair mean',
                       'software':{'torch':str(torch.__version__),'numpy':np.__version__,'generation_device':'cpu'}})
    entry = writer.save_dataset(label,arrays,provenance=provenance)
    with np.load(writer.output/entry['path'],allow_pickle=False) as saved:
        assert set(saved.files)==set(arrays)
        for name,tensor in arrays.items():
            assert hashlib.sha256(saved[name].tobytes()).hexdigest()==digest(tensor)
    for split in ('train','dev'):
        spot=arrays[split+'_inputs'][:4].to('cuda:0')
        d=(spot[:,PAIRS[0].to('cuda:0'),:2]-spot[:,PAIRS[1].to('cuda:0'),:2]).square().sum(-1)
        assert torch.allclose(d.cpu(),arrays[split+'_geometry_spot_distances'],atol=1e-5,rtol=0)
        rbf=torch.exp(-d[:,:,None]/BANDWIDTHS.to('cuda:0')).mean(1)
        assert torch.allclose(rbf.cpu(),arrays[split+'_geometry_spot_rbf_mean'],atol=1e-5,rtol=0)
    event('dataset_saved',label=label,path=entry['path'],sha256=entry['sha256'],saved_array_hashes_verified=True,geometry_cpu_cuda_spot_equality=True)
    return entry


@torch.no_grad()
def evaluate(model,x,y,device):
    total_ce,total_correct,total_squared,total_values = 0.,0,0.,0
    logits_cpu=[]
    for start in range(0,len(x),BATCH):
        inputs=x[start:start+BATCH].to(device)
        labels=y[start:start+BATCH].to(device)
        logits=model(inputs)
        finite(logits,'endpoint logits')
        loss=F.cross_entropy(logits,labels,reduction='sum')
        finite(loss,'endpoint CE')
        total_ce+=loss.item()
        total_correct+=(logits.argmax(1)==labels).sum().item()
        total_squared+=logits.square().sum().item()
        total_values+=logits.numel()
        logits_cpu.append(logits.cpu())
    return {'loss':total_ce/len(x),'accuracy':total_correct/len(x),
            'logit_rms':math.sqrt(total_squared/total_values)},torch.cat(logits_cpu)


def fit(model,arrays,writer,entry,arm,seed,fit_id,source_sha,device):
    model=model.to(device)
    assert all(p.is_cuda for p in model.parameters())
    initial_hash={k:digest(v) for k,v in model.state_dict().items()}
    def activation_hook(module,args,output):
        if isinstance(output,torch.Tensor):
            finite(output,'activation '+type(module).__name__)
    hooks=[module.register_forward_hook(activation_hook) for module in model.modules()]
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001,betas=(.9,.999),eps=1e-8)
    best=BestCheckpoint(scope='minimum over every observed training CE: initial/final canonical full-training CE and pre-update minibatch CE; unequal batch samples, no claim minimum full empirical risk; strict earliest tie; no held-out selection')
    observations=[]
    def observe(value,step,view,epoch=None,batch=None):
        best.observe(model,value,step)
        observations.append({'step':step,'view':view,'loss':value,'epoch':epoch,'batch':batch,'example_count':len(arrays['train_labels']) if batch is None else min(BATCH,len(arrays['train_labels'])-batch*BATCH)})
    model.eval()
    initial,_=evaluate(model,arrays['train_inputs'],arrays['train_labels'],device)
    observe(initial['loss'],0,'full_train_initial')
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started=time.monotonic()
    updates=0
    epoch_summaries=[]
    for epoch in range(EPOCHS):
        model.train()
        order=arrays['epoch_train_indices'][epoch]
        item_order=arrays['epoch_dense_item_permutation'][epoch].to(device)
        epoch_losses=[]
        for batch,start in enumerate(range(0,len(order),BATCH)):
            indices=order[start:start+BATCH]
            x=arrays['train_inputs'][indices].to(device)
            y=arrays['train_labels'][indices].to(device)
            if arm=='centered_aug_dense':
                x=x[:,item_order]
            optimizer.zero_grad(set_to_none=True)
            logits=model(x)
            finite(logits,'training logits')
            loss=F.cross_entropy(logits,y)
            finite(loss,'training CE')
            value=loss.item()
            observe(value,updates,'pre_update_minibatch',epoch,batch)
            epoch_losses.append(value)
            loss.backward()
            for name,p in model.named_parameters():
                finite(p.grad,'gradient '+name)
            optimizer.step()
            for name,p in model.named_parameters():
                finite(p,'weight '+name)
            updates+=1
        row={'epoch':epoch+1,'mean_observed_minibatch_ce':sum(epoch_losses)/len(epoch_losses),'updates':updates}
        epoch_summaries.append(row)
        event('epoch_complete',fit_id=fit_id,arm=arm,**row)
    torch.cuda.synchronize(device)
    training_seconds=time.monotonic()-started
    peak_memory=torch.cuda.max_memory_allocated(device)
    peak_reserved=torch.cuda.max_memory_reserved(device)
    assert updates==(1024 if MODE=='synthetic' else 3768)
    model.eval()
    final_train,_=evaluate(model,arrays['train_inputs'],arrays['train_labels'],device)
    observe(final_train['loss'],updates,'full_train_final')
    # Independent final endpoint before any selected-state reload; no dev read above.
    final_dev,original_logits=evaluate(model,arrays['dev_inputs'],arrays['dev_labels'],device)
    perm=arrays['dev_probe_indices']
    permuted_inputs=arrays['dev_inputs'].gather(1,perm[:,:,None].expand(-1,-1,3))
    permuted_metrics,permuted_logits=evaluate(model,permuted_inputs,arrays['dev_labels'],device)
    difference=(original_logits-permuted_logits).abs()
    invariance={'permutation':'saved independent per-dev-row permutation indices','max_absolute_logit_difference':difference.max().item(),
                'mean_absolute_logit_difference':difference.mean().item(),
                'prediction_disagreement':(original_logits.argmax(1)!=permuted_logits.argmax(1)).float().mean().item(),
                'dev_ce_change':permuted_metrics['loss']-final_dev['loss']}
    if arm!='centered_aug_dense' and invariance['max_absolute_logit_difference']>1e-4:
        raise AssertionError('pool permutation invariance failed')
    cfg={'mode':MODE,'arm':arm,'source_sha256':source_sha,'fit_id':fit_id,'seed':seed,
         'classes':CLASSES,'points':64,'input_dim':3,'item_width':64,'pair_width':32,
         'architectures':{'hybrid':'item3,64,64 GELU; pair1,32,32 GELU; means concat96; rho96,96,classes GELU',
                          'centered_DS':'item3,96,96 GELU; mean; rho96,96,classes GELU',
                          'centered_aug_dense':'flatten192; Linear192,192 GELU;2 residual Linear192,192 GELU; Linear192,classes'},
         'preprocessing':'saved centered xy and unchanged intensity; pair squared xy distances; constant_pair zeros pair inputs; RBF exp(-D/bandwidth),32 logspace1e-4..8',
         'initial_state_sha256':initial_hash,'epochs':EPOCHS,'batch_size':BATCH,'updates':updates,
         'optimizer':{'name':'AdamW','lr':.001,'weight_decay':.0001,'betas':[.9,.999],'eps':1e-8},
         'dataset_path':entry['path'],'dataset_sha256':entry['sha256'],'device':str(device),
         'dataset_manifest_sha256_at_fit_completion':hashlib.sha256((OUT/'artifacts.json').read_bytes()).hexdigest(),
         'training_selection':best.scope,'endpoint':'independent final scheduled epoch dev, no selected checkpoint reload'}
    checkpoint=best.save(writer,fit_id+'-'+arm+'-best',arm=arm,seed=seed,config=cfg,datasets=[entry['path']])
    assert best.observations==updates+2==len(observations)
    minimum=min(row['loss'] for row in observations)
    assert checkpoint['selection']['value']==minimum
    earliest=next(row for row in observations if row['loss']==minimum)
    result={'config':cfg,'parameters':sum(p.numel() for p in model.parameters()),
            'initial_training_metrics':initial,'final_training_metrics':final_train,'final_dev_metrics':final_dev,
            'dev_minus_train_ce':final_dev['loss']-final_train['loss'],'permutation':invariance,
            'observations':observations,'epoch_summaries':epoch_summaries,
            'checkpoint':checkpoint,'selected_observation':earliest,'training_seconds':training_seconds,
            'peak_cuda_memory_bytes':peak_memory,'peak_cuda_reserved_bytes':peak_reserved,'all_parameters_on_cuda':True}
    write_json(fit_id+'-'+arm+'-fit.json',result)
    event('fit_complete',fit_id=fit_id,arm=arm,training_seconds=training_seconds,
          peak_cuda_memory_bytes=peak_memory,final_dev_ce=final_dev['loss'],checkpoint=checkpoint['path'])
    for hook in hooks:
        hook.remove()
    model.cpu()
    del optimizer
    torch.cuda.empty_cache()
    return result


def summarize(records):
    groups=sorted(set(r['dataset_label'] for r in records))
    summaries={}
    contrasts={}
    for group in groups:
        rows=[r for r in records if r['dataset_label']==group]
        summaries[group]={arm:{metric:sum(r['arms'][arm]['final_dev_metrics'][metric] for r in rows)/len(rows)
                               for metric in ('loss','accuracy')} for arm in ARMS}
        contrasts[group]={arm:{'init_differences':[r['arms']['relational']['final_dev_metrics']['loss']-r['arms'][arm]['final_dev_metrics']['loss'] for r in rows],
                               'mean':summaries[group]['relational']['loss']-summaries[group][arm]['loss']} for arm in ARMS[1:]}
    return summaries,contrasts



def screen(summaries,contrasts):
    criteria={}
    grouped={}
    for regime in (('local','radius') if MODE=='synthetic' else ('mnist',)):
        labels=[key for key in summaries if key.startswith(regime+'-')]
        grouped[regime]={arm:{'dataset_means':[contrasts[label][arm]['mean'] for label in labels],
                             'mean':sum(contrasts[label][arm]['mean'] for label in labels)/len(labels),
                             'init_differences':[value for label in labels for value in contrasts[label][arm]['init_differences']]}
                         for arm in ARMS[1:]}
    specs=([('local','constant_pair',-.01,True),('local','centered_DS',-.01,True),
            ('local','fixed_RBF',.01,False),('local','centered_aug_dense',.01,False),
            ('radius','centered_DS',.02,False),('radius','centered_aug_dense',.02,False)]
           if MODE=='synthetic' else [('mnist','constant_pair',-.005,True),('mnist','centered_DS',-.005,True),
                                     ('mnist','fixed_RBF',.01,False),('mnist','centered_aug_dense',.01,False)])
    for regime,arm,margin,require_negative in specs:
        row=grouped[regime][arm]
        negatives=row['dataset_means'] if MODE=='synthetic' else row['init_differences']
        criteria[regime+'_versus_'+arm]={'mean_ce_difference':row['mean'],'maximum_allowed':margin,
            'require_each_dataset_negative' if MODE=='synthetic' else 'require_each_seed_negative':require_negative,
            'passed':row['mean']<=margin and (not require_negative or all(value<0 for value in negatives))}
    return criteria,grouped


def main():
    global STARTED
    STARTED=time.monotonic()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; CPU training is forbidden by frozen protocol')
    device=torch.device('cuda:0')
    if 'RTX 2070' not in torch.cuda.get_device_name(device):
        raise RuntimeError('frozen target requires RTX 2070, got '+torch.cuda.get_device_name(device))
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(70100)
    torch.cuda.manual_seed_all(70100)
    writer=ArtifactWriter('/output')
    source_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    try:
        driver_query=subprocess.run(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],
                                    capture_output=True,text=True,timeout=5,check=True)
        driver={'value':driver_query.stdout.strip(),'source':'nvidia-smi --query-gpu=driver_version'}
        if not driver['value']:
            raise ValueError('empty driver query response')
    except (OSError,ValueError,subprocess.SubprocessError) as error:
        driver={'value':None,'unavailable_reason':type(error).__name__+': '+str(error)}
    environment={'python':platform.python_version(),'torch':str(torch.__version__),'numpy':np.__version__,
                 'cuda_version':torch.version.cuda,'driver_version':driver,
                 'device_properties':str(torch.cuda.get_device_properties(device)),'device':str(device),'device_name':torch.cuda.get_device_name(device),
                 'device_total_memory_bytes':torch.cuda.get_device_properties(device).total_memory,
                 'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),'tf32':False,'threads':4}
    write_json('environment.json',environment)
    event('job_start',mode=MODE,source_sha256=source_sha,environment=environment)
    records=[]
    datasets=[(regime,draw) for regime in ('local','radius') for draw in range(2)] if MODE=='synthetic' else [('mnist',0)]
    for index,(regime,draw) in enumerate(datasets):
        label=regime+'-draw-'+str(draw)
        arrays,provenance=synthetic_dataset(regime,draw) if MODE=='synthetic' else hf_dataset()
        entry=save_data(writer,label,arrays,provenance,source_sha)
        for init in range(2):
            models=make_models(init)
            row={'dataset_label':label,'regime':regime,'draw':draw,'init':init,'dataset':entry,'arms':{}}
            order=ARMS if (index*2+init)%2==0 else tuple(reversed(ARMS))
            row['arm_order']=order
            for arm in order:
                fit_id=label+'-init-'+str(init)
                seed=(66101 if arm=='centered_DS' else 67101 if arm=='centered_aug_dense' else 65101)+init
                row['arms'][arm]=fit(models.pop(arm),arrays,writer,entry,arm,seed,fit_id,source_sha,device)
            records.append(row)
            write_json('completed-records.json',records)
        del arrays
    expected_fits=40 if MODE=='synthetic' else 10
    assert sum(len(row['arms']) for row in records)==expected_fits
    assert len(writer.manifest['checkpoints'])==expected_fits
    summaries,contrasts=summarize(records)
    criteria,regime_contrasts=screen(summaries,contrasts)
    metrics={'screening_criterion_met':int(all(row['passed'] for row in criteria.values()))}
    for arm in ARMS:
        metrics[arm+'_mean_dev_ce']=sum(v[arm]['loss'] for v in summaries.values())/len(summaries)
        metrics[arm+'_mean_dev_accuracy']=sum(v[arm]['accuracy'] for v in summaries.values())/len(summaries)
    metrics['training_seconds']=sum(f['training_seconds'] for r in records for f in r['arms'].values())
    metrics['peak_cuda_memory_bytes']=max(f['peak_cuda_memory_bytes'] for r in records for f in r['arms'].values())
    metrics['elapsed_seconds']=time.monotonic()-STARTED
    details={'status':'completed','mode':MODE,'source_sha256':source_sha,'environment':environment,
             'records':records,'summaries':summaries,'paired_relational_minus_control_dev_ce':contrasts,
             'regime_contrasts':regime_contrasts,'frozen_criteria':criteria,
             'screening_criterion_met':bool(metrics['screening_criterion_met']),
             'artifact_manifest_sha256':hashlib.sha256((OUT/'artifacts.json').read_bytes()).hexdigest(),
             'expected_fits':expected_fits,'runtime_estimate':'uncalibrated10-20min per job; actual timing reported, no padding or adaptive budget',
             'endpoint':'independent final fixed epoch dev; held-out development data only, official test untouched',
             'limitations':'Established relation/distance pooling and DeepSets; no novelty claim. Two initializations on each dataset are optimization repeats. Synthetic targets deliberately geometry aligned. Fixed RBF unequal scale/parameters; constant branch unequal effective capacity; equal epochs are unequal compute. MNIST top64 intensity representation, development only; no test, tuning, significance, causal mechanism, convergence or general superiority claim. Checkpoint minima include unequal minibatches and are retention artifacts, not primary endpoints.',
             'elapsed_seconds':metrics['elapsed_seconds']}
    write_json('details.json',details)
    summary=f'Frozen {MODE} CUDA relational campaign: five arms, two initialization seeds, fixed {EPOCHS} epochs. Exact data/schedules and all-observed training-CE selected weights retained; independent final dev endpoints; no test/tuning/promotion.'
    result={'metrics':metrics,'summary':summary}
    assert len(metrics)<=32 and all(math.isfinite(v) for v in metrics.values())
    write_json('result.json',result)
    event('job_complete',metrics=metrics)


if __name__=='__main__':
    try:
        main()
    except Exception:
        # Queue cleanup discards container stderr; preserve the original traceback.
        try:
            OUT.mkdir(parents=True,exist_ok=True)
            (OUT/'failure-traceback.txt').write_text(traceback.format_exc())
        except Exception:
            pass
        raise
