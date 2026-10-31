"""Frozen CUDA-only scientific campaign; timing-derived epoch budget.

Established dynamic-convolution/ConvMixer family; no architecture invention claim.
Epoch budget frozen from operational calibration before scientific development endpoints.
"""
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
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from neural_lab.artifacts import ArtifactWriter, BestCheckpoint


MODE = 'campaign'
BUDGET_LABEL = 'short'
CAMPAIGN = dict(epochs=9, seeds=[1701, 1702], split_seed=20261030, protocol='native-CIFAR conditional-bank fixed-budget short comparison; planning-cifar-20261030.md; frozen before scientific endpoints from calibration119a7704757740b4a92db73b5305a90b timing only: epochs9/13, useful622.724/899.490s, conservative total1014.192/1346.312s including20percent train overhead,2S evaluation,1.5P preparation,60seconds reserve,4A serialization; original10/15 rejected forecast1512greater1500', criteria=dict(dynamic_static_ce_max=-.01, both_dynamic_static_seeds_negative=True, dynamic_learned_ce_max=.02, dynamic_cnn_ce_max=.02))
ARMS = ('dynamic', 'static', 'learned', 'cnn')
WIDTH, DEPTH, KERNEL, PATCH, BATCH = 64, 4, 5, 2, 256
OUT = Path('/output')
START = time.monotonic()


def write(name, value):
    p = OUT / name
    p.write_text(json.dumps(value, indent=2, allow_nan=False))


def event(kind, **values):
    row = dict(event=kind, elapsed_seconds=time.monotonic()-START, **values)
    with (OUT/'progress.jsonl').open('a') as f:
        f.write(json.dumps(row, allow_nan=False)+'\n')
    print(json.dumps(row), flush=True)


class FixedBank(nn.Module):
    def __init__(self, static=False):
        super().__init__()
        self.static = static
        self.bank = nn.Conv2d(WIDTH, 2*WIDTH, KERNEL, padding=KERNEL//2, groups=WIDTH)
        for p in self.bank.parameters():
            p.requires_grad_(False)
        self.gate = nn.Sequential(nn.Linear(WIDTH, 16), nn.GELU(), nn.Linear(16, 2*WIDTH))

    def forward(self, x):
        pooled = x.mean((2, 3))
        if self.static:
            pooled = torch.ones_like(pooled)
        weights = self.gate(pooled).reshape(len(x), WIDTH, 2).softmax(2)
        bank = self.bank(x).reshape(len(x), WIDTH, 2, x.shape[2], x.shape[3])
        return (bank*weights[:, :, :, None, None]).sum(2)


class MixerBlock(nn.Module):
    def __init__(self, arm):
        super().__init__()
        self.dw = FixedBank(arm == 'static') if arm in ('dynamic', 'static') else nn.Conv2d(WIDTH, WIDTH, KERNEL, padding=KERNEL//2, groups=WIDTH)
        self.spatial = nn.Sequential(nn.GELU(), nn.BatchNorm2d(WIDTH))
        self.channel = nn.Sequential(nn.Conv2d(WIDTH, WIDTH, 1), nn.GELU(), nn.BatchNorm2d(WIDTH))

    def forward(self, x):
        return self.channel(x+self.spatial(self.dw(x)))


class Model(nn.Module):
    def __init__(self, arm):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, WIDTH, PATCH, stride=PATCH), nn.GELU(), nn.BatchNorm2d(WIDTH))
        if arm == 'cnn':
            self.blocks = nn.Sequential(*[nn.Sequential(nn.Conv2d(WIDTH, WIDTH, 3, padding=1), nn.GELU(), nn.BatchNorm2d(WIDTH)) for _ in range(DEPTH)])
        else:
            self.blocks = nn.Sequential(*[MixerBlock(arm) for _ in range(DEPTH)])
        self.head = nn.Linear(WIDTH, 10)

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)).mean((2, 3)))


def schedule(seed, epochs, n):
    order, crop, flip = [], [], []
    for epoch in range(epochs):
        rng = np.random.default_rng(seed+100000+epoch*1000003)
        order.append(rng.permutation(n).astype(np.int32))
        crop.append(rng.integers(0, 9, (n, 2), dtype=np.int16))
        flip.append(rng.integers(0, 2, n, dtype=np.uint8))
    order, crop, flip = np.stack(order), np.stack(crop), np.stack(flip)
    return order, crop, flip


def inputs(raw, ids, mean, std, crop=None, flip=None):
    x = raw[ids].float().div_(255)
    if crop is not None:
        x = F.pad(x, (4, 4, 4, 4), mode='reflect')
        rows = torch.arange(32, device='cuda')[None, :]+crop[:, 0, None]
        cols = torch.arange(32, device='cuda')[None, :]+crop[:, 1, None]
        x = x[torch.arange(len(ids), device='cuda')[:, None, None], :, rows[:, :, None], cols[:, None, :]].permute(0, 3, 1, 2)
        x = torch.where(flip[:, None, None, None].bool(), x.flip(-1), x)
    return (x-mean)/std


@torch.no_grad()
def evaluate(model, raw, labels, ids, mean, std):
    model.eval()
    loss, correct = 0., 0
    for start in range(0, len(ids), BATCH):
        ix = ids[start:start+BATCH]
        logits = model(inputs(raw, ix, mean, std))
        loss += F.cross_entropy(logits, labels[ix], reduction='sum').item()
        correct += (logits.argmax(1)==labels[ix]).sum().item()
    return dict(ce=loss/len(ids), accuracy=correct/len(ids))


def main():
    if MODE not in ('calibration', 'campaign'):
        raise ValueError('unknown mode')
    if MODE == 'campaign' and (not isinstance(CAMPAIGN, dict) or not CAMPAIGN.get('protocol') or not CAMPAIGN.get('criteria')):
        raise ValueError('campaign requires reviewed frozen protocol and criteria')
    cfg = dict(epochs=2, seeds=[1701], split_seed=20261030) if MODE == 'calibration' else CAMPAIGN
    if type(cfg['epochs']) is not int or cfg['epochs'] < 2:
        raise ValueError('epochs must be prospectively frozen from calibration timing before queueing')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA mandatory; no CPU fallback')
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    writer = ArtifactWriter('/output')
    source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    event('prepare_start', mode=MODE, configuration=cfg, source_sha256=source_sha)
    import datasets
    import huggingface_hub
    from huggingface_hub import HfApi, hf_hub_download
    revision = '0b2714987fa478483af9968de7c934580d0bb9a2'
    metadata = HfApi(token=False).dataset_info('uoft-cs/cifar10', revision=revision, timeout=20)
    if metadata.sha != revision:
        raise ValueError('immutable HF revision mismatch')
    local = hf_hub_download(repo_id='uoft-cs/cifar10', repo_type='dataset', filename='plain_text/train-00000-of-00001.parquet', revision=revision, token=False, local_dir=str(OUT/'hf-source'))
    if hashlib.sha256(Path(local).read_bytes()).hexdigest() != '8428b53a88a11ac374111006708df51469e315a22ac6d66470afd9c78d2ae883':
        raise ValueError('official train parquet hash mismatch')
    source = writer.register_file('datasets', local, 'cifar10-official-train', format='parquet', source=dict(repo_id='uoft-cs/cifar10', revision=revision, url='https://huggingface.co/datasets/uoft-cs/cifar10/tree/'+revision, creator_provenance_url='https://www.cs.toronto.edu/~kriz/cifar.html', filename='plain_text/train-00000-of-00001.parquet', authentication='public token=False', license=metadata.card_data.get('license') if metadata.card_data else None, datasets_version=datasets.__version__, huggingface_hub_version=huggingface_hub.__version__))
    ds = datasets.load_dataset('parquet', data_files={'train':local}, split='train', cache_dir=str(OUT/'hf-cache'/'decoded'))
    if len(ds) != 50000:
        raise ValueError('expected complete official train pool')
    images = np.stack([np.asarray(row['img'].convert('RGB'), dtype=np.uint8) for row in ds])
    labels = np.asarray(ds['label'], dtype=np.int64)
    if images.shape != (50000, 32, 32, 3) or not np.array_equal(np.bincount(labels), np.full(10, 5000)):
        raise ValueError('unexpected native CIFAR10 layout/class counts')
    rng = np.random.default_rng(cfg['split_seed'])
    train, dev = [], []
    for c in range(10):
        ids = rng.permutation(np.flatnonzero(labels==c))
        train.extend(ids[:4500]); dev.extend(ids[4500:])
    train, dev = np.asarray(sorted(train)), np.asarray(sorted(dev))
    pixels = images[train].astype(np.float64)/255
    mean, std = pixels.mean((0, 1, 2)), pixels.std((0, 1, 2))
    del pixels
    data = writer.save_dataset('cifar10-native-split', dict(images=images, labels=labels, train_ids=train, dev_ids=dev, mean=mean, std=std), provenance=dict(source=source, split_seed=cfg['split_seed'], split_generator='numpy PCG64; independent per-class permutations in class order0..9', numpy_version=np.__version__, official_test='never requested', normalization='population mean/std from train RGB only', augmentation='reflect pad4 random crop32 and horizontal flip; exact schedules retained'))
    raw = torch.from_numpy(images.transpose(0, 3, 1, 2).copy()).to('cuda')
    y = torch.from_numpy(labels).to('cuda')
    mu = torch.tensor(mean, dtype=torch.float32, device='cuda')[None, :, None, None]
    sd = torch.tensor(std, dtype=torch.float32, device='cuda')[None, :, None, None]
    train_gpu, dev_gpu = torch.tensor(train, device='cuda'), torch.tensor(dev, device='cuda')
    config = dict(budget_label=BUDGET_LABEL, bank_filters_per_channel=2, gate_hidden=16, gate='64->16 GELU->128 softmax2/channel; dynamic pooled block input versus static allones', architecture='patch stem GELU BN;4 residual spatial/GELU/BN + pointwise/GELU/BN; meanpool linear10; CNN4 ordinary3x3/GELU/BN', shared_initialization='all common name/shape tensors copied from dynamic base; dynamic/static exact states copied', width=WIDTH, depth=DEPTH, kernel=KERNEL, patch=PATCH, batch=BATCH, optimizer='AdamW lr=.001 weight_decay=.01 constant', normalization='BatchNorm', precision='float32 deterministic; TF32 disabled', **cfg)
    preparation_seconds = time.monotonic()-START
    records = []
    for seed in cfg['seeds']:
        order, crop, flip = schedule(seed, cfg['epochs'], len(train))
        sched = writer.save_dataset(f'schedule-{seed}', dict(order=order, crop=crop, flip=flip), provenance=dict(seed=seed, generator='numpy PCG64 epoch seed=initialization_seed+100000+epoch*1000003; order,crop,flip generated in that order', numpy_version=np.__version__, indexing='epoch,position; order indexes sorted train_ids', shared_arms=list(ARMS)))
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        base = Model('dynamic')
        initial = copy.deepcopy(base.state_dict())
        arm_order = ARMS if seed == 1701 else tuple(reversed(ARMS))
        event('arm_order', seed=seed, arms=list(arm_order))
        for arm in arm_order:
            torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            model = Model(arm)
            if arm in ('dynamic', 'static'):
                model.load_state_dict(initial, strict=True)
            else:
                state = model.state_dict()
                for name, value in initial.items():
                    if name in state and state[name].shape == value.shape:
                        state[name] = value.clone()
                model.load_state_dict(state, strict=True)
            model = model.to('cuda')
            assert all(t.is_cuda for t in list(model.parameters())+list(model.buffers()))
            assert all(torch.isfinite(t).all().item() for t in model.state_dict().values())
            assert raw.is_cuda and y.is_cuda and mu.is_cuda and sd.is_cuda
            optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=.001, weight_decay=.01)
            best = BestCheckpoint(scope='minimum over every observed augmented training minibatch CE and final canonical training inference CE; strict earliest tie; minibatch state after BN forward buffer updates before optimizer update; mixed scopes and unequal minibatches, not minimum empirical risk')
            traces, epochs = [], []
            frozen_before = {k: v.clone() for k, v in model.state_dict().items() if '.dw.bank.' in k} if arm in ('dynamic', 'static') else {}
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize(); started = time.monotonic()
            step = 0
            for epoch in range(cfg['epochs']):
                model.train(); torch.cuda.synchronize(); tick = time.monotonic()
                for start in range(0, len(train), BATCH):
                    stop = min(start+BATCH, len(train))
                    ix = train_gpu[torch.tensor(order[epoch, start:stop].astype(np.int64), device='cuda')]
                    c = torch.tensor(crop[epoch, start:stop].astype(np.int64), device='cuda')
                    f = torch.tensor(flip[epoch, start:stop], device='cuda')
                    optimizer.zero_grad(set_to_none=True)
                    loss = F.cross_entropy(model(inputs(raw, ix, mu, sd, c, f)), y[ix])
                    value = float(loss.detach())
                    if not math.isfinite(value):
                        raise FloatingPointError('nonfinite training loss')
                    best.observe(model, value, step)  # BEFORE backward/optimizer; BN buffers belong to this observation.
                    traces.append([step, epoch, start, stop-start, value, 0])
                    loss.backward()
                    for p in model.parameters():
                        if p.grad is not None:
                            torch._assert_async(torch.isfinite(p.grad).all(), 'nonfinite gradient')
                    optimizer.step(); step += 1
                torch.cuda.synchronize(); seconds = time.monotonic()-tick
                epochs.append(seconds)
                event('epoch_complete', seed=seed, arm=arm, epoch=epoch+1, seconds=seconds, updates=step)
            torch.cuda.synchronize(); training_seconds = time.monotonic()-started
            assert all(torch.equal(model.state_dict()[k], v) for k, v in frozen_before.items())
            artifact_started = time.monotonic()
            assert all(torch.isfinite(t).all().item() for t in model.state_dict().values())
            final_train = evaluate(model, raw, y, train_gpu, mu, sd)
            best.observe(model, final_train['ce'], step)
            traces.append([step, cfg['epochs'], -1, len(train), final_train['ce'], 1])
            final_dev = evaluate(model, raw, y, dev_gpu, mu, sd) if MODE == 'campaign' else None
            # Final endpoint is measured before any selected checkpoint could be restored.
            refs = [source['path'], data['path'], sched['path']]
            final = writer.save_checkpoint(f'{seed}-{arm}-final', model, arm=arm, seed=seed, selection=dict(metric='fixed final update', step=step), config=config, datasets=refs)
            assert best.observations == step+1 == len(traces)
            assert best.value == min(row[4] for row in traces)
            assert best.step == next(row[0] for row in traces if row[4] == best.value)
            selected = best.save(writer, f'{seed}-{arm}-training-selected', arm=arm, seed=seed, config=config, datasets=refs)
            writer.save_dataset(f'{seed}-{arm}-loss-trace', dict(observations=np.asarray(traces, dtype=np.float64)), provenance=dict(columns=['step', 'epoch', 'order_start', 'batch_count', 'training_ce', 'scope_code_0_augmented_training_minibatch_1_final_canonical_training_inference'], selected=selected['path']))
            record = dict(seed=seed, arm=arm, training_seconds=training_seconds, evaluation_artifact_seconds=time.monotonic()-artifact_started, steady_epoch_seconds=epochs[-1], epoch_seconds=epochs, updates=step, examples=len(train)*cfg['epochs'], peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(), parameters=sum(p.numel() for p in model.parameters()), trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad), final_train=final_train, final_dev=final_dev, dev_minus_train_ce=final_dev['ce']-final_train['ce'], dev_minus_train_accuracy=final_dev['accuracy']-final_train['accuracy'], arm_order=list(arm_order), final=final['path'], selected=selected['path'])
            records.append(record); event('fit_complete', **record)
            del optimizer, model, best
    means = {arm: {metric: sum(r['final_dev'][metric] for r in records if r['arm']==arm)/len(cfg['seeds']) for metric in ('ce', 'accuracy')} for arm in ARMS}
    paired = {control: [next(r['final_dev']['ce'] for r in records if r['seed']==seed and r['arm']=='dynamic')-next(r['final_dev']['ce'] for r in records if r['seed']==seed and r['arm']==control) for seed in cfg['seeds']] for control in ('static', 'learned', 'cnn')}
    contrasts = {control: sum(values)/len(values) for control, values in paired.items()}
    clauses = dict(dynamic_static_mean_ce=contrasts['static']<=cfg['criteria']['dynamic_static_ce_max'], dynamic_static_both_seeds_negative=all(value<0 for value in paired['static']), dynamic_learned_mean_ce=contrasts['learned']<=cfg['criteria']['dynamic_learned_ce_max'], dynamic_cnn_mean_ce=contrasts['cnn']<=cfg['criteria']['dynamic_cnn_ce_max'])
    screening_criterion_met = all(clauses.values())
    useful = sum(r['training_seconds'] for r in records)
    details = dict(arm_means=means, paired_dynamic_minus_control_dev_ce=paired, mean_dynamic_minus_control_dev_ce=contrasts, frozen_screen_clauses=clauses, screening_criterion_met=screening_criterion_met, preparation_seconds=preparation_seconds, mode=MODE, config=config, records=records, source_sha256=source_sha, device=torch.cuda.get_device_name(), torch_version=torch.__version__, cuda_version=torch.version.cuda, elapsed_seconds=time.monotonic()-START, training_seconds=useful, development_evaluated=MODE=='campaign', limitations='Established input-conditioned dynamic convolution family; two frozen filters per channel and pooled-input softmax gate, matched static all-ones input gate; no architecture invention claim. Static gate parameter redundancy and unequal effective capacity; learned ConvMixer/CNN unequal capacity and compute; optimization seeds on one split; no test', projection='Use per-arm epoch timings and observed preparation/artifact overhead; freeze future jobs before dev evaluation; no padding or automatic extension')
    write('details.json', details)
    write('result.json', dict(metrics=dict(screening_criterion_met=int(screening_criterion_met), dynamic_mean_dev_ce=means['dynamic']['ce'], static_mean_dev_ce=means['static']['ce'], learned_mean_dev_ce=means['learned']['ce'], cnn_mean_dev_ce=means['cnn']['ce'], training_seconds=useful, elapsed_seconds=details['elapsed_seconds'], fits=len(records), peak_cuda_memory_bytes=max(r['peak_allocated_bytes'] for r in records), development_evaluated=int(MODE=='campaign')), summary='CUDA native CIFAR10 '+MODE+' complete; exact data, schedules, training-selected and final weights retained. Calibration never evaluates dev.'))


if __name__ == '__main__':
    try:
        main()
    except Exception:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT/'failure-traceback.txt').write_text(traceback.format_exc())
        raise
