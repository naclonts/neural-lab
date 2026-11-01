"""No-dev CUDA calibration of paired CIFAR-10 anti-alias arms."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['HF_HOME'] = '/output/hf-cache'
os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
import copy
import hashlib
import json
import math
import shutil
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from neural_lab.artifacts import ArtifactWriter, BestCheckpoint, load_json_state

OUT = Path('/output')
SEEDS = (3801,)
ARMS = ('plain', 'binomial')
EPOCHS, BATCH, WIDTH, DEPTH = 2, 256, 64, 4
SOURCE_SHA = '8428b53a88a11ac374111006708df51469e315a22ac6d66470afd9c78d2ae883'
START = time.monotonic()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def event(kind, **data):
    row = dict(event=kind, elapsed_seconds=time.monotonic()-START, **data)
    with (OUT/'progress.jsonl').open('a') as f:
        f.write(json.dumps(row, allow_nan=False)+'\n')
    print(json.dumps(row), flush=True)


class Model(nn.Module):
    def __init__(self, arm):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(arm)
        self.arm = arm
        if arm == 'binomial':
            v = torch.tensor([1., 2., 1.])
            self.register_buffer('kernel', (v[:, None]*v[None, :]/16).view(1, 1, 3, 3).repeat(3, 1, 1, 1))
        self.stem = nn.Sequential(nn.Conv2d(3, WIDTH, 2, stride=2), nn.GELU(), nn.BatchNorm2d(WIDTH))
        self.blocks = nn.Sequential(*[nn.Sequential(nn.Conv2d(WIDTH, WIDTH, 3, padding=1), nn.GELU(), nn.BatchNorm2d(WIDTH)) for _ in range(DEPTH)])
        self.head = nn.Linear(WIDTH, 10)

    def forward(self, x):
        if self.arm == 'binomial':
            x = F.conv2d(F.pad(x, (1, 1, 1, 1), mode='reflect'), self.kernel, groups=3)
        return self.head(self.blocks(self.stem(x)).mean((2, 3)))


def schedule(seed, n):
    order, crop, flip = [], [], []
    for epoch in range(EPOCHS):
        rng = np.random.default_rng(seed+100000+epoch*1000003)
        order.append(rng.permutation(n).astype(np.int32))
        crop.append(rng.integers(0, 9, (n, 2), dtype=np.int16))
        flip.append(rng.integers(0, 2, n, dtype=np.uint8))
    return np.stack(order), np.stack(crop), np.stack(flip)


def inputs(raw, ids, mean, std, crop=None, flip=None):
    x = raw[ids].float().div_(255)
    if crop is not None:
        x = F.pad(x, (4, 4, 4, 4), mode='reflect')
        rows = torch.arange(32, device=raw.device)[None, :]+crop[:, 0, None]
        cols = torch.arange(32, device=raw.device)[None, :]+crop[:, 1, None]
        x = x[torch.arange(len(ids), device=raw.device)[:, None, None], :, rows[:, :, None], cols[:, None, :]].permute(0, 3, 1, 2)
        x = torch.where(flip[:, None, None, None].bool(), x.flip(-1), x)
    return (x-mean)/std


@torch.no_grad()
def evaluate(model, raw, labels, ids, mean, std, shifts=False, capture=None):
    model.eval()
    totals = np.zeros((5 if shifts else 1, 3), dtype=np.float64)
    directions = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)) if shifts else ((0, 0),)
    if capture is not None:
        capture.update(ids=[], labels=[], logits=[])
    for start in range(0, len(ids), BATCH):
        ix = ids[start:start+BATCH]
        x = inputs(raw, ix, mean, std)
        batch_logits = []
        base = None
        for i, shift in enumerate(directions):
            logits = model(torch.roll(x, shift, dims=(2, 3)) if i else x)
            pred = logits.argmax(1)
            if capture is not None:
                batch_logits.append(logits.cpu().numpy())
            if base is None:
                base = pred
            totals[i] += [F.cross_entropy(logits, labels[ix], reduction='sum').item(), (pred == labels[ix]).sum().item(), (pred != base).sum().item()]
        if capture is not None:
            capture['ids'].append(ix.cpu().numpy())
            capture['labels'].append(labels[ix].cpu().numpy())
            capture['logits'].append(np.stack(batch_logits, axis=1))
    if capture is not None:
        capture.update({k: np.concatenate(v) for k, v in capture.items()})
    if not np.isfinite(totals).all():
        raise FloatingPointError('nonfinite evaluation')
    stats = [dict(ce=float(a/len(ids)), accuracy=float(b/len(ids)), disagreement=float(c/len(ids))) for a, b, c in totals]
    return dict(clean=stats[0], translations=dict(zip(('up', 'down', 'left', 'right'), stats[1:])), consistency_error=float(totals[1:, 2].sum()/(4*len(ids))) if shifts else None)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA mandatory')
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'artifacts.json').exists() or (OUT/'progress.jsonl').exists():
        raise RuntimeError('fresh output directory required')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    import datasets
    import huggingface_hub
    from huggingface_hub import HfApi, hf_hub_download
    import neural_lab.artifacts as artifacts
    writer = ArtifactWriter(OUT)
    (OUT/'provenance').mkdir()
    revision = '0b2714987fa478483af9968de7c934580d0bb9a2'
    metadata = HfApi(token=False).dataset_info('uoft-cs/cifar10', revision=revision, timeout=20)
    if metadata.sha != revision:
        raise ValueError('immutable HF revision mismatch')
    source_path = hf_hub_download(repo_id='uoft-cs/cifar10', repo_type='dataset', filename='plain_text/train-00000-of-00001.parquet', revision=revision, token=False, local_dir=str(OUT/'hf-source'))
    if sha(source_path) != SOURCE_SHA:
        raise ValueError('pinned official training parquet hash mismatch')
    shutil.copyfile(source_path, OUT/'provenance/train.parquet')
    shutil.copyfile(__file__, OUT/'provenance/experiment.py')
    shutil.copyfile(artifacts.__file__, OUT/'provenance/artifacts.py')
    source = writer.register_file('datasets', OUT/'provenance/train.parquet', 'official-train-parquet', format='parquet', source=dict(repo='uoft-cs/cifar10', revision='0b2714987fa478483af9968de7c934580d0bb9a2', path='plain_text/train-00000-of-00001.parquet', creator='https://www.cs.toronto.edu/~kriz/cifar.html', url='https://huggingface.co/datasets/uoft-cs/cifar10/tree/'+revision, authentication='public token=False', license=metadata.card_data.get('license') if metadata.card_data else None, datasets_version=datasets.__version__, huggingface_hub_version=huggingface_hub.__version__))
    for filename in ('experiment.py', 'artifacts.py'):
        writer.register_file('datasets', OUT/'provenance'/filename, filename, format='python', source=dict(role='executed source snapshot'))
    ds = datasets.load_dataset('parquet', data_files={'train': str(OUT/'provenance/train.parquet')}, split='train', cache_dir=str(OUT/'cache'))
    images = np.stack([np.asarray(row['img'].convert('RGB'), dtype=np.uint8) for row in ds])
    labels = np.asarray(ds['label'], dtype=np.int64)
    if images.shape != (50000, 32, 32, 3) or not np.array_equal(np.bincount(labels), np.full(10, 5000)):
        raise ValueError('unexpected CIFAR layout')
    rng = np.random.default_rng(20261030)
    train, dev = [], []
    for c in range(10):
        indices = rng.permutation(np.flatnonzero(labels == c))
        train.extend(indices[:4500]); dev.extend(indices[4500:])
    train, dev = np.asarray(sorted(train)), np.asarray(sorted(dev))
    pixels = images[train].astype(np.float64)/255
    mean, std = pixels.mean((0, 1, 2)), pixels.std((0, 1, 2))
    del pixels
    data = writer.save_dataset('split', dict(images=images, labels=labels, train_ids=train, dev_ids=dev, mean=mean, std=std), provenance=dict(source=source['path'], split_seed=20261030, numpy=np.__version__, split='PCG64 per-class permutations, sorted IDs, 4500/500 per class', normalization='train RGB population moments'))
    raw = torch.from_numpy(images.transpose(0, 3, 1, 2).copy()).cuda()
    y = torch.from_numpy(labels).cuda()
    mu = torch.tensor(mean, dtype=torch.float32, device='cuda')[None, :, None, None]
    sd = torch.tensor(std, dtype=torch.float32, device='cuda')[None, :, None, None]
    train_gpu, dev_gpu = torch.tensor(train, device='cuda'), torch.tensor(dev, device='cuda')
    config = dict(purpose='no-dev calibration only; no scientific endpoint', epochs=EPOCHS, seeds=SEEDS, arms=ARMS, split_seed=20261030, batch=BATCH, optimizer='AdamW lr=0.001 weight_decay=0.01 constant', dtype='float32 deterministic, TF32 disabled', intervention='reflect pad1 then channelwise binomial [1,2,1] outer /16 on normalized input before learned stem', checkpoint_selection='BestCheckpoint earliest minimum of post-epoch canonical train inference CE')
    # Tensor-level CUDA checks run before fitting, using only synthetic inputs.
    test = Model('binomial').cuda().eval()
    const = torch.ones((1, 3, 8, 8), device='cuda')
    blurred = F.conv2d(F.pad(const, (1, 1, 1, 1), mode='reflect'), test.kernel, groups=3)
    assert torch.equal(blurred, const)
    impulse = torch.zeros_like(const); impulse[0, 0, 4, 4] = 1
    blurred = F.conv2d(F.pad(impulse, (1, 1, 1, 1), mode='reflect'), test.kernel, groups=3)
    assert torch.equal(blurred[0, 0, 3:6, 3:6], test.kernel[0, 0])
    assert torch.count_nonzero(blurred[:, 1:]).item() == 0
    assert 'kernel' in dict(test.named_buffers()) and 'kernel' not in dict(test.named_parameters())
    plain = Model('plain').cuda().eval()
    state = test.state_dict(); state.update(plain.state_dict()); test.load_state_dict(state)
    with torch.no_grad():
        test.kernel.zero_(); test.kernel[:, :, 1, 1] = 1
        z = torch.randn((3, 3, 32, 32), device='cuda')
        assert torch.equal(test(z), plain(z))
        v = torch.tensor([1., 2., 1.], device='cuda')
        test.kernel.copy_((v[:, None]*v[None, :]/16).view(1, 1, 3, 3).repeat(3, 1, 1, 1))
    test.zero_grad(set_to_none=True); test(z).sum().backward()
    assert test.stem[0].weight.grad is not None and torch.isfinite(test.stem[0].weight.grad).all().item()
    assert test.kernel.grad is None
    del test, plain, z
    event('synthetic_cuda_checks_passed')
    records = []
    for seed in SEEDS:
        order, crop, flip = schedule(seed, len(train))
        sched = writer.save_dataset(f'schedule-{seed}', dict(order=order, crop=crop, flip=flip), provenance=dict(seed=seed, numpy=np.__version__, generator='PCG64 per epoch seed+100000+epoch*1000003, order then crop then flip', indexing='epoch,position; order indexes sorted train_ids'))
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        initial = copy.deepcopy(Model('plain').state_dict())
        arm_order = ARMS if seed == SEEDS[0] else ARMS[::-1]
        for arm in arm_order:
            model = Model(arm)
            state = model.state_dict()
            state.update({k: v.clone() for k, v in initial.items()})
            model.load_state_dict(state, strict=True)
            assert all(torch.equal(model.state_dict()[k], v) for k, v in initial.items())
            model.cuda()
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
            best = BestCheckpoint(scope='earliest minimum post-epoch canonical training inference CE; no dev selection')
            traces, durations = [], []
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize(); started = time.monotonic()
            step = 0
            for epoch in range(EPOCHS):
                model.train(); tick = time.monotonic()
                for start in range(0, len(train), BATCH):
                    stop = min(start+BATCH, len(train))
                    ix = train_gpu[torch.as_tensor(order[epoch, start:stop].astype(np.int64), device='cuda')]
                    c = torch.as_tensor(crop[epoch, start:stop].astype(np.int64), device='cuda')
                    f = torch.as_tensor(flip[epoch, start:stop], device='cuda')
                    optimizer.zero_grad(set_to_none=True)
                    loss = F.cross_entropy(model(inputs(raw, ix, mu, sd, c, f)), y[ix])
                    if not math.isfinite(float(loss.detach())):
                        raise FloatingPointError('nonfinite loss')
                    loss.backward()
                    for p in model.parameters():
                        if p.grad is not None:
                            torch._assert_async(torch.isfinite(p.grad).all(), 'nonfinite gradient')
                    optimizer.step(); step += 1
                torch.cuda.synchronize(); durations.append(time.monotonic()-tick)
                train_eval = evaluate(model, raw, y, train_gpu, mu, sd)['clean']
                best.observe(model, train_eval['ce'], step)
                traces.append([epoch+1, step, train_eval['ce'], train_eval['accuracy']])
                event('epoch', seed=seed, arm=arm, epoch=epoch+1, step=step, train_ce=train_eval['ce'])
            torch.cuda.synchronize(); training_seconds = time.monotonic()-started
            if not all(torch.isfinite(t).all().item() for t in model.state_dict().values()):
                raise FloatingPointError('nonfinite state')
            refs = [source['path'], data['path'], sched['path']]
            final = writer.save_checkpoint(f'{seed}-{arm}-final', model, arm=arm, seed=seed, selection=dict(metric='fixed final update', step=step), config=config, datasets=refs)
            selected = best.save(writer, f'{seed}-{arm}-training-selected', arm=arm, seed=seed, config=config, datasets=refs)
            # Reload serialized final state and reproduce exact train-input logits, including BN/filter.
            restored = Model(arm).cuda().eval()
            restored.load_state_dict(load_json_state(OUT/final['path']), strict=True)
            model.eval()
            with torch.no_grad():
                sample = inputs(raw, train_gpu[:17], mu, sd)
                expected, actual = model(sample), restored(sample)
                assert torch.equal(expected, actual)
            del restored, sample, expected, actual
            writer.save_dataset(f'{seed}-{arm}-train-trace', dict(observations=np.asarray(traces)), provenance=dict(columns=['epoch', 'step', 'canonical_train_ce', 'canonical_train_accuracy'], selected=selected['path']))
            record = dict(seed=seed, arm=arm, arm_order=arm_order, updates=step, epochs=EPOCHS, parameters=sum(p.numel() for p in model.parameters()), trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad), training_seconds_including_train_eval=training_seconds, epoch_update_seconds=durations, peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(), final_train=traces[-1][2:], checkpoint_roundtrip_exact=True, final=final['path'], selected=selected['path'])
            records.append(record); event('fit_complete', **record)
            del model, optimizer, best
    details = dict(config=config, records=records, development_evaluated=False, source_sha256=sha(__file__), artifact_helper_sha256=sha(artifacts.__file__), torch_version=torch.__version__, cuda_version=torch.version.cuda, cudnn_version=torch.backends.cudnn.version(), numpy_version=np.__version__, datasets_version=datasets.__version__, huggingface_hub_version=huggingface_hub.__version__, device=torch.cuda.get_device_name(), elapsed_seconds=time.monotonic()-START)
    (OUT/'details.json').write_text(json.dumps(details, indent=2, allow_nan=False))
    (OUT/'result.json').write_text(json.dumps(dict(metrics=dict(fits=len(records), development_evaluated=0, elapsed_seconds=details['elapsed_seconds']), summary='No-dev CUDA anti-alias calibration complete; timing does not establish scientific outcome.'), indent=2, allow_nan=False))


if __name__ == '__main__':
    try:
        main()
    except Exception:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT/'failure-traceback.txt').write_text(traceback.format_exc())
        raise
