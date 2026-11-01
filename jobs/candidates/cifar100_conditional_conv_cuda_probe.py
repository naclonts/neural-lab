"""PROPOSED synthetic CUDA probe; no optimizer, dataset, network, or training.

Requires independent review and explicit freeze before any execution beyond the guard.
It is not a substitute for a no-development real-data training calibration.
"""
REVIEWED_AND_FROZEN = False


def main():
    if not REVIEWED_AND_FROZEN:
        raise RuntimeError('CUDA probe is proposed only; independent review and freeze required')
    import hashlib
    import importlib.util
    import json
    import os
    import platform
    import resource
    import tempfile
    import time
    from pathlib import Path

    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    from torch.nn import functional as F

    source = Path(__file__).with_name('cifar100_conditional_conv_proposed.py')
    expected = '03883f110960c4bfb857a37b572411bd7ef536ab6115ff52354f70eeb0df697e'
    actual = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError('candidate source hash mismatch; review a new probe')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no CPU fallback')
    spec = importlib.util.spec_from_file_location('candidate', source)
    candidate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(candidate)  # __main__ guard prevents science execution
    if candidate.REVIEWED_AND_FROZEN:
        raise RuntimeError('expected guarded candidate')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(4801)
    device = torch.device('cuda')
    single, mixture = candidate.paired_initial_states(4801)
    # Fixed synthetic inputs and labels; reuse across arms. No optimizer or updates.
    x = torch.randn(256, 3, 32, 32, device=device)
    y = torch.arange(256, device=device) % 100
    rows = []
    for arm in candidate.ARMS:
        torch.cuda.empty_cache()
        model = candidate.Model(arm).to(device)
        model.load_state_dict(single if arm == 'single' else mixture)
        model.train()
        torch.cuda.reset_peak_memory_stats()
        timings = []
        for i in range(5):  # two warmups, three timed samples; BN buffers do update
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            start = time.perf_counter()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            torch.cuda.synchronize()
            middle = time.perf_counter()
            loss.backward()
            torch.cuda.synchronize()
            end = time.perf_counter()
            if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                raise RuntimeError(f'{arm}: nonfinite output')
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise RuntimeError(f'{arm}: nonfinite gradient')
            if i >= 2:
                timings.append({'forward_loss_seconds': middle-start, 'backward_seconds': end-middle})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'state.pt'
            start = time.perf_counter()
            torch.save(model.state_dict(), path)
            save_seconds = time.perf_counter()-start
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            clone = candidate.Model(arm).to(device)
            clone.load_state_dict(torch.load(path, map_location=device, weights_only=True))
            model.eval(); clone.eval()
            with torch.no_grad():
                a, b = model(x[:17]), clone(x[:17])
            if not torch.equal(a, b):
                raise RuntimeError(f'{arm}: checkpoint round trip differs')
            size = path.stat().st_size
        rows.append({'arm': arm, 'timings': timings, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                     'peak_reserved_bytes': torch.cuda.max_memory_reserved(), 'checkpoint_bytes': size,
                     'checkpoint_sha256': digest, 'checkpoint_save_seconds': save_seconds,
                     'roundtrip_bitwise_equal': True})
        del clone, model, logits, loss
    print(json.dumps({'kind': 'synthetic_cuda_probe_no_optimizer_no_data', 'candidate_sha256': actual,
                      'torch': torch.__version__, 'cuda_build': torch.version.cuda,
                      'cudnn': torch.backends.cudnn.version(), 'gpu': torch.cuda.get_device_name(),
                      'python': platform.python_version(), 'platform': platform.platform(),
                      'process_maxrss_kib_linux': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      'batch_size': 256, 'warmups': 2, 'samples': 3, 'arms': rows}, allow_nan=False))


if __name__ == '__main__':
    main()
