"""Unreviewed exact-step NL-001 training primitive; no queue entry point.

The caller must freeze and audit all data, code and protocol before use. This
module never accepts validation or test data; evaluation is a separate stage.
"""
import math
import time
from pathlib import Path

import torch

from .artifacts import ArtifactWriter, BestCheckpoint
from .nl001_manifest import authenticate_split
from .nl001_models import CalibrationModel, update
from .nl001_training import _batch, _flatten, _save_training_artifacts, batch_indices


def fit_curve(*, programs, manifest, manifest_sha256, output, arm, seed,
              batch_size, steps, learning_rate, device):
    """Train once and retain initialization and exactly requested update states."""
    if arm not in ('A', 'B') or type(seed) is not int:
        raise ValueError('invalid arm or seed')
    if (not isinstance(steps, (tuple, list)) or not steps or
            any(type(s) is not int or s <= 0 for s in steps) or
            sorted(set(steps)) != list(steps)):
        raise ValueError('steps must be strictly increasing positive integers')
    if (isinstance(learning_rate, bool) or not isinstance(learning_rate, (float, int))
            or not math.isfinite(learning_rate) or learning_rate <= 0):
        raise ValueError('learning rate must be finite and positive')
    authenticate_split(programs, 'train', manifest, manifest_sha256)
    records = _flatten(programs)
    schedule = list(batch_indices(len(records), batch_size, steps[-1], seed))
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError('output directory must be empty')
    torch.manual_seed(seed)
    if torch.device(device).type == 'cuda':
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)
    torch.use_deterministic_algorithms(True)
    hidden = 192 if arm == 'A' else 128
    model = CalibrationModel(arm, hidden=hidden, embedding=32).to(device)
    actual_devices = sorted({str(p.device) for p in model.parameters()})
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999),
                                  eps=1e-8, weight_decay=0.01, amsgrad=False)
    writer = ArtifactWriter(out)
    source, tensors = _save_training_artifacts(writer, programs, manifest, manifest_sha256, records)
    config = {'arm': arm, 'seed': seed, 'hidden': hidden, 'embedding': 32,
              'batch_size': batch_size, 'learning_rate': learning_rate,
              'optimizer': 'AdamW', 'betas': [0.9, 0.999], 'eps': 1e-8,
              'weight_decay': 0.01, 'amsgrad': False, 'steps': list(steps), 'device': str(device)}
    checkpoints = {}
    def save(step):
        entry = writer.save_checkpoint('step-' + str(step), model, arm=arm, seed=seed,
            selection={'scope': 'fixed update count', 'step': step}, config=config,
            datasets=[source['sha256'], tensors['sha256']])
        checkpoints[str(step)] = {'path': entry['path'], 'sha256': entry['sha256']}
    save(0)
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    losses = []
    observations = []
    completed_updates = 0
    actual_batch_sizes = []
    best = BestCheckpoint(scope='training-mode scheduled minibatch CE before each optimizer update; '
                          'step is completed updates; A uses status CE; B sums property, subject '
                          'and object CE and excludes witness BCE; earliest tie; held-out data not used')
    for step, indices in enumerate(schedule, 1):
        inputs, targets = _batch(records, indices, device)
        def observe(current_model, cross_entropy):
            best.observe(current_model, cross_entropy, step - 1)
            observations.append({'step': step - 1, 'cross_entropy': cross_entropy,
                                 'batch_size': len(indices)})
        loss = update(model, optimizer, inputs, targets, observe=observe)
        completed_updates += 1
        actual_batch_sizes.append(len(indices))
        if not math.isfinite(float(loss)):
            raise ValueError('non-finite loss')
        losses.append(float(loss))
        if step in steps:
            save(step)
    if completed_updates != steps[-1]:
        raise ValueError('completed update count does not match requested steps')
    selected = best.save(writer, 'best-training-ce', arm=arm, seed=seed, config=config,
                         datasets=[source['sha256'], tensors['sha256']])
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)
    return {'config': config, 'checkpoints': checkpoints, 'losses': losses,
            'best_checkpoint': {'path': selected['path'], 'sha256': selected['sha256'],
                                'selection': selected['selection']},
            'training_ce_observations': observations,
            'examples': len(records), 'exposures': sum(actual_batch_sizes),
            'completed_updates': completed_updates, 'actual_model_devices': actual_devices,
            'batch_sizes': actual_batch_sizes, 'parameters': sum(p.numel() for p in model.parameters()),
            'training_seconds': time.perf_counter() - start,
            'timing_scope': 'batch construction, optimizer updates, checkpoint serialization and synchronization',
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(device) if torch.device(device).type == 'cuda' else None,
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(device) if torch.device(device).type == 'cuda' else None}


def run_curve(*, train, validation, manifest, manifest_sha256, output, seeds,
              batch_size, steps, learning_rate, device, edge_threshold=0.5):
    """Evaluate fixed checkpoints on validation only; never accepts test data.

    Audit the corpus and retain independent anchors externally before calling.
    This draft has no queue or completion/inventory contract.
    """
    import json
    from .nl001_inference import infer_checkpoint, replay_saved
    from .nl001_run_evaluation import evaluate_split
    from .nl001_curve_replay import verify_checkpoint
    if (not isinstance(seeds, (list, tuple)) or not seeds or
            any(type(s) is not int for s in seeds) or len(set(seeds)) != len(seeds)):
        raise ValueError('seeds must be unique integers')
    authenticate_split(train, 'train', manifest, manifest_sha256)
    authenticate_split(validation, 'validation', manifest, manifest_sha256)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError('output directory must be empty')
    def write(path, value):
        path.write_text(json.dumps(value, sort_keys=True, allow_nan=False, indent=2))
    protocol = {'seeds': list(seeds), 'batch_size': batch_size, 'steps': list(steps),
                'learning_rate': learning_rate, 'device': str(device), 'edge_threshold': edge_threshold,
                'manifest_sha256': manifest_sha256, 'evaluation_split': 'validation'}
    write(out / 'protocol.json', protocol)
    write(out / 'validation.json', validation)
    # Complete and persist the full prescribed training cohort before exposing
    # any model validation predictions or metrics.
    fits = {}
    for seed in seeds:
        fits[str(seed)] = {}
        for arm in ('A', 'B'):
            directory = out / (str(seed) + '-' + arm)
            report = fit_curve(programs=train, manifest=manifest, manifest_sha256=manifest_sha256,
                output=directory, arm=arm, seed=seed, batch_size=batch_size, steps=steps,
                learning_rate=learning_rate, device=device)
            fits[str(seed)][arm] = {'training': report}
            write(directory / 'training-report.json', report)
    controls = {}
    for mode in ('oracle', 'parser'):
        controls[mode] = evaluate_split(programs=validation, split='validation', manifest=manifest,
            manifest_sha256=manifest_sha256, output=out / mode, mode=mode)
        if controls[mode]['counts']['aligned_label_conflicts']:
            raise ValueError('control label conflict')
    for seed in seeds:
        for arm in ('A', 'B'):
            directory = out / (str(seed) + '-' + arm)
            report = fits[str(seed)][arm]['training']
            evaluations = {}
            for step in steps:
                checkpoint = report['checkpoints'][str(step)]
                predictions = directory / ('predictions-' + str(step) + '.json')
                start = time.perf_counter()
                pin = infer_checkpoint(programs=validation, split='validation', manifest=manifest,
                    manifest_sha256=manifest_sha256, checkpoint=directory / checkpoint['path'],
                    checkpoint_sha256=checkpoint['sha256'], arm=arm, output=predictions,
                    edge_threshold=edge_threshold)
                inference_seconds = time.perf_counter() - start
                evaluation = replay_saved(programs=validation, split='validation', manifest=manifest,
                    manifest_sha256=manifest_sha256, checkpoint_sha256=checkpoint['sha256'],
                    predictions_path=predictions, predictions_sha256=pin, arm=arm,
                    edge_threshold=edge_threshold)
                verification = verify_checkpoint(programs=validation, manifest=manifest,
                    manifest_sha256=manifest_sha256, checkpoint=directory / checkpoint['path'],
                    checkpoint_sha256=checkpoint['sha256'], predictions=predictions,
                    prediction_sha256=pin, arm=arm, edge_threshold=edge_threshold,
                    evaluation=evaluation, directory=directory, step=step)
                if arm == 'B' and evaluation['counts']['aligned_label_conflicts']:
                    raise ValueError('aligned label conflict')
                evaluations[str(step)] = {'prediction_sha256': pin, 'evaluation': evaluation,
                    'inference_seconds': inference_seconds, 'verification': verification}
            fits[str(seed)][arm]['validation'] = evaluations
            write(directory / 'curve-report.json', fits[str(seed)][arm])
    summary = {'protocol': protocol, 'controls': controls, 'fits': fits,
               'limitations': ['Unreviewed draft; replay shares checker implementation.',
                               'Fresh-process checkpoint inference and independent certificate audit do not verify claim interpretation.',
                               'No external inventory or fresh-process same-checker replay.',
                               'No automatic plateau decision; inspect complete fixed-step curves.',
                               'Validation inference uses CPU; external access control is required to hide partial outputs.']}
    write(out / 'summary.json', summary)
    return summary
