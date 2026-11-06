"""Unfrozen calibration training driver; never a queue entry point.

Callers must provide a digest-pinned training split. This driver deliberately
has no command line, test-data loading, result.json, or submission mechanism.
Its output is not an independently evaluated experiment.
"""
import json
import random
import time
from pathlib import Path

import torch

from .artifacts import ArtifactWriter, BestCheckpoint
from .nl001_data import digest
from .nl001_models import CalibrationModel, update, training_loss
from .nl001_training_data import example
from .nl001_manifest import authenticate_split

VERSION = 'nl001-training-draft-v3'


def batch_indices(count, batch_size, updates, seed):
    """Deterministic shuffled epochs, preserving each epoch's final partial batch."""
    if any(type(x) is not int for x in (count, batch_size, updates, seed)) or min(count, batch_size, updates) <= 0:
        raise ValueError('count, batch size and updates must be positive integers; seed must be integer')
    rng = random.Random(seed)
    emitted = 0
    while emitted < updates:
        order = list(range(count))
        rng.shuffle(order)
        for offset in range(0, count, batch_size):
            yield order[offset:offset + batch_size]
            emitted += 1
            if emitted == updates:
                return


def _flatten(programs):
    records = []
    for entry in programs:
        variants = entry['variants']
        if len(variants) != 1 or variants[0]['presentation'] != {'heldout_template': False, 'renamed': False}:
            raise ValueError('training driver accepts only original training-template presentations')
        program = variants[0]
        for question in program['questions']:
            records.append(example(program, question))
    if not records:
        raise ValueError('empty split')
    return records


def _batch(records, indices, device):
    inputs, targets = zip(*(records[i] for i in indices))
    return ({key: torch.tensor([x[key] for x in inputs], device=device) for key in inputs[0]},
            {key: torch.tensor([x[key] for x in targets], device=device) for key in targets[0]})


def _save_training_artifacts(writer, programs, manifest, manifest_sha256, records):
    """Persist source and the exact ordered CPU tensors used to construct batches."""
    out = writer.output
    manifest_path = out / 'split-manifest.json'
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, allow_nan=False, indent=2))
    writer.register_file('datasets', manifest_path, 'split-manifest', format='json', canonical_sha256=manifest_sha256)
    payload_path = out / 'training-programs.json'
    payload_path.write_text(json.dumps(programs, sort_keys=True, allow_nan=False, separators=(',', ':')))
    entry = writer.register_file('datasets', payload_path, 'training-programs',
                                 format='json', canonical_sha256=digest(programs))
    inputs, targets = _batch(records, range(len(records)), 'cpu')
    tensor_entry = writer.save_dataset('training-tensors',
        {**{'input_' + k: v for k, v in inputs.items()},
         **{'target_' + k: v for k, v in targets.items()}},
        provenance={'programs_sha256': entry['sha256'], 'manifest_sha256': manifest_sha256,
                    'order': 'program order, then question order'})
    return entry, tensor_entry


def _save_best_checkpoint(best, writer, arm, seed, config, source, tensors):
    return best.save(writer, 'best-' + arm, arm=arm, seed=seed,
                     config=config, datasets=[source['sha256'], tensors['sha256']])


def full_training_cross_entropy(model, records, batch_size, device):
    """Fixed-order, example-weighted CE; excludes B's witness objective."""
    model.eval()
    total = 0.0
    with torch.inference_mode():
        for offset in range(0, len(records), batch_size):
            indices = range(offset, min(offset + batch_size, len(records)))
            inputs, targets = _batch(records, indices, device)
            _, ce = training_loss(model, inputs, targets, with_cross_entropy=True)
            if not torch.isfinite(ce):
                raise ValueError('non-finite selection loss')
            total += float(ce) * len(indices)
    return total / len(records)


def fit_fixed_updates(*, programs, manifest, manifest_sha256, output, arm, seed, batch_size, updates, learning_rate, device):
    """Train fixed updates and select minimum full-training-set cross entropy.

    The independently frozen manifest digest binds `programs` to the training
    split. Caller must also audit corpus construction and freeze the protocol
    and executed source bytes. Observe after every 100 updates and the final update;
    equal losses retain the earliest observation. B selection sums the property,
    subject and object cross entropies; its witness BCE is excluded.
    """
    if arm not in ('A', 'B') or type(seed) is not int:
        raise ValueError('invalid arm or seed')
    if not isinstance(programs, list):
        raise ValueError('programs must be a list')
    if isinstance(learning_rate, bool) or not isinstance(learning_rate, (float, int)) or not 0 < learning_rate < float('inf'):
        raise ValueError('learning rate must be finite and positive')
    authenticate_split(programs, 'train', manifest, manifest_sha256)
    # Materialize schedule before allocating resources to validate all arguments.
    started = time.perf_counter()
    records = _flatten(programs)
    schedule = list(batch_indices(len(records), batch_size, updates, seed))
    prepare_seconds = time.perf_counter() - started
    torch.manual_seed(seed)
    if torch.device(device).type == 'cuda':
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)
    torch.use_deterministic_algorithms(True)
    hidden = 192 if arm == 'A' else 128
    model = CalibrationModel(arm, hidden=hidden, embedding=32).to(device)
    optimizer_settings = {'lr': learning_rate, 'betas': (0.9, 0.999), 'eps': 1e-8,
                          'weight_decay': 0.01, 'amsgrad': False}
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_settings)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError('output directory must be empty')
    writer = ArtifactWriter(out)
    artifact_started = time.perf_counter()
    entry, tensor_entry = _save_training_artifacts(writer, programs, manifest, manifest_sha256, records)
    prepare_seconds += time.perf_counter() - artifact_started
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    losses = []
    selection_observations = []
    best = BestCheckpoint(scope='full training-set CE in fixed order, evaluation mode, after updates 100, 200, ..., and final update; B sums property, subject, object CE and excludes witness BCE; earliest tie')
    selection_seconds = 0.0
    for step, indices in enumerate(schedule, 1):
        inputs, targets = _batch(records, indices, device)
        losses.append(update(model, optimizer, inputs, targets))
        if step % 100 == 0 or step == updates:
            if torch.device(device).type == 'cuda':
                torch.cuda.synchronize(device)
            selection_start = time.perf_counter()
            value = full_training_cross_entropy(model, records, batch_size, device)
            if torch.device(device).type == 'cuda':
                torch.cuda.synchronize(device)
            selection_seconds += time.perf_counter() - selection_start
            selection_observations.append({'step': step, 'training_cross_entropy': value, 'examples': len(records)})
            best.observe(model, value, step)
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)
    train_seconds = time.perf_counter() - started
    config = {'version': VERSION, 'arm': arm, 'seed': seed, 'batch_size': batch_size,
              'updates': updates, 'hidden': hidden, 'embedding': 32, 'optimizer': 'AdamW', 'optimizer_settings': optimizer_settings, 'device': str(device)}
    checkpoint = _save_best_checkpoint(best, writer, arm, seed, config, entry, tensor_entry)
    report = {'config': config, 'training_programs_sha256': entry['sha256'], 'manifest_sha256': manifest_sha256,
              'training_tensors_sha256': tensor_entry['sha256'],
              'checkpoint_sha256': checkpoint['sha256'], 'checkpoint_selection': checkpoint['selection'], 'examples': len(records),
              'parameters': sum(p.numel() for p in model.parameters()),
              'example_exposures': sum(map(len, schedule)), 'batch_sizes': list(map(len, schedule)),
              'preparation_seconds': prepare_seconds,
              'preparation_timing_scope': 'flatten and schedule plus source and tensor artifact serialization; excludes model/optimizer setup',
              'training_seconds': train_seconds, 'selection_seconds': selection_seconds,
              'training_timing_scope': 'batch tensor construction, optimizer updates, selection, and device synchronization',
              'losses': losses, 'selection_observations': selection_observations, 'torch_version': torch.__version__,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(device) if torch.device(device).type == 'cuda' else None,
              'peak_reserved_bytes': torch.cuda.max_memory_reserved(device) if torch.device(device).type == 'cuda' else None,
              'limitations': 'Unreviewed training draft; no validation, inference, replay, frozen source, or scientific result.'}
    report_path = out / 'training-report.json'
    report_path.write_text(json.dumps(report, sort_keys=True, allow_nan=False, indent=2))
    writer.register_file('datasets', report_path, 'training-report', format='json')
    return report
