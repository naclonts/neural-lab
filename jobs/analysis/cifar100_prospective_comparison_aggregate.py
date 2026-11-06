"""Post-terminal exploratory comparison; run only after the prospective review gate.

Requires explicit expected source digests from the prospective freeze. This script
checks queue status, inventory, identity, logits and counts; it does not replace
independent provenance, execution attestation, CUDA replay, or timing validation.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sqlite3

SEEDS = ((5901, 5902), (5903, 5904), (5905, 5906))
ARMS = ('dynamic', 'static', 'single')

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--queue-db', required=True, type=Path)
    p.add_argument('--outputs', required=True, type=Path)
    p.add_argument('--job', action='append', required=True, metavar='ID:SOURCE_SHA256', help='repeat in cohort order')
    args = p.parse_args()
    if len(args.job) != 3: p.error('exactly three ordered cohort jobs are required')
    jobs = [v.split(':') for v in args.job]
    if any(len(v) != 2 or not v[0] or not all(c.isalnum() or c in '-_' for c in v[0]) or len(v[1]) != 64 or any(c not in '0123456789abcdef' for c in v[1]) for v in jobs) or len({v[0] for v in jobs}) != 3:
        p.error('expected three distinct ID:SOURCE_SHA256 values')
    db = sqlite3.connect(f'file:{args.queue_db.resolve()}?mode=ro', uri=True)
    # Check every terminal status before opening any output or endpoint.
    for job_id, expected_sha in jobs:
        row = db.execute('select status, sha256, finished_at from jobs where id=?', (job_id,)).fetchone()
        if row is None or row[0] != 'succeeded' or row[1] != expected_sha or row[2] is None:
            raise ValueError(f'job {job_id} is not a completed job with the expected source digest')
    db.close()
    import numpy as np
    records = {}
    reference_split = None
    for index, (job_id, expected_sha) in enumerate(jobs):
        root = (args.outputs / job_id / 'prospective_comparison').resolve()
        inventory = json.loads((root / 'integrity.json').read_text())
        actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p.name not in ('integrity.json', 'result.json')}
        if set(inventory) != actual: raise ValueError('inventory does not match output files')
        for relative, item in inventory.items():
            path = (root / relative).resolve()
            if not path.is_relative_to(root) or not path.is_file() or path.stat().st_size != item['bytes'] or digest(path) != item['sha256']:
                raise ValueError(f'invalid inventory entry {relative}')
        required={'details.json','provenance/experiment.py','provenance/artifacts.py','artifacts.json'}
        if not required.issubset(inventory): raise ValueError('critical inventory entries missing')
        if digest(root / 'provenance/artifacts.py') != '6ef5863486d3f9cfa4fac8b06823ff399986c78e898130e979a9035c6f062baf': raise ValueError('helper snapshot digest mismatch')
        if digest(root / 'provenance/experiment.py') != expected_sha:
            raise ValueError('executed source snapshot does not match frozen digest')
        manifest = json.loads((root / 'artifacts.json').read_text())
        if manifest.get('version') != 1 or any(not isinstance(manifest.get(k), list) for k in ('datasets', 'checkpoints')):
            raise ValueError('invalid artifact manifest')
        entries = manifest['datasets'] + manifest['checkpoints']
        if len({e['path'] for e in entries}) != len(entries): raise ValueError('duplicate artifact path')
        for entry in entries:
            item = inventory.get(entry['path'])
            if item is None or item['sha256'] != entry['sha256'] or item['bytes'] != entry['bytes']:
                raise ValueError('manifest artifact missing or inconsistent')
        splits = [e for e in manifest['datasets'] if e['label'] == 'cifar100-training-split']
        if len(splits) != 1: raise ValueError('missing or duplicate split')
        with np.load(root / splits[0]['path'], allow_pickle=False) as data:
            split_ids, train_ids, split_labels = data['dev_ids'].copy(), data['train_ids'].copy(), data['labels'].copy()
        if (split_ids.shape != (5000,) or split_labels.shape != (50000,) or
                split_ids.dtype.kind not in 'iu' or split_labels.dtype.kind not in 'iu' or
                np.any(split_ids < 0) or np.any(split_ids >= 50000) or np.any(split_ids[1:] <= split_ids[:-1])):
            raise ValueError('invalid split IDs or labels')
        if (train_ids.shape != (45000,) or train_ids.dtype.kind not in 'iu' or
                np.any(train_ids < 0) or np.any(train_ids >= 50000) or np.any(train_ids[1:] <= train_ids[:-1]) or
                np.any(split_labels < 0) or np.any(split_labels >= 100) or
                not np.array_equal(np.bincount(split_labels, minlength=100), np.full(100, 500))):
            raise ValueError('invalid training IDs or class counts')
        rng = np.random.default_rng(20261105)
        expected_train, expected_dev = [], []
        for cls in range(100):
            indices = rng.permutation(np.flatnonzero(split_labels == cls))
            expected_train.extend(indices[:450]); expected_dev.extend(indices[450:])
        if not np.array_equal(train_ids, sorted(expected_train)) or not np.array_equal(split_ids, sorted(expected_dev)):
            raise ValueError('split does not match the fixed stratified PCG64 algorithm')
        current_split = (train_ids, split_ids, split_labels)
        if reference_split is None: reference_split = current_split
        elif any(not np.array_equal(a, b) for a, b in zip(current_split, reference_split)):
            raise ValueError('development split differs between cohorts')
        for seed in SEEDS[index]:
            if sum(e['label'] == f'schedule-{seed}' for e in manifest['datasets']) != 1:
                raise ValueError('missing or duplicate schedule')
        details = json.loads((root / 'details.json').read_text())
        if details['cohort'] != str(index + 1) or details['cohort_seeds'] != list(SEEDS[index]):
            raise ValueError('cohort identity mismatch')
        if details['config'].get('seeds') != list(range(5901, 5907)) or details['config'].get('split_seed') != 20261105 or 'threshold' in details['config']:
            raise ValueError('study configuration mismatch')
        for key, expected in {'epochs': 100, 'batch': 256, 'lr': .001, 'weight_decay': .01}.items():
            if details['config'].get(key) != expected: raise ValueError('study configuration mismatch: ' + key)
        rows = details['records']
        if len(rows) != 6 or {(r['seed'], r['arm']) for r in rows} != {(s,a) for s in SEEDS[index] for a in ARMS}:
            raise ValueError('incomplete or duplicate cohort fits')
        used_paths = set()
        for r in rows:
            if r['updates'] != 17600 or r['presentations'] != 4500000 or r['partial_batch'] != 200:
                raise ValueError('fit count mismatch')
            global_index = r['seed'] - 5901
            if r['arm_order'] != list(ARMS[global_index % 3:] + ARMS[:global_index % 3]):
                raise ValueError('arm order mismatch')
            for key, suffix, category in (('initial','initial','checkpoints'), ('final','final','checkpoints'),
                                          ('selected','training-selected','checkpoints'), ('trace','trace','datasets'),
                                          ('dev_logits','dev-logits','datasets')):
                path = r[key]
                if path in used_paths: raise ValueError('artifact path reused across roles or fits')
                used_paths.add(path)
                label = f"{r['seed']}-{r['arm']}-{suffix}"
                matches = [e for e in manifest[category] if e['path'] == path and e['label'] == label]
                if path not in inventory or len(matches) != 1:
                    raise ValueError(f'{key} missing or has incorrect manifest identity')
                if category == 'checkpoints' and (matches[0].get('seed') != r['seed'] or matches[0].get('arm') != r['arm']):
                    raise ValueError('checkpoint manifest seed or arm mismatch')
            rel = r['dev_logits']
            if rel not in inventory: raise ValueError('logit file absent from inventory')
            with np.load(root / rel, allow_pickle=False) as data:
                ids, labels, logits = data['ids'], data['labels'], data['logits'].astype(np.float64)
            if ids.dtype.kind not in 'iu' or labels.dtype.kind not in 'iu' or ids.shape != (5000,) or labels.shape != (5000,) or np.any(ids < 0) or np.any(ids >= 50000) or logits.shape != (5000,100) or not np.isfinite(logits).all() or np.any(labels < 0) or np.any(labels >= 100) or np.any(ids[1:] <= ids[:-1]):
                raise ValueError('invalid ordered development logits')
            if not np.array_equal(ids, split_ids) or not np.array_equal(labels, split_labels[split_ids]):
                raise ValueError('logits do not match split IDs and labels')
            if not isinstance(r['replay_max_abs'], (int, float)) or not math.isfinite(r['replay_max_abs']) or r['replay_max_abs'] < 0:
                raise ValueError('invalid replay record')
            z = logits - logits.max(axis=1, keepdims=True)
            ce = float(np.mean(np.log(np.exp(z).sum(axis=1)) - z[np.arange(len(z)), labels]))
            acc = float(np.mean(np.argmax(logits, axis=1) == labels))
            if not all(math.isfinite(r['final_dev'][key]) for key in ('ce','accuracy')): raise ValueError('nonfinite recorded metric')
            if abs(ce-r['final_dev']['ce']) > 1e-5 or abs(acc-r['final_dev']['accuracy']) > 1e-12:
                raise ValueError('recorded metrics do not match logits')
            records[(r['seed'],r['arm'])] = (ce,acc,ids.copy(),labels.copy())
    reference = records[(5901,'dynamic')]
    if any(not np.array_equal(v[2],reference[2]) or not np.array_equal(v[3],reference[3]) for v in records.values()):
        raise ValueError('development IDs or labels differ between fits')
    diffs = [records[(s,'dynamic')][0]-records[(s,'static')][0] for s in range(5901,5907)]
    mean = sum(diffs)/6
    sd = math.sqrt(sum((x-mean)**2 for x in diffs)/5)
    margin = 2.570581835636314*sd/math.sqrt(6)
    print(json.dumps({'paired_dynamic_minus_static_ce':diffs,'mean':mean,'sample_sd':sd,'descriptive_t95':[mean-margin,mean+margin], 'per_fit':{f'{s}-{a}':{'ce':records[(s,a)][0],'accuracy':records[(s,a)][1]} for s in range(5901,5907) for a in ARMS},'interpretation':'Exploratory aggregate conditional on the separately frozen review and evidence contract; same-job replay and hashes do not establish independent execution or immutable custody.'},indent=2,allow_nan=False))

if __name__ == '__main__':
    main()
