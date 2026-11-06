"""Post-terminal cohort aggregation. Run only after the independent artifact audit.

Requires explicit expected source digests from the prospective freeze. This script
checks queue status, inventory, identity, logits and counts; it does not replace
the independent audit of provenance, training, checkpoint replay or timing.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sqlite3

SEEDS = ((4801, 4802), (4803, 4804), (4805, 4806))
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
    if any(len(v) != 2 or len(v[1]) != 64 for v in jobs) or len({v[0] for v in jobs}) != 3:
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
    for index, (job_id, expected_sha) in enumerate(jobs):
        root = (args.outputs / job_id / 'campaign').resolve()
        inventory = json.loads((root / 'integrity.json').read_text())
        for relative, item in inventory.items():
            path = (root / relative).resolve()
            if not path.is_relative_to(root) or not path.is_file() or path.stat().st_size != item['bytes'] or digest(path) != item['sha256']:
                raise ValueError(f'invalid inventory entry {relative}')
        required={'details.json','provenance/experiment.py','provenance/artifacts.py','artifacts.json'}
        if not required.issubset(inventory): raise ValueError('critical inventory entries missing')
        if digest(root / 'provenance/artifacts.py') != '6ef5863486d3f9cfa4fac8b06823ff399986c78e898130e979a9035c6f062baf': raise ValueError('helper snapshot digest mismatch')
        if digest(root / 'provenance/experiment.py') != expected_sha:
            raise ValueError('executed source snapshot does not match frozen digest')
        details = json.loads((root / 'details.json').read_text())
        if details['cohort'] != str(index + 1) or details['cohort_seeds'] != list(SEEDS[index]):
            raise ValueError('cohort identity mismatch')
        rows = details['records']
        if len(rows) != 6 or {(r['seed'], r['arm']) for r in rows} != {(s,a) for s in SEEDS[index] for a in ARMS}:
            raise ValueError('incomplete or duplicate cohort fits')
        for r in rows:
            if r['updates'] != 17600 or r['presentations'] != 4500000 or r['partial_batch'] != 200:
                raise ValueError('fit count mismatch')
            global_index = r['seed'] - 4801
            if r['arm_order'] != list(ARMS[global_index % 3:] + ARMS[:global_index % 3]):
                raise ValueError('arm order mismatch')
            for key in ('initial','final','selected','trace','dev_logits'):
                if r[key] not in inventory: raise ValueError(f'{key} missing from inventory')
            rel = r['dev_logits']
            if rel not in inventory: raise ValueError('logit file absent from inventory')
            with np.load(root / rel, allow_pickle=False) as data:
                ids, labels, logits = data['ids'], data['labels'], data['logits'].astype(np.float64)
            if ids.dtype.kind not in 'iu' or labels.dtype.kind not in 'iu' or ids.shape != (5000,) or labels.shape != (5000,) or np.any(ids < 0) or np.any(ids >= 50000) or logits.shape != (5000,100) or not np.isfinite(logits).all() or np.any(labels < 0) or np.any(labels >= 100) or np.any(ids[1:] <= ids[:-1]):
                raise ValueError('invalid ordered development logits')
            z = logits - logits.max(axis=1, keepdims=True)
            ce = float(np.mean(np.log(np.exp(z).sum(axis=1)) - z[np.arange(len(z)), labels]))
            acc = float(np.mean(np.argmax(logits, axis=1) == labels))
            if not all(math.isfinite(r['final_dev'][key]) for key in ('ce','accuracy')): raise ValueError('nonfinite recorded metric')
            if abs(ce-r['final_dev']['ce']) > 1e-5 or abs(acc-r['final_dev']['accuracy']) > 1e-12:
                raise ValueError('recorded metrics do not match logits')
            records[(r['seed'],r['arm'])] = (ce,acc,ids.copy(),labels.copy())
    reference = records[(4801,'dynamic')]
    if any(not np.array_equal(v[2],reference[2]) or not np.array_equal(v[3],reference[3]) for v in records.values()):
        raise ValueError('development IDs or labels differ between fits')
    diffs = [records[(s,'dynamic')][0]-records[(s,'static')][0] for s in range(4801,4807)]
    mean = sum(diffs)/6
    sd = math.sqrt(sum((x-mean)**2 for x in diffs)/5)
    margin = 2.570581835636314*sd/math.sqrt(6)
    print(json.dumps({'paired_dynamic_minus_static_ce':diffs,'mean':mean,'sample_sd':sd,'descriptive_t95':[mean-margin,mean+margin], 'criterion_met':mean <= -.01 and all(x < 0 for x in diffs), 'per_fit':{f'{s}-{a}':{'ce':records[(s,a)][0],'accuracy':records[(s,a)][1]} for s in range(4801,4807) for a in ARMS},'interpretation':'Provisional aggregate; a valid result requires the separately completed independent full campaign audit.'},indent=2,allow_nan=False))

if __name__ == '__main__':
    main()
