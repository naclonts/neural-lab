#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
python3 - <<'PY'
import fcntl
import hashlib
import json
from pathlib import Path
from neural_lab.queue import Queue, read_script

state = Path('.neural-lab')
state.mkdir(mode=0o700, exist_ok=True)
jobs = [
    ('linear_probe.py', 'Linear-control architecture on a fixed synthetic classification task.'),
    ('deep_mlp.py', 'A three-layer MLP improves held-out accuracy over the linear control.'),
    ('residual_mlp.py', 'Residual connections improve a three-layer MLP at the same width and training budget.'),
]
# A repeated start must preserve the queue, including already completed baselines.
with (state / 'starter-enqueue.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    queue = Queue(state)
    try:
        for name, hypothesis in jobs:
            script = Path('jobs/starter') / name
            digest = hashlib.sha256(read_script(script)).hexdigest()
            existing = queue.db.execute('SELECT id FROM jobs WHERE sha256=? LIMIT 1', (digest,)).fetchone()
            if existing:
                print(json.dumps({'id': existing[0], 'status': 'already_enqueued', 'script': name}))
            else:
                print(json.dumps({'id': queue.enqueue(script, hypothesis), 'status': 'queued', 'script': name}))
    finally:
        queue.close()
PY
