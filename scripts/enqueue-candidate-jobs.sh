#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
python3 - <<'PY'
import fcntl
import hashlib
from pathlib import Path

from neural_lab.queue import Queue, read_script


jobs = (
    (
        Path("jobs/candidates/conditioned_low_rank_gate.py"),
        "An input-conditioned low-rank residual gate improves held-out accuracy "
        "over a parameter-matched dense residual MLP on a cross-feature task.",
    ),
    (
        Path("jobs/candidates/energy_normalized_mixer.py"),
        "Energy-normalized feature mixing improves held-out accuracy over a "
        "parameter-matched ordinary residual MLP on a cross-feature task.",
    ),
)
state = Path(".neural-lab")
state.mkdir(mode=0o700, exist_ok=True)
with (state / "candidate-enqueue.lock").open("a") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    queue = Queue(Path(".neural-lab"))
    try:
        for path, hypothesis in jobs:
            digest = hashlib.sha256(read_script(path)).hexdigest()
            existing = queue.db.execute(
                "SELECT id, status FROM jobs WHERE sha256=? AND hypothesis=? "
                "ORDER BY created_at LIMIT 1",
                (digest, hypothesis),
            ).fetchone()
            if existing:
                print(f"{path}: already queued as {existing['id']} ({existing['status']})")
            else:
                job_id = queue.enqueue(path, hypothesis, timeout_seconds=300)
                print(f"{path}: queued as {job_id}")
    finally:
        queue.close()
PY
