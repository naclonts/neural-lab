#!/usr/bin/env python3
"""Rebuild the worker and retry the jobs affected by the old metric limit.

Run from the repository root with Docker access. Recreating the worker stops
the active experiment; its original inputs and outputs are retained.
"""
import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neural_lab.queue import Queue  # noqa: E402

JOBS = (
    "914efed0752b4f318a269ab41c57ab5f",
    "c6e34e33f19149689a2913f8a1a563ea",
    "896225ca6ab042afbc67b3ed4a958946",
    "bc21c355c8f547dc8a6c1350e41ade08",
    "8bdbae04c89840338a2cb2ba66508fbd",
)


def main():
    queue = Queue(ROOT / ".neural-lab")
    try:
        rows = {}
        for job_id in JOBS:
            row = queue.db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row is None:
                raise RuntimeError(f"Missing original job {job_id}")
            source = queue.root / "queue-inputs" / job_id / "train.py"
            if hashlib.sha256(source.read_bytes()).hexdigest() != row["sha256"]:
                raise RuntimeError(f"Input hash mismatch for {job_id}")
            rows[job_id] = (row, source)

        compose = ["docker", "compose", "-f", "compose.research.yaml"]
        subprocess.run(compose + ["build", "worker"], cwd=ROOT, check=True)
        subprocess.run(compose + ["up", "-d", "--no-deps", "--force-recreate", "worker"], cwd=ROOT, check=True)

        for job_id, (original, source) in rows.items():
            row = queue.db.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row["status"] == "succeeded":
                print(f"{job_id}: already succeeded; skipped")
                continue
            if row["status"] not in ("failed", "interrupted"):
                raise RuntimeError(f"{job_id}: unexpected status {row['status']}; retry later")
            prefix = f"Metric-limit retry of {job_id}: "
            previous = queue.db.execute(
                "SELECT id, status FROM jobs WHERE substr(hypothesis, 1, ?) = ? ORDER BY created_at DESC LIMIT 1",
                (len(prefix), prefix),
            ).fetchone()
            if previous:
                print(f"{job_id}: retry already exists: {previous['id']} ({previous['status']})")
                continue
            hypothesis = (prefix + original["hypothesis"])[:2000]
            new_id = queue.enqueue(source, hypothesis, original["timeout_seconds"])
            print(f"{job_id}: queued retry {new_id}", flush=True)
    finally:
        queue.close()


if __name__ == "__main__":
    main()
