"""Persistent, deterministic experiment queue for the trusted host supervisor."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import sqlite3
import stat
import sys
import threading
import time
import uuid

from neural_lab.manage import check_readiness
from neural_lab.runner import CleanupError, DockerRunner

MAX_SCRIPT_BYTES = 65536
DEFAULT_STATE = Path(".neural-lab")
DEFAULT_JOB_TIMEOUT = 1800


def read_script(path: Path) -> bytes:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or not 1 <= info.st_size <= MAX_SCRIPT_BYTES:
            raise ValueError("Script must be a nonempty regular file of at most 64 KiB")
        data = stream.read(MAX_SCRIPT_BYTES + 1)
    if len(data) > MAX_SCRIPT_BYTES:
        raise ValueError("Script exceeds 64 KiB")
    data.decode("utf-8")
    return data


class Queue:
    def __init__(self, state_dir: Path):
        self.root = Path(state_dir).resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.db = sqlite3.connect(self.root / "queue.sqlite3")
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS queue_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS jobs(
                id TEXT PRIMARY KEY, hypothesis TEXT NOT NULL, sha256 TEXT NOT NULL,
                timeout_seconds INTEGER NOT NULL, status TEXT NOT NULL,
                created_at REAL NOT NULL, started_at REAL, finished_at REAL,
                result TEXT, error TEXT
            );
        """)
        self.db.execute("INSERT OR IGNORE INTO queue_meta VALUES('owner', ?)", (uuid.uuid4().hex,))
        self.db.commit()
        self.owner = self.db.execute("SELECT value FROM queue_meta WHERE key='owner'").fetchone()[0]

    def enqueue(self, script: Path, hypothesis: str, timeout_seconds: int = DEFAULT_JOB_TIMEOUT) -> str:
        if not 1 <= len(hypothesis) <= 2000:
            raise ValueError("Hypothesis must contain 1–2000 characters")
        if not 1 <= timeout_seconds <= 86400:
            raise ValueError("Timeout must be between 1 second and 24 hours")
        data = read_script(script)
        job_id = uuid.uuid4().hex
        input_dir = self.root / "queue-inputs" / job_id
        input_dir.mkdir(parents=True, mode=0o700)
        train = input_dir / "train.py"
        with train.open("xb") as stream:
            stream.write(data)
        digest = hashlib.sha256(data).hexdigest()
        with self.db:
            self.db.execute(
                "INSERT INTO jobs(id,hypothesis,sha256,timeout_seconds,status,created_at) VALUES(?,?,?,?,?,?)",
                (job_id, hypothesis, digest, timeout_seconds, "queued", time.time()),
            )
        return job_id

    def recover(self) -> None:
        with self.db:
            self.db.execute(
                "UPDATE jobs SET status='interrupted', error='Worker restarted', finished_at=? WHERE status='running'",
                (time.time(),),
            )

    def next_job(self):
        with self.db:
            row = self.db.execute("SELECT * FROM jobs WHERE status='queued' ORDER BY created_at,id LIMIT 1").fetchone()
            if row is None:
                return None
            self.db.execute("UPDATE jobs SET status='running', started_at=? WHERE id=?", (time.time(), row["id"]))
        return dict(row)

    def finish(self, job_id: str, outcome: dict) -> None:
        status = outcome["status"]
        if status not in {"succeeded", "failed", "interrupted"}:
            raise ValueError("Invalid worker status")
        with self.db:
            self.db.execute(
                "UPDATE jobs SET status=?, result=?, error=?, finished_at=? WHERE id=? AND status='running'",
                (status, json.dumps(outcome.get("result")), outcome.get("error"), time.time(), job_id),
            )

    def status(self, limit: int = 20) -> dict:
        counts = {row[0]: row[1] for row in self.db.execute("SELECT status,COUNT(*) FROM jobs GROUP BY status")}
        rows = self.db.execute(
            "SELECT id,hypothesis,sha256,timeout_seconds,status,created_at,started_at,finished_at,result,error "
            "FROM jobs ORDER BY created_at DESC,id DESC LIMIT ?", (limit,),
        )
        jobs = []
        for row in rows:
            item = dict(row)
            item["result"] = json.loads(item["result"]) if item["result"] else None
            item["output_dir"] = str(self.root / "queue-outputs" / item["id"])
            jobs.append(item)
        return {"counts": counts, "jobs": jobs}

    def close(self):
        self.db.close()


@contextmanager
def worker_lock(root: Path):
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (root / "queue-worker.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def run_job(queue: Queue, row: dict, stop: threading.Event, *, image: str, gpu: str | None) -> dict:
    job_id = row["id"]
    input_dir = queue.root / "queue-inputs" / job_id
    data = read_script(input_dir / "train.py")
    if hashlib.sha256(data).hexdigest() != row["sha256"]:
        return {"status": "failed", "error": "Queued script changed after enqueue"}
    output_dir = queue.root / "queue-outputs" / job_id
    output_dir.mkdir(parents=True, mode=0o700)
    runner = DockerRunner(queue.owner, image=image, gpu=gpu, timeout=row["timeout_seconds"],
                          toolkit=os.environ.get("NEURAL_LAB_TOOLKIT_HOST_PATH"))
    return runner.run(job_id, input_dir, output_dir, stop)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE)
    commands = parser.add_subparsers(dest="command", required=True)
    enqueue = commands.add_parser("enqueue")
    enqueue.add_argument("script", type=Path)
    enqueue.add_argument("--hypothesis", required=True)
    enqueue.add_argument("--timeout-seconds", type=int, default=DEFAULT_JOB_TIMEOUT)
    status = commands.add_parser("status")
    status.add_argument("--limit", type=int, default=20)
    worker = commands.add_parser("worker")
    worker.add_argument("--once", action="store_true")
    worker.add_argument("--idle-seconds", type=float, default=30)
    worker.add_argument("--image", default=os.environ.get("NEURAL_LAB_WORKER_IMAGE", "pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime"))
    worker.add_argument("--gpu", default="0")
    worker.add_argument("--cpu", action="store_true")
    args = parser.parse_args(argv)
    os.umask(0o077)
    if args.command == "status" and not (args.state_dir / "queue.sqlite3").exists():
        print(json.dumps({"counts": {}, "jobs": []}))
        return 0
    queue = Queue(args.state_dir)
    try:
        if args.command == "enqueue":
            print(json.dumps({"id": queue.enqueue(args.script, args.hypothesis, args.timeout_seconds), "status": "queued"}))
            return 0
        if args.command == "status":
            if args.limit < 1:
                parser.error("--limit must be positive")
            print(json.dumps(queue.status(args.limit)))
            return 0
        if os.getuid() == 0:
            parser.error("Run worker as a regular Linux user")
        if args.idle_seconds < 0:
            parser.error("--idle-seconds must be nonnegative")
        gpu = None if args.cpu else args.gpu
        stop = threading.Event()
        signal.signal(signal.SIGINT, lambda *_: stop.set())
        signal.signal(signal.SIGTERM, lambda *_: stop.set())
        with worker_lock(queue.root):
            DockerRunner(queue.owner, image=args.image, gpu=gpu).recover()
            queue.recover()
            checks = check_readiness(
                state_dir=queue.root, image=args.image, planner="demo", auth_dir=None,
                gpu=gpu, minimum_gb=10,
            )
            if not all(item["ok"] for item in checks):
                print(json.dumps({"status": "preflight_failed", "checks": checks}))
                return 1
            while not stop.is_set():
                row = queue.next_job()
                if row is None:
                    if args.once:
                        break
                    stop.wait(args.idle_seconds)
                    continue
                try:
                    outcome = run_job(queue, row, stop, image=args.image, gpu=gpu)
                except CleanupError:
                    queue.finish(row["id"], {"status": "failed", "error": "Container cleanup failed; worker halted"})
                    raise
                except Exception as exc:
                    outcome = {"status": "failed", "error": type(exc).__name__}
                queue.finish(row["id"], outcome)
                print(json.dumps({"id": row["id"], **outcome}), flush=True)
                if args.once:
                    break
        return 0
    finally:
        queue.close()


if __name__ == "__main__":
    sys.exit(main())
