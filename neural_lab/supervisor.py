"""Bounded persistent research loop. Run on the trusted Linux host."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import sqlite3
import sys
import threading
import time
import uuid

from neural_lab.planner import APIPlanner, DemoPlanner, PlannerError
from neural_lab.runner import CleanupError, DockerRunner


class BudgetExhausted(RuntimeError):
    pass


class Journal:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.db = sqlite3.connect(self.root / "journal.sqlite3")
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY, attempt INTEGER NOT NULL,
                status TEXT NOT NULL, reserved_seconds INTEGER NOT NULL,
                hypothesis TEXT, result TEXT, error TEXT,
                created_at REAL NOT NULL, finished_at REAL
            );
        """)
        self.db.execute("INSERT OR IGNORE INTO metadata VALUES ('owner', ?)", (uuid.uuid4().hex,))
        self.db.commit()
        self.owner = self.db.execute("SELECT value FROM metadata WHERE key='owner'").fetchone()[0]

    def reserve(self, *, max_attempts, budget_seconds, timeout):
        # Reservations are intentionally not refunded. Crashes and planner errors
        # cannot silently replenish API-attempt or compute budgets.
        with self.db:
            count, reserved = self.db.execute(
                "SELECT COUNT(*), COALESCE(SUM(reserved_seconds),0) FROM runs"
            ).fetchone()
            if count >= max_attempts or reserved + timeout > budget_seconds:
                raise BudgetExhausted("Lifetime attempt or reserved-runtime budget reached")
            run_id = uuid.uuid4().hex
            self.db.execute(
                "INSERT INTO runs(id,attempt,status,reserved_seconds,created_at) VALUES(?,?,?,?,?)",
                (run_id, count + 1, "planning", timeout, time.time()),
            )
        return run_id, count + 1

    def update(self, run_id, status, *, hypothesis=None, result=None, error=None):
        final = status in {"succeeded", "failed", "interrupted"}
        with self.db:
            self.db.execute(
                """UPDATE runs SET status=?, hypothesis=COALESCE(?,hypothesis),
                   result=COALESCE(?,result),error=?,finished_at=? WHERE id=?""",
                (status, hypothesis, json.dumps(result) if result is not None else None,
                 error, time.time() if final else None, run_id),
            )

    def recover(self):
        with self.db:
            self.db.execute(
                """UPDATE runs SET status='interrupted', error='Supervisor restart',
                   finished_at=? WHERE status IN ('planning','planned','running')""",
                (time.time(),),
            )

    def history(self):
        rows = self.db.execute(
            "SELECT attempt,status,hypothesis,result,error FROM runs ORDER BY attempt DESC LIMIT 5"
        ).fetchall()
        return [
            {**dict(row), "result": json.loads(row["result"]) if row["result"] else None}
            for row in reversed(rows)
        ]

    def failures(self):
        count = 0
        for row in self.db.execute("SELECT status FROM runs ORDER BY attempt DESC"):
            if row["status"] == "succeeded":
                break
            if row["status"] in {"failed", "interrupted"}:
                count += 1
        return count

    def close(self):
        self.db.close()


@contextmanager
def instance_lock(root):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (root / "supervisor.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another supervisor is using this state directory") from None
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def experiment_step(journal, planner, runner, stop, *, max_attempts, budget_seconds, timeout):
    run_id, attempt = journal.reserve(
        max_attempts=max_attempts, budget_seconds=budget_seconds, timeout=timeout,
    )
    try:
        history = [row for row in journal.history() if row["attempt"] != attempt]
        plan = planner.plan(history, attempt)
        if stop.is_set():
            journal.update(run_id, "interrupted", error="Shutdown requested")
            return "interrupted"
        # Only these two directories enter the worker. The database, owner ID,
        # planner credentials, previous runs, and code checkout remain outside.
        inputs = journal.root / "inputs" / run_id
        outputs = journal.root / "outputs" / run_id
        inputs.mkdir(parents=True, mode=0o700)
        outputs.mkdir(parents=True, mode=0o700)
        (inputs / "train.py").write_text(plan.code, encoding="utf-8")
        (inputs / "hypothesis.txt").write_text(plan.hypothesis, encoding="utf-8")
        journal.update(run_id, "planned", hypothesis=plan.hypothesis)
        journal.update(run_id, "running")
        outcome = runner.run(run_id, inputs, outputs, stop)
        journal.update(run_id, outcome["status"], result=outcome.get("result"), error=outcome.get("error"))
        print(json.dumps({"run_id": run_id, "attempt": attempt, **outcome}), flush=True)
        return outcome["status"]
    except CleanupError:
        journal.update(run_id, "failed", error="Container cleanup failed; supervisor halted")
        raise
    except Exception as exc:
        # Record bounded diagnostics. Avoid including credentials or raw API replies.
        error = str(exc)[:200] if isinstance(exc, PlannerError) else type(exc).__name__
        journal.update(run_id, "failed", error=error)
        print(json.dumps({"run_id": run_id, "attempt": attempt, "status": "failed", "error": error}), flush=True)
        return "failed"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, default=Path(".neural-lab"))
    parser.add_argument("--planner", choices=("demo", "api"), default="demo")
    parser.add_argument("--model")
    parser.add_argument("--api-url", default="https://api.openai.com/v1/chat/completions")
    parser.add_argument("--api-key-env", default="NEURAL_LAB_API_KEY")
    parser.add_argument("--image", default="pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--max-attempts", type=int, default=20)
    parser.add_argument("--budget-seconds", type=int, default=7200)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--idle-seconds", type=float, default=60)
    parser.add_argument("--backoff-seconds", type=float, default=30)
    parser.add_argument("--max-consecutive-failures", type=int, default=3)
    args = parser.parse_args(argv)
    if os.getuid() == 0:
        parser.error("Run as a regular Linux user, without sudo")
    if min(args.max_attempts, args.timeout_seconds, args.max_consecutive_failures) < 1:
        parser.error("Attempt, timeout, and failure limits must be positive")
    if args.budget_seconds < args.timeout_seconds:
        parser.error("Runtime budget must cover at least one experiment timeout")
    if not all(math.isfinite(value) and value >= 0 for value in (args.idle_seconds, args.backoff_seconds)):
        parser.error("Delays must be non-negative")
    if args.planner == "api" and not args.model:
        parser.error("--model is required for the API planner")

    stop = threading.Event()
    def shutdown(signum, frame):
        stop.set()
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    # Restrict new host-side state files. Workers inherit no API credentials.
    os.umask(0o077)
    with instance_lock(args.state_dir):
        journal = Journal(args.state_dir)
        try:
            planner = DemoPlanner() if args.planner == "demo" else APIPlanner(
                args.model, url=args.api_url, key_env=args.api_key_env,
                runtime_seconds=args.timeout_seconds,
            )
            runner = DockerRunner(journal.owner, image=args.image,
                                  gpu=None if args.cpu else args.gpu, timeout=args.timeout_seconds)
            runner.recover()
            journal.recover()
            while not stop.is_set():
                failures = journal.failures()
                if failures >= args.max_consecutive_failures:
                    print("Paused: persistent consecutive-failure limit reached.", flush=True)
                    break
                # Backoff is applied before work, including after restart.
                if failures and stop.wait(min(3600, args.backoff_seconds * 2 ** min(failures - 1, 10))):
                    break
                try:
                    status = experiment_step(
                        journal, planner, runner, stop, max_attempts=args.max_attempts,
                        budget_seconds=args.budget_seconds, timeout=args.timeout_seconds,
                    )
                except BudgetExhausted as exc:
                    print(str(exc), flush=True)
                    break
                if args.once or status == "interrupted":
                    break
                if status == "succeeded":
                    stop.wait(args.idle_seconds)
        finally:
            journal.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
