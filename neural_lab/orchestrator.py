"""Headless ChatGPT Codex research coordinator. Runs inside the Codex container."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque

from neural_lab.research_direction import RESEARCH_DIRECTION

CONTEXT_HANDOFF_REASON = "context exceeded 60% of reported window"

PI_ADDENDUM = """

Backend note: this orchestrator session is driven through the pi coding agent
over its RPC protocol (pi --mode rpc). Where the instructions above reference
Codex subagents, use your pi subagents and extensions instead and never invoke
Codex itself. All queueing, verification, provenance, and handoff requirements
are unchanged."""

PROMPT = """You are the neural-lab research orchestrator. Work in /repo. Use Codex
subagents for independent literature research, experiment planning, implementation,
and independent review. Delegate those tasks rather than doing all work yourself.
Read /repo/docs/research-direction.md before planning, then read
/repo/.neural-lab/notes/latest-session.json and earlier notes as historical context.
The current research direction below takes precedence over old handoffs and
unfinished legacy campaigns. Write a transition note with the new objective and
the first verification campaign; do not delete results or cancel existing jobs.
Your objective is reproducible neural reasoning experiments with independently
checkable logical evidence. You may edit code and research with internet
access. Queue deterministic training jobs with `python3 -m neural_lab.queue enqueue
SCRIPT --hypothesis TEXT`; never execute training yourself. Review existing results
with `python3 -m neural_lab.queue status`. Keep handoff notes in
/repo/.neural-lab/notes/ (ignored by Git). Do not put credentials or personal data
in the public repository. Do not use API keys or paid API credits. Do not modify
the trusted worker or its Docker boundary to give a training script host access.
Require subagents to write their findings to /repo/.neural-lab/notes/. Update
/repo/.neural-lab/notes/ after each completed research result or queued
experiment. Keep notes durable throughout the session: the controller hands off
above 60% of the server-reported context window. Finish this session with a concise
handoff note. A fresh session will read the notes and continue later.

The operator wants substantive training per scientist session. Synthetic program
graphs are a primary benchmark for this direction; tiny CPU screens alone do not
establish neural reasoning gains. Exact CPU checker and data-generation work is
required before training and must not be replaced by unnecessary GPU work.
Prioritize substantive CUDA campaigns on the RTX 2070 (8 GiB VRAM): aim for
roughly 1–4 hours of useful training per queued substantive job; longer experiments
are authorized when scientifically justified. Set an explicit per-job timeout
with headroom for downloads, evaluation, and artifact writes, up to the queue's
86,400-second (24-hour) maximum. The old 10–20-minute target and 1,800-second cap
are superseded. Increase
dataset size, model capacity, minibatch work, and optimization budget where the
hypothesis warrants it; never add sleeps, redundant fits, or arbitrary work just
to meet a runtime target. Use public real-world benchmarks only when they test
the verification objective, and label synthetic results with their formal scope.
Substantive jobs must assert torch.cuda.is_available(), move models and training
tensors to CUDA, and report device, training_seconds, and peak allocated GPU
memory. Do not silently fall back to CPU or freeze a CPU-only protocol for the
next main campaign. CPU data preparation and small correctness checks are fine.
Freeze fair controls, seed replicates, splits, evaluation, and compute budgets
before training. Queue 2–4 justified complementary jobs per campaign when useful,
so the independent worker has work during subscription pauses. Explicitly pass
--timeout-seconds with the planned budget plus overhead; do not rely on the
queue's short default. Keep one GPU job at a time and all existing boundaries.
Use learning curves and existing results to justify optimization length and
competitive baseline training. Short calibration or mechanistic diagnostics
remain appropriate, but do not substitute for substantive experiments. Freeze
longer follow-ups prospectively; preserve completed negative results rather than
extending an observed run selectively until it succeeds.
Use bounded subagent briefs and reuse their durable findings across a campaign;
avoid repeating literature reviews while a runnable, reviewed experiment waits.

Every new experiment must save reproducible data provenance and model weights,
not just scalar metrics. Use neural_lab.artifacts.ArtifactWriter('/output') for
datasets and an artifacts.json manifest. For synthetic tasks save the exact split
tensors with save_dataset and record the generator, seeds, split sizes, and hashes.
For public Hugging Face datasets record repository, resolved revision, split,
license/provenance URL, preprocessing, and exact used data. Download only inside
queued training containers. Never copy Codex credentials into a training job.
Use BestCheckpoint to retain each arm/seed's lowest observed training cross entropy
and save its selection metric, step, scope, model config, and linked dataset paths.
Preserve final held-out metrics independently; do not select weights on test data
or claim a final-only checkpoint was the best over the full training trajectory.
In experiment notes and result summaries link artifacts.json, checkpoint weights,
dataset payloads/manifests, source script, and any Hugging Face source page. State
explicitly when older jobs have no checkpoint; never invent historical weights.
For non-cross-entropy objectives, select checkpoints using a prospectively frozen
training or validation metric; record its direction and never select on test data.
""" + "\n" + RESEARCH_DIRECTION


def rpc_snapshot(timeout: int = 20) -> dict:
    """Read public app-server account and quota metadata, never auth token files."""
    process = subprocess.Popen(
        ["codex", "app-server", "--stdio"], stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    requests = [
        {"id": 1, "method": "initialize", "params": {"clientInfo": {"name": "neural-lab", "version": "1"},
                                                    "capabilities": {"experimentalApi": True}}},
        {"method": "initialized", "params": {}},
        {"id": 2, "method": "account/read", "params": {"refreshToken": False}},
        {"id": 3, "method": "account/rateLimits/read", "params": {"excludeResetCreditDetails": True, "supportsLunaReserve": False}},
    ]
    replies = {}
    pending = bytearray()
    deadline = time.monotonic() + timeout

    def send(request):
        process.stdin.write((json.dumps(request) + "\n").encode("utf-8"))
        process.stdin.flush()

    def wait_for(ids):
        while not ids.issubset(replies):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            ready, _, _ = select.select([process.stdout], [], [], remaining)
            if not ready:
                break
            chunk = os.read(process.stdout.fileno(), 65536)
            if not chunk:
                break
            pending.extend(chunk)
            while b"\n" in pending:
                line, _, rest = pending.partition(b"\n")
                pending[:] = rest
                try:
                    value = json.loads(line)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if isinstance(value, dict) and value.get("id") in (1, 2, 3):
                    replies[value["id"]] = value
    try:
        send(requests[0])
        wait_for({1})
        if 1 not in replies or "error" in replies[1]:
            raise RuntimeError("Codex app-server initialization failed")
        for request in requests[1:]:
            send(request)
        wait_for({2, 3})
    finally:
        process.stdin.close()
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        process.stdout.close()
    if 2 not in replies or 3 not in replies or "error" in replies[2] or "error" in replies[3]:
        raise RuntimeError("Codex account or usage data unavailable")
    account = replies[2].get("result", {}).get("account") or {}
    if account.get("type") != "chatgpt":
        raise RuntimeError("ChatGPT account login required; API-key auth is forbidden")
    return replies[3].get("result") or {}


def capacity_ok(snapshot: dict) -> tuple[bool, str]:
    if snapshot.get("ordinaryUsageAllowed") is False:
        return False, "included subscription usage unavailable"
    buckets = snapshot.get("rateLimitsByLimitId") or {"default": snapshot.get("rateLimits")}
    windows = []
    for bucket in buckets.values():
        if not isinstance(bucket, dict):
            continue
        for name, label, threshold in (("primary", "5-hour", 75), ("secondary", "weekly", 95)):
            window = bucket.get(name)
            if isinstance(window, dict) and type(window.get("usedPercent")) in (int, float):
                value = window["usedPercent"]
                if 0 <= value <= 100:
                    windows.append((label, value, threshold))
    if not windows:
        return False, "subscription usage percentage unavailable"
    blocked = [window for window in windows if window[1] >= window[2]]
    detail = "; ".join(f"{label} usage {value}% (pause at {threshold}%)"
                       for label, value, threshold in (blocked or windows))
    return not blocked, detail


class AppServer:
    """Small buffered JSON-line transport for the Codex app-server."""

    def __init__(self):
        self.process = subprocess.Popen(
            ["codex", "app-server", "--stdio", "-c", 'forced_login_method="chatgpt"',
             "-c", 'model_provider="openai"', "-c", 'web_search="live"',
             "-c", "check_for_update_on_startup=false", "-c", "features.multi_agent=true"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True,
        )
        self.buffer = bytearray()
        self.pending = deque()
        self.sequence = 0
        self.model_catalog_error = None
        self.stderr_reader = threading.Thread(target=self._read_diagnostics, daemon=True)
        self.stderr_reader.start()

    def _read_diagnostics(self):
        """Keep only fixed diagnostic categories, never raw CLI logs or secrets."""
        while chunk := self.process.stderr.readline(8192):
            if b"failed to refresh available models:" in chunk:
                self.model_catalog_error = (
                    "model catalog refresh timed out" if b"request timed out" in chunk
                    else "model catalog refresh failed"
                )

    def send(self, method: str, params: dict | None = None) -> int:
        self.sequence += 1
        self.process.stdin.write((json.dumps({"id": self.sequence, "method": method,
                                              "params": params or {}}) + "\n").encode())
        self.process.stdin.flush()
        return self.sequence

    def notify(self, method: str, params: dict | None = None):
        self.process.stdin.write((json.dumps({"method": method, "params": params or {}}) + "\n").encode())
        self.process.stdin.flush()

    def receive(self, timeout: float) -> dict | None:
        if self.pending:
            return self.pending.popleft()
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            ready, _, _ = select.select([self.process.stdout], [], [], remaining)
            if not ready:
                return None
            chunk = os.read(self.process.stdout.fileno(), 65536)
            if not chunk:
                raise RuntimeError("Codex app-server closed unexpectedly")
            self.buffer.extend(chunk)
            while b"\n" in self.buffer:
                line, _, rest = self.buffer.partition(b"\n")
                self.buffer[:] = rest
                try:
                    value = json.loads(line)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if isinstance(value, dict):
                    self.pending.append(value)
            if self.pending:
                return self.pending.popleft()

    def request(self, method: str, params: dict | None = None, timeout: float = 30) -> dict:
        request_id = self.send(method, params)
        deadline = time.monotonic() + timeout
        deferred = []
        try:
            while time.monotonic() < deadline:
                message = self.receive(max(0.001, deadline - time.monotonic()))
                if message is None:
                    break
                if message.get("id") == request_id:
                    if "error" in message:
                        raise RuntimeError(f"Codex {method} request failed")
                    return message.get("result") or {}
                deferred.append(message)
            raise RuntimeError(f"Codex {method} request timed out")
        finally:
            self.pending.extendleft(reversed(deferred))

    def close(self):
        try:
            self.process.stdin.close()
        except BrokenPipeError:
            pass
        try:
            os.killpg(self.process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.process.wait()
        self.process.stdout.close()
        self.stderr_reader.join(timeout=1)
        self.process.stderr.close()


def context_usage(notification: dict) -> tuple[int, int]:
    """Read the active call's tokens and server-reported effective context window."""
    usage = notification.get("params", {}).get("tokenUsage", {})
    window = usage.get("modelContextWindow")
    last = usage.get("last", {})
    used = last.get("totalTokens")
    if type(window) is not int or window <= 0:
        observed = str(window) if type(window) is int else "unavailable"
        raise RuntimeError(f"Codex did not report a positive context window (reported {observed})")
    if type(used) is not int or used < 0:
        raise RuntimeError("Codex did not report live context usage")
    return used, window


def context_fraction(notification: dict) -> float:
    used, window = context_usage(notification)
    return used / window


def write_handoff(repo: Path, state: dict, *, filename: str = "latest-session.json") -> None:
    """Persist a bounded private checkpoint before interrupting a live turn."""
    notes = Path(repo) / ".neural-lab" / "notes"
    notes.mkdir(parents=True, exist_ok=True, mode=0o700)
    data = json.dumps(state, ensure_ascii=False, indent=2).encode("utf-8")
    with tempfile.NamedTemporaryFile(dir=notes, prefix=".latest-session-", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            os.chmod(temporary, 0o600)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, notes / filename)
    directory_fd = os.open(notes, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def previous_handoff(repo: Path) -> str:
    """Snapshot the prior private handoff before replacing latest-session.json."""
    path = Path(repo) / ".neural-lab" / "notes" / "latest-session.json"
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        return ""
    with os.fdopen(fd, "rb") as stream:
        raw = stream.read(16385)
    if len(raw) > 16384:
        return "Previous handoff was too large; inspect notes manually."
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return "Previous handoff was unreadable; inspect notes manually."
    return json.dumps(value, ensure_ascii=False)


def run_session(model: str, repo: Path, prompt: str, *, usage_poll_seconds: int = 60,
                usage_timeout_seconds: float = 20,
                stop: threading.Event | None = None) -> dict:
    """Run one fresh turn, interrupting on live context or subscription limits."""
    server = AppServer()
    stop = stop or threading.Event()
    try:
        old_handoff = previous_handoff(repo)
        if old_handoff:
            prompt += "\nPrevious session handoff (private local state):\n" + old_handoff
        server.request("initialize", {"clientInfo": {"name": "neural-lab", "version": "1"},
                                      "capabilities": {"experimentalApi": True}})
        server.notify("initialized")
        if stop.is_set():
            return {"exit_code": 1, "reason": "shutdown requested"}
        started = server.request("thread/start", {
            "cwd": str(repo), "model": model, "modelProvider": "openai",
            "approvalPolicy": "never", "sandbox": "danger-full-access", "ephemeral": True,
            "config": {"forced_login_method": "chatgpt", "model_provider": "openai",
                       "web_search": "live", "features": {"multi_agent": True}},
        })
        thread_id = started.get("thread", {}).get("id")
        if not thread_id or started.get("modelProvider") != "openai":
            raise RuntimeError("Codex did not start an OpenAI subscription thread")
        turn = server.request("turn/start", {
            "threadId": thread_id, "input": [{"type": "text", "text": prompt, "text_elements": []}],
        }).get("turn") or {}
        turn_id = turn.get("id")
        if not turn_id:
            raise RuntimeError("Codex did not return a turn ID")
        reason = None
        fraction = None
        used_tokens = None
        context_window = None
        telemetry_seen = False
        recent_messages = deque(maxlen=4)
        handoff = {"thread_id": thread_id, "turn_id": turn_id, "model": model,
                   "reason": "running", "last_tokens": None, "model_context_window": None,
                   "context_handoff_fraction": 0.6, "context_handoff_tokens": None,
                   "recent_agent_messages": [], "updated_at": time.time()}
        write_handoff(repo, handoff)
        next_usage_poll = time.monotonic() + usage_poll_seconds
        poll_id = None
        poll_deadline = None
        interrupt_deadline = None
        while True:
            now = time.monotonic()
            if stop.is_set():
                reason = reason or "shutdown requested"
            if reason and interrupt_deadline and now >= interrupt_deadline:
                raise RuntimeError("Codex turn did not stop after interruption")
            if poll_id is not None and poll_deadline is not None and now >= poll_deadline:
                reason = "subscription usage check timed out"
            if reason and interrupt_deadline is None:
                handoff.update(reason=reason, last_tokens=used_tokens,
                               model_context_window=context_window,
                               recent_agent_messages=list(recent_messages), updated_at=time.time())
                write_handoff(repo, handoff)
                server.send("turn/interrupt", {"threadId": thread_id, "turnId": turn_id})
                interrupt_deadline = time.monotonic() + 30
            if not reason and now >= next_usage_poll and poll_id is None:
                poll_id = server.send("account/rateLimits/read", {
                    "excludeResetCreditDetails": True, "supportsLunaReserve": False,
                })
                next_usage_poll = now + usage_poll_seconds
                poll_deadline = now + usage_timeout_seconds
            message = server.receive(min(1.0, max(0.001, next_usage_poll - now)) if not reason else 1.0)
            if message is None:
                continue
            if poll_id is not None and message.get("id") == poll_id:
                poll_id = None
                poll_deadline = None
                if "error" in message:
                    reason = "subscription usage check failed"
                else:
                    permitted, detail = capacity_ok(message.get("result") or {})
                    if not permitted:
                        reason = detail
            elif message.get("method") == "account/rateLimits/updated":
                update = message.get("params") or {}
                permitted, detail = capacity_ok(update)
                if not permitted and detail != "subscription usage percentage unavailable":
                    reason = detail
                elif poll_id is None:
                    poll_id = server.send("account/rateLimits/read", {
                        "excludeResetCreditDetails": True, "supportsLunaReserve": False,
                    })
                    poll_deadline = time.monotonic() + usage_timeout_seconds
            elif message.get("method") == "thread/tokenUsage/updated":
                params = message.get("params") or {}
                if params.get("threadId") != thread_id or params.get("turnId") != turn_id:
                    continue
                telemetry_seen = True
                try:
                    used_tokens, context_window = context_usage(message)
                    fraction = used_tokens / context_window
                except RuntimeError as exc:
                    reason = str(exc)
                    fraction = None
                    # Preserve the rejected numeric observation for diagnosis instead
                    # of losing it when tuple assignment above raises.
                    raw = params.get("tokenUsage") or {}
                    reported_window = raw.get("modelContextWindow")
                    reported_used = (raw.get("last") or {}).get("totalTokens")
                    context_window = reported_window if type(reported_window) is int else None
                    used_tokens = reported_used if type(reported_used) is int else None
                    catalog_error = getattr(server, "model_catalog_error", None)
                    if catalog_error:
                        reason += "; " + catalog_error
                        handoff["model_catalog_error"] = catalog_error
                else:
                    # Integer comparison keeps the strict >60% boundary exact.
                    if used_tokens * 5 > context_window * 3:
                        reason = CONTEXT_HANDOFF_REASON
                handoff.update(last_tokens=used_tokens, model_context_window=context_window,
                               context_handoff_tokens=context_window * 3 / 5 if context_window and context_window > 0 else None,
                               reason=reason or "running", updated_at=time.time())
                write_handoff(repo, handoff)
            elif message.get("method") == "item/completed":
                params = message.get("params") or {}
                if params.get("threadId") != thread_id or params.get("turnId") != turn_id:
                    continue
                item = params.get("item") or {}
                if item.get("type") == "agentMessage" and isinstance(item.get("text"), str):
                    recent_messages.append(item["text"][-2000:])
                    handoff.update(recent_agent_messages=list(recent_messages), updated_at=time.time())
                    write_handoff(repo, handoff)
            elif message.get("method") == "turn/completed":
                params = message.get("params") or {}
                if params.get("threadId") != thread_id or params.get("turn", {}).get("id") != turn_id:
                    continue
                if not telemetry_seen:
                    reason = reason or "live context telemetry unavailable"
                status = params["turn"].get("status")
                handoff.update(reason=reason or status, last_tokens=used_tokens,
                               model_context_window=context_window,
                               recent_agent_messages=list(recent_messages), updated_at=time.time())
                write_handoff(repo, handoff)
                success = (status == "completed" and reason is None) or reason == CONTEXT_HANDOFF_REASON
                return {"exit_code": 0 if success else 1,
                        "reason": reason or status, "context_fraction": fraction,
                        "context_window_verified": telemetry_seen and fraction is not None,
                        "last_tokens": used_tokens, "model_context_window": context_window,
                        "context_handoff_fraction": 0.6,
                        "context_handoff_tokens": handoff["context_handoff_tokens"]}
    except Exception as exc:
        if "handoff" in locals():
            handoff.update(reason="error: " + type(exc).__name__, updated_at=time.time())
            write_handoff(repo, handoff)
        raise
    finally:
        server.close()


class PiSession:
    """Small buffered JSON-line transport for the pi RPC backend."""

    def __init__(self, model=None):
        args = ["pi", "--mode", "rpc", "--no-session"]
        if model:
            args.extend(["--model", model])
        self.process = subprocess.Popen(
            args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True,
        )
        self.buffer = bytearray()
        self.pending = deque()
        self.sequence = 0
        self.auth_error = None
        self.stderr_reader = threading.Thread(target=self._read_diagnostics, daemon=True)
        self.stderr_reader.start()

    def _read_diagnostics(self):
        """Keep only fixed diagnostic categories, never raw CLI logs or secrets."""
        while chunk := self.process.stderr.readline(8192):
            lowered = chunk.lower()
            if b"unauthorized" in lowered or b"api key" in lowered:
                self.auth_error = "pi could not authenticate with the configured provider"

    def send(self, record_type: str, fields: dict | None = None) -> str:
        self.sequence += 1
        request_id = f"neural-lab-{self.sequence}"
        record = {"id": request_id, "type": record_type, **(fields or {})}
        self.process.stdin.write((json.dumps(record) + "\n").encode())
        self.process.stdin.flush()
        return request_id

    def notify(self, record_type: str, fields: dict | None = None):
        record = {"type": record_type, **(fields or {})}
        self.process.stdin.write((json.dumps(record) + "\n").encode())
        self.process.stdin.flush()

    def receive(self, timeout: float) -> dict | None:
        if self.pending:
            return self.pending.popleft()
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            ready, _, _ = select.select([self.process.stdout], [], [], remaining)
            if not ready:
                return None
            chunk = os.read(self.process.stdout.fileno(), 65536)
            if not chunk:
                raise RuntimeError("pi RPC process closed unexpectedly")
            self.buffer.extend(chunk)
            while b"\n" in self.buffer:
                line, _, rest = self.buffer.partition(b"\n")
                self.buffer[:] = rest
                try:
                    value = json.loads(line)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if isinstance(value, dict):
                    self.pending.append(value)
            if self.pending:
                return self.pending.popleft()

    def request(self, record_type: str, fields: dict | None = None, timeout: float = 30) -> dict:
        request_id = self.send(record_type, fields)
        deadline = time.monotonic() + timeout
        deferred = []
        try:
            while time.monotonic() < deadline:
                message = self.receive(max(0.001, deadline - time.monotonic()))
                if message is None:
                    break
                if message.get("type") == "response" and message.get("id") == request_id:
                    if not message.get("success"):
                        raise RuntimeError(f"pi {record_type} command failed")
                    return message.get("data") or {}
                deferred.append(message)
            raise RuntimeError(f"pi {record_type} command timed out")
        finally:
            self.pending.extendleft(reversed(deferred))

    def close(self):
        try:
            self.process.stdin.close()
        except BrokenPipeError:
            pass
        try:
            os.killpg(self.process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.process.wait()
        self.process.stdout.close()
        self.stderr_reader.join(timeout=1)
        self.process.stderr.close()


def pi_usage(message: dict) -> tuple[int | None, int | None]:
    """Read the reported token usage and context window from an RPC event."""
    usage = message.get("usage") or {}
    used = usage.get("totalTokens")
    window = usage.get("contextWindow")
    if type(used) is not int or used < 0 or type(window) is not int or window <= 0:
        return None, None
    return used, window


def run_pi_session(model: str, repo: Path, prompt: str, *, stop: threading.Event | None = None,
                   max_session_seconds: int = 14400) -> dict:
    """Run one pi RPC turn, interrupting on missing context telemetry or limits."""
    session = PiSession(model)
    stop = stop or threading.Event()
    try:
        old_handoff = previous_handoff(repo)
        if old_handoff:
            prompt += "\nPrevious session handoff (private local state):\n" + old_handoff
        deadline = time.monotonic() + max_session_seconds
        handoff = {"backend": "pi", "model": model, "reason": "running", "last_tokens": None,
                   "model_context_window": None, "context_handoff_fraction": 0.6,
                   "context_handoff_tokens": None, "updated_at": time.time()}
        write_handoff(repo, handoff)
        prompt_id = session.send("prompt", {"message": prompt})
        reason = None
        used_tokens = None
        context_window = None
        aborted = False
        grace = 0
        outcome = None
        while outcome is None:
            now = time.monotonic()
            if stop.is_set():
                reason = reason or "shutdown requested"
            if now >= deadline:
                reason = reason or "pi session exceeded its time budget"
            if reason and not aborted:
                handoff.update(reason=reason, last_tokens=used_tokens,
                               model_context_window=context_window,
                               context_handoff_tokens=context_window * 3 / 5 if context_window else None,
                               updated_at=time.time())
                write_handoff(repo, handoff)
                session.notify("abort")
                aborted = True
            message = session.receive(min(1.0, max(0.001, deadline - now)))
            if message is None:
                if aborted:
                    # Bounded grace for the settled confirmation after an abort.
                    grace += 1
                    if grace > 30:
                        outcome = {"exit_code": 1, "reason": reason or "pi session aborted",
                                   "context_fraction": None,
                                   "context_window_verified": used_tokens is not None and context_window is not None,
                                   "last_tokens": used_tokens, "model_context_window": context_window,
                                   "context_handoff_fraction": 0.6,
                                   "context_handoff_tokens": handoff["context_handoff_tokens"]}
                continue
            method = message.get("type")
            if method == "response" and message.get("id") == prompt_id and not message.get("success"):
                reason = reason or "pi rejected the prompt command"
            elif method in ("message_update", "message_end", "turn_end"):
                observed_used, observed_window = pi_usage(message)
                if observed_used is None:
                    reason = reason or "pi context telemetry unavailable"
                else:
                    used_tokens, context_window = observed_used, observed_window
                    handoff.update(last_tokens=used_tokens, model_context_window=context_window,
                                   context_handoff_tokens=context_window * 3 / 5,
                                   updated_at=time.time())
                    write_handoff(repo, handoff)
                    # Integer comparison keeps the strict >60% boundary exact.
                    if used_tokens * 5 > context_window * 3:
                        reason = reason or CONTEXT_HANDOFF_REASON
            elif method == "agent_settled":
                verified = used_tokens is not None and context_window is not None
                if reason == CONTEXT_HANDOFF_REASON:
                    success = verified
                else:
                    success = verified and not message.get("aborted", False) and reason is None
                outcome = {"exit_code": 0 if success else 1,
                           "reason": reason or "pi session settled",
                           "context_fraction": used_tokens / context_window if verified else None,
                           "context_window_verified": verified,
                           "last_tokens": used_tokens, "model_context_window": context_window,
                           "context_handoff_fraction": 0.6,
                           "context_handoff_tokens": handoff["context_handoff_tokens"]}
        handoff.update(reason=outcome["reason"], last_tokens=used_tokens,
                       model_context_window=context_window, updated_at=time.time())
        write_handoff(repo, handoff)
        return outcome
    except Exception as exc:
        if "handoff" in locals():
            handoff.update(reason="error: " + type(exc).__name__, updated_at=time.time())
            write_handoff(repo, handoff)
        raise
    finally:
        session.close()


def budget_ok(state_path, *, max_seconds, max_sessions) -> tuple[bool, str]:
    """Conservative read-only gate over the local pi usage budget window."""
    try:
        value = json.loads(Path(state_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False, "pi usage budget state unavailable"
    if not isinstance(value, dict):
        return False, "pi usage budget state unavailable"
    sessions = value.get("sessions")
    active = value.get("active_seconds")
    if type(sessions) is not int or sessions < 0 or type(active) not in (int, float) or active < 0:
        return False, "pi usage budget state unavailable"
    if sessions >= max_sessions:
        return False, f"pi session budget reached ({sessions}/{max_sessions})"
    if active >= max_seconds:
        return False, f"pi active-second budget reached ({active:g}/{max_seconds:g})"
    return True, f"pi budget: {sessions}/{max_sessions} sessions, {active:g}/{max_seconds:g} active seconds"


def record_pi_budget(repo: Path, seconds: float) -> None:
    """Add one settled session to the rolling pi budget window after it ends."""
    path = Path(repo) / ".neural-lab" / "notes" / "pi-budget.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError
    except (OSError, ValueError):
        value = {}
    now = time.time()
    window_start = value.get("window_start")
    if type(window_start) not in (int, float) or window_start < now - 86400:
        value = {"sessions": 0, "active_seconds": 0, "window_start": now}
    value["sessions"] = int(value.get("sessions", 0)) + 1
    value["active_seconds"] = value.get("active_seconds", 0) + max(0, seconds)
    write_handoff(repo, value, filename="pi-budget.json")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("pi", "codex"),
                        default=os.environ.get("NEURAL_LAB_ORCH_BACKEND", "codex"))
    parser.add_argument("--model", default=os.environ.get("NEURAL_LAB_ORCH_MODEL", "gpt-6.1-sol"))
    parser.add_argument("--repo", type=Path, default=Path("/repo"))
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--pi-budget-seconds", type=int,
                        default=int(os.environ.get("NEURAL_LAB_PI_BUDGET_SECONDS", "14400")))
    parser.add_argument("--pi-budget-sessions", type=int,
                        default=int(os.environ.get("NEURAL_LAB_PI_BUDGET_SESSIONS", "8")))
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    if not args.model:
        parser.error("Set a ChatGPT subscription-backed Codex model")
    if args.poll_seconds < 1:
        parser.error("--poll-seconds must be positive")
    if min(args.pi_budget_seconds, args.pi_budget_sessions) < 1:
        parser.error("--pi-budget-seconds and --pi-budget-sessions must be positive")
    notes = args.repo / ".neural-lab" / "notes"
    notes.mkdir(parents=True, exist_ok=True, mode=0o700)
    stop = threading.Event()
    prior_handlers = {name: signal.getsignal(name) for name in (signal.SIGINT, signal.SIGTERM)}
    for name in prior_handlers:
        signal.signal(name, lambda *_: stop.set())
    try:
        while not stop.is_set():
            if args.backend == "pi":
                permitted, reason = budget_ok(
                    args.repo / ".neural-lab" / "notes" / "pi-budget.json",
                    max_seconds=args.pi_budget_seconds, max_sessions=args.pi_budget_sessions,
                )
            else:
                try:
                    permitted, reason = capacity_ok(rpc_snapshot())
                except Exception as exc:
                    permitted, reason = False, "usage check error: " + type(exc).__name__
            if stop.is_set():
                break
            if not permitted:
                write_handoff(args.repo, {"reason": reason, "updated_at": time.time()},
                              filename="pause-state.json")
                print(json.dumps({"orchestrator": "paused", "reason": reason}), flush=True)
                code = 1
            else:
                print(json.dumps({"orchestrator": "starting", "model": args.model, "usage": reason}), flush=True)
                try:
                    prompt = PROMPT
                    if args.backend == "pi":
                        prompt += PI_ADDENDUM
                        started = time.monotonic()
                        outcome = run_pi_session(args.model, args.repo, prompt, stop=stop)
                        record_pi_budget(args.repo, time.monotonic() - started)
                    else:
                        outcome = run_session(args.model, args.repo, prompt, stop=stop)
                except Exception as exc:
                    outcome = {"exit_code": 1, "reason": "session error: " + type(exc).__name__}
                code = outcome["exit_code"]
                print(json.dumps({"orchestrator": "session_finished", **outcome}), flush=True)
            if args.once:
                return 0 if permitted and code == 0 else 1
            stop.wait(args.poll_seconds)
        write_handoff(args.repo, {"reason": "shutdown requested", "updated_at": time.time()},
                      filename="pause-state.json")
        return 0
    finally:
        for name, handler in prior_handlers.items():
            signal.signal(name, handler)


if __name__ == "__main__":
    sys.exit(main())
