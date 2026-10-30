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


CONTEXT_HANDOFF_REASON = "context exceeded 60% of reported window"

PROMPT = """You are the neural-lab research orchestrator. Work in /repo. Use Codex
subagents for independent literature research, experiment planning, implementation,
and independent review. Delegate those tasks rather than doing all work yourself.
Read /repo/.neural-lab/notes/latest-session.json and earlier notes before work.
Your objective is novel neural-network architecture experiments with reproducible
comparisons against the queued baselines. You may edit code and research with internet
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

The operator wants substantially more training per scientist session. Tiny
synthetic CPU screens are preliminary diagnostics, not the main research output.
Prioritize substantive CUDA campaigns on the RTX 2070 (8 GiB VRAM): aim for
roughly 10–20 minutes of useful training per queued job, with a 1,800-second
timeout covering downloads, training, evaluation, and artifact writes. Increase
dataset size, model capacity, minibatch work, and optimization budget where the
hypothesis warrants it; never add sleeps, redundant fits, or arbitrary work just
to meet a runtime target. Use public real-world Hugging Face benchmarks when
appropriate, and keep exploratory synthetic diagnostics clearly labeled.
Substantive jobs must assert torch.cuda.is_available(), move models and training
tensors to CUDA, and report device, training_seconds, and peak allocated GPU
memory. Do not silently fall back to CPU or freeze a CPU-only protocol for the
next main campaign. CPU data preparation and small correctness checks are fine.
Freeze fair controls, seed replicates, splits, evaluation, and compute budgets
before training. Queue 2–4 justified complementary jobs per campaign when useful,
so the independent worker has work during subscription pauses. Explicitly pass
--timeout-seconds 1800. Keep one GPU job at a time and all existing boundaries.
Use bounded subagent briefs and reuse their durable findings across a campaign;
avoid repeating literature reviews while a runnable, reviewed experiment waits.
The prior relational-geometry proposal is still unqueued: revise its tiny CPU
training budget and complete review under this new policy before enqueuing it.

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
"""


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


def capacity_ok(snapshot: dict, threshold: int = 75) -> tuple[bool, str]:
    if snapshot.get("ordinaryUsageAllowed") is False:
        return False, "included subscription usage unavailable"
    buckets = snapshot.get("rateLimitsByLimitId") or {"default": snapshot.get("rateLimits")}
    percentages = []
    for bucket in buckets.values():
        if not isinstance(bucket, dict):
            continue
        for name in ("primary", "secondary"):
            window = bucket.get(name)
            if isinstance(window, dict) and type(window.get("usedPercent")) in (int, float):
                value = window["usedPercent"]
                if 0 <= value <= 100:
                    percentages.append(value)
    if not percentages:
        return False, "subscription usage percentage unavailable"
    high = max(percentages)
    return high < threshold, f"highest subscription window usage {high}%"


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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.environ.get("NEURAL_LAB_ORCH_MODEL", "gpt-6.1-sol"))
    parser.add_argument("--repo", type=Path, default=Path("/repo"))
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    if not args.model:
        parser.error("Set a ChatGPT subscription-backed Codex model")
    if args.poll_seconds < 1:
        parser.error("--poll-seconds must be positive")
    notes = args.repo / ".neural-lab" / "notes"
    notes.mkdir(parents=True, exist_ok=True, mode=0o700)
    stop = threading.Event()
    prior_handlers = {name: signal.getsignal(name) for name in (signal.SIGINT, signal.SIGTERM)}
    for name in prior_handlers:
        signal.signal(name, lambda *_: stop.set())
    try:
        while not stop.is_set():
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
                    outcome = run_session(args.model, args.repo, PROMPT, stop=stop)
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
