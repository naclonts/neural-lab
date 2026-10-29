"""Headless ChatGPT Codex research coordinator. Runs inside the Codex container."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import time


PROMPT = """You are the neural-lab research orchestrator. Work in /repo. Use Codex
subagents for independent literature research, experiment design, and implementation.
Your objective is novel neural-network architecture experiments with reproducible
comparisons against the queued baselines. You may edit code and research with internet
access. Queue deterministic training jobs with `python3 -m neural_lab.queue enqueue
SCRIPT --hypothesis TEXT`; never execute training yourself. Review existing results
with `python3 -m neural_lab.queue status`. Keep handoff notes in
/repo/.neural-lab/notes/ (ignored by Git). Do not put credentials or personal data
in the public repository. Do not use API keys or paid API credits. Do not modify
the trusted worker or its Docker boundary to give a training script host access.
Finish this session with a concise handoff note. A new session will begin later.
"""


def rpc_snapshot(timeout: int = 20) -> dict:
    """Read public app-server account and quota metadata, never auth token files."""
    process = subprocess.Popen(
        ["codex", "app-server", "--stdio"], stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    requests = [
        {"id": 1, "method": "initialize", "params": {"clientInfo": {"name": "neural-lab", "version": "1"}}},
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
    if snapshot.get("ordinaryUsageAllowed") is not True:
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


def run_session(model: str, context_tokens: int, repo: Path, prompt: str) -> tuple[int, int]:
    """One fresh Codex session per cycle; return code and reported input tokens."""
    command = [
        "codex", "exec", "--json", "--ephemeral", "--ignore-user-config",
        "--skip-git-repo-check",
        "--dangerously-bypass-approvals-and-sandbox", "-C", str(repo),
        "--model", model, "--enable", "multi_agent",
        "-c", 'forced_login_method="chatgpt"',
        "-c", 'check_for_update_on_startup=false',
        "-",
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.DEVNULL, text=True, bufsize=1)
    process.stdin.write(prompt)
    process.stdin.close()
    last_input = 0
    for line in process.stdout:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "turn.completed":
            last_input = max(last_input, int(event.get("usage", {}).get("input_tokens", 0)))
        if event.get("type") in ("turn.failed", "error"):
            print(json.dumps({"event": event.get("type")}), flush=True)
    return process.wait(), last_input


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.environ.get("NEURAL_LAB_ORCH_MODEL"))
    parser.add_argument("--context-tokens", type=int, default=int(os.environ.get("NEURAL_LAB_CONTEXT_TOKENS", "0")))
    parser.add_argument("--repo", type=Path, default=Path("/repo"))
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    if not args.model or args.context_tokens < 1_000_000:
        parser.error("Set a subscription-backed Codex model with a VERIFIED >=1M context and NEURAL_LAB_CONTEXT_TOKENS; current default is 500k")
    notes = args.repo / ".neural-lab" / "notes"
    notes.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            permitted, reason = capacity_ok(rpc_snapshot())
        except Exception as exc:
            permitted, reason = False, str(exc)
        if not permitted:
            print(json.dumps({"orchestrator": "paused", "reason": reason}), flush=True)
        else:
            print(json.dumps({"orchestrator": "starting", "model": args.model, "usage": reason}), flush=True)
            code, tokens = run_session(args.model, args.context_tokens, args.repo, PROMPT)
            print(json.dumps({"orchestrator": "session_finished", "exit_code": code,
                              "reported_input_tokens": tokens,
                              "live_context_rollover_enforced": False}), flush=True)
        if args.once:
            return 0 if permitted and code == 0 else 1
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    sys.exit(main())
