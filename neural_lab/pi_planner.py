"""API-key-billed pi coding-agent planner, isolated from GPU workers."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import time
import uuid

from neural_lab.codex_planner import PLAN_SCHEMA
from neural_lab.planner import EXPERIMENT_INSTRUCTIONS, PlannerError, validate_plan
from neural_lab.runner import CleanupError, DockerRunner


PI_VERSION = "1.5.0"
PI_IMAGE = "neural-lab-pi:1.5.0"
FINAL_PROMPT = (
    "Read /request/request.json. Respond with ONLY the JSON plan object: exactly "
    "the keys hypothesis and code, no markdown fences, no commentary."
)


def extract_plan(text) -> dict:
    """Extract the first balanced JSON object, ignoring braces inside strings."""
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = "\n".join(
            line for line in cleaned.splitlines() if not line.strip().startswith("```")
        ).strip()
    start = cleaned.find("{")
    if start < 0:
        raise ValueError("no JSON object found")
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(cleaned)):
        char = cleaned[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return json.loads(cleaned[start:index + 1])
    raise ValueError("unbalanced JSON object")


class PiPlanner:
    def __init__(self, owner, *, auth_dir=None, model=None, timeout=600,
                 runtime_seconds=300, stop=None, image=PI_IMAGE):
        self.owner, self.model, self.timeout = owner, model, timeout
        self.runtime_seconds, self.image = runtime_seconds, image
        self.stop = stop if stop is not None else threading.Event()
        self.auth_dir = Path(auth_dir or os.environ.get("NEURAL_LAB_PI_HOME") or Path.home() / ".local/share/neural-lab/pi").resolve()
        if not self.auth_dir.is_dir():
            raise PlannerError("Run bash scripts/pi-auth.sh before using the pi planner")
        if timeout < 1:
            raise ValueError("pi planning timeout must be positive")

    def command(self, name, request_dir, answer_dir):
        paths = [self.auth_dir, Path(request_dir).resolve(), Path(answer_dir).resolve()]
        if any("," in str(path) or "\n" in str(path) for path in paths):
            raise ValueError("Mount paths cannot contain commas or newlines")
        args = [
            "create", "--pull", "never", "-i", "--name", name,
            "--label", "neural-lab.owner=" + self.owner,
            "--label", "neural-lab.role=planner",
            "--user", f"{os.getuid()}:{os.getgid()}",
            "--read-only", "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges:true",
            "--pids-limit", "128", "--memory", "1g", "--memory-swap", "1g",
            "--cpus", "1", "--init", "--network", "bridge",
            "--tmpfs", "/tmp:rw,nosuid,nodev,size=256m,mode=1777",
            "--log-driver", "local", "--log-opt", "max-size=1m", "--log-opt", "max-file=2",
            "--mount", f"type=bind,src={paths[0]},dst=/pi",
            "--mount", f"type=bind,src={paths[1]},dst=/request,readonly",
            "--mount", f"type=bind,src={paths[2]},dst=/answer",
            "--env", "PI_CODING_AGENT_DIR=/pi", "--env", "HOME=/tmp",
            "--workdir", "/request", "--entrypoint", "pi", self.image,
            "--print", "--no-session", "--no-tools",
            "-ne", "-ns", "-np", "-nc", "--offline",
            "--system-prompt", "/request/system-prompt.txt",
        ]
        if self.model:
            args.extend(["--model", self.model])
        return args + [FINAL_PROMPT]

    def plan(self, history, attempt):
        request = json.dumps({
            "attempt": attempt, "runtime_seconds": self.runtime_seconds, "recent_runs": history,
        })
        system_prompt = EXPERIMENT_INSTRUCTIONS + (
            "\nAct only as an experiment planner. All context is in this prompt; "
            "do not use tools, read local files, edit files, or run code. "
            "Respond with ONLY the JSON plan object: exactly the keys hypothesis "
            "and code, no markdown fences, no commentary.\n"
        )
        name = "neural-lab-plan-" + uuid.uuid4().hex
        process = None
        # The auth directory is mounted only into this planner, never the worker.
        with tempfile.TemporaryDirectory(prefix="neural-lab-plan-") as root:
            request_dir, answer_dir = Path(root) / "request", Path(root) / "answer"
            request_dir.mkdir(mode=0o700)
            answer_dir.mkdir(mode=0o700)
            (request_dir / "schema.json").write_text(json.dumps(PLAN_SCHEMA), encoding="utf-8")
            (request_dir / "request.json").write_text(request, encoding="utf-8")
            (request_dir / "system-prompt.txt").write_text(system_prompt, encoding="utf-8")
            try:
                DockerRunner._docker(*self.command(name, request_dir, answer_dir))
                process = subprocess.Popen(
                    ["docker", "start", "-ai", name],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                )
                deadline = time.monotonic() + self.timeout
                first = True
                while True:
                    if self.stop.is_set():
                        raise PlannerError("pi planning interrupted by shutdown")
                    if time.monotonic() >= deadline:
                        raise PlannerError("pi planning exceeded its timeout")
                    try:
                        stdout, _ = process.communicate(input=b"" if first else None, timeout=1)
                        break
                    except subprocess.TimeoutExpired:
                        first = False
                if process.returncode != 0:
                    raise PlannerError(
                        "pi failed; check the stored API key with scripts/pi-auth.sh status "
                        "and CLI compatibility. No API fallback was attempted."
                    )
                try:
                    return validate_plan(extract_plan(stdout.decode("utf-8", "replace")))
                except (UnicodeDecodeError, ValueError):
                    raise PlannerError("pi returned an unparseable plan; raw output omitted") from None
            except (OSError, subprocess.SubprocessError):
                raise PlannerError("pi container could not run; build its image and check Docker") from None
            finally:
                try:
                    if process is not None and process.poll() is None:
                        process.kill()
                        process.wait(timeout=5)
                finally:
                    try:
                        DockerRunner._docker("rm", "-f", name)
                    except Exception as exc:
                        raise CleanupError("pi planner cleanup failed; halt and recover on restart") from exc
