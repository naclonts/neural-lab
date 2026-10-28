"""ChatGPT-authenticated Codex CLI planner, isolated from GPU workers."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import time
import uuid

from neural_lab.planner import EXPERIMENT_INSTRUCTIONS, PlannerError, validate_plan
from neural_lab.runner import CleanupError, DockerRunner, read_json_file


CODEX_IMAGE = "neural-lab-codex:0.165.0"
DISABLED_FEATURES = (
    "shell_tool", "view_image", "apps", "plugins", "multi_agent",
    "code_mode", "js_repl", "browser_use", "computer_use", "memories",
    "memory_tool", "skill_search", "tool_suggest",
)
PLAN_SCHEMA = {
    "type": "object",
    "properties": {"hypothesis": {"type": "string"}, "code": {"type": "string"}},
    "required": ["hypothesis", "code"],
    "additionalProperties": False,
}


class CodexPlanner:
    def __init__(self, owner, *, auth_dir=None, model=None, timeout=600,
                 runtime_seconds=300, stop=None, image=CODEX_IMAGE):
        self.owner, self.model, self.timeout = owner, model, timeout
        self.runtime_seconds, self.image = runtime_seconds, image
        self.stop = stop if stop is not None else threading.Event()
        self.auth_dir = Path(auth_dir or os.environ.get("NEURAL_LAB_CODEX_HOME") or Path.home() / ".local/share/neural-lab/codex").resolve()
        if not self.auth_dir.is_dir():
            raise PlannerError("Run bash scripts/codex-login.sh before using the Codex planner")
        if timeout < 1:
            raise ValueError("Codex planning timeout must be positive")

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
            "--mount", f"type=bind,src={paths[0]},dst=/codex",
            "--mount", f"type=bind,src={paths[1]},dst=/request,readonly",
            "--mount", f"type=bind,src={paths[2]},dst=/answer",
            "--env", "CODEX_HOME=/codex", "--env", "HOME=/tmp",
            "--workdir", "/request", "--entrypoint", "codex", self.image,
            "exec", "--ignore-user-config", "--ephemeral", "--skip-git-repo-check",
            "--sandbox", "read-only",
            "--output-schema", "/request/schema.json",
            "--output-last-message", "/answer/plan.json",
            "-c", 'forced_login_method="chatgpt"',
            "-c", 'cli_auth_credentials_store="file"',
            "-c", 'check_for_update_on_startup=false',
            "-c", 'approval_policy="never"',
            "-c", 'web_search="disabled"',
        ]
        for feature in DISABLED_FEATURES:
            args.extend(["--disable", feature])
        if self.model:
            args.extend(["--model", self.model])
        return args + ["-"]

    def plan(self, history, attempt):
        prompt = EXPERIMENT_INSTRUCTIONS + (
            "\nAct only as an experiment planner. All context is in this prompt; "
            "do not use tools, read local files, edit files, or run code. "
            "Return the hypothesis and code as the schema requires.\n"
        ) + json.dumps({
            "attempt": attempt, "runtime_seconds": self.runtime_seconds, "recent_runs": history,
        })
        name = "neural-lab-plan-" + uuid.uuid4().hex
        process = None
        # The auth directory is mounted only into this planner, never the worker.
        with tempfile.TemporaryDirectory(prefix="neural-lab-plan-") as root:
            request_dir, answer_dir = Path(root) / "request", Path(root) / "answer"
            request_dir.mkdir(mode=0o700)
            answer_dir.mkdir(mode=0o700)
            (request_dir / "schema.json").write_text(json.dumps(PLAN_SCHEMA), encoding="utf-8")
            try:
                DockerRunner._docker(*self.command(name, request_dir, answer_dir))
                process = subprocess.Popen(
                    ["docker", "start", "-ai", name],
                    stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                deadline = time.monotonic() + self.timeout
                first = True
                while True:
                    if self.stop.is_set():
                        raise PlannerError("Codex planning interrupted by shutdown")
                    if time.monotonic() >= deadline:
                        raise PlannerError("Codex planning exceeded its timeout")
                    try:
                        process.communicate(input=prompt.encode() if first else None, timeout=1)
                        break
                    except subprocess.TimeoutExpired:
                        first = False
                if process.returncode != 0:
                    raise PlannerError(
                        "Codex failed; check ChatGPT login, subscription allowance, and CLI compatibility. "
                        "No API fallback was attempted."
                    )
                return validate_plan(read_json_file(answer_dir, "plan.json", limit=131072))
            except (OSError, subprocess.SubprocessError):
                raise PlannerError("Codex container could not run; build its image and check Docker") from None
            finally:
                try:
                    if process is not None and process.poll() is None:
                        process.kill()
                        process.wait(timeout=5)
                finally:
                    try:
                        DockerRunner._docker("rm", "-f", name)
                    except Exception as exc:
                        raise CleanupError("Codex planner cleanup failed; halt and recover on restart") from exc
