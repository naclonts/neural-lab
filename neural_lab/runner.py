"""Trusted Docker launcher. No generated Python is imported by the host."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import stat
import subprocess
import time


class CleanupError(RuntimeError):
    """Container removal failed; halt rather than overlap experiments."""


MAX_RESULT_BYTES = 65536


def read_result(directory):
    """Read only a bounded regular file; never follow a worker-created symlink."""
    fd = os.open(Path(directory) / "result.json", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        metadata = os.fstat(stream.fileno())
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > MAX_RESULT_BYTES:
            raise ValueError("Result must be a regular file under 64 KiB")
        raw = stream.read(MAX_RESULT_BYTES + 1)
    if len(raw) > MAX_RESULT_BYTES:
        raise ValueError("Result exceeds 64 KiB")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("Result must be an object")
    metrics = value.get("metrics")
    summary = value.get("summary")
    if not isinstance(metrics, dict) or not 1 <= len(metrics) <= 32:
        raise ValueError("Expected 1-32 metrics")
    for key, number in metrics.items():
        if not isinstance(key, str) or not 1 <= len(key) <= 80:
            raise ValueError("Invalid metric name")
        if type(number) not in (int, float) or not math.isfinite(number):
            raise ValueError("Metrics must be finite numbers")
    if not isinstance(summary, str) or not 1 <= len(summary) <= 2000:
        raise ValueError("Expected a summary of at most 2000 characters")
    return {"metrics": metrics, "summary": summary}


class DockerRunner:
    def __init__(self, owner, *, image="pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime",
                 gpu="0", timeout=300, toolkit=None):
        self.owner, self.image, self.gpu, self.timeout = owner, image, gpu, timeout
        self.toolkit = Path(toolkit or Path(__file__).parent).resolve()
        if gpu is not None and not str(gpu).isdigit():
            raise ValueError("GPU must be a numeric device index")
        for path in (self.toolkit,):
            if "," in str(path) or "\n" in str(path):
                raise ValueError("Mount paths cannot contain commas or newlines")

    @staticmethod
    def _docker(*args):
        return subprocess.run(
            ["docker", *args], check=True, text=True, capture_output=True, timeout=30,
        ).stdout.strip()

    def recover(self):
        ids = self._docker("ps", "-aq", "--filter", "label=neural-lab.owner=" + self.owner).split()
        if ids:
            self._docker("rm", "-f", *ids)

    def command(self, name, input_dir, output_dir):
        paths = [Path(input_dir).resolve(), Path(output_dir).resolve(), self.toolkit]
        if any("," in str(path) or "\n" in str(path) for path in paths):
            raise ValueError("Mount paths cannot contain commas or newlines")
        args = [
            "create", "--pull", "never", "--name", name, "--label", "neural-lab.owner=" + self.owner,
            "--user", f"{os.getuid()}:{os.getgid()}",
            "--read-only", "--cap-drop", "ALL", "--security-opt", "no-new-privileges:true",
            "--pids-limit", "256", "--memory", "8g", "--memory-swap", "8g",
            "--cpus", "4", "--shm-size", "1g", "--init", "--no-healthcheck",
            "--network", "bridge", "--tmpfs", "/tmp:rw,nosuid,nodev,size=1g,mode=1777",
            "--log-driver", "local", "--log-opt", "max-size=1m", "--log-opt", "max-file=2",
            "--mount", f"type=bind,src={paths[0]},dst=/input,readonly",
            "--mount", f"type=bind,src={paths[1]},dst=/output",
            "--mount", f"type=bind,src={paths[2]},dst=/opt/tooling/neural_lab,readonly",
            "--workdir", "/output",
            "--env", "HOME=/tmp", "--env", "PYTHONPATH=/opt/tooling",
            "--env", "PYTHONDONTWRITEBYTECODE=1", "--env", "OMP_NUM_THREADS=4",
            "--env", "HF_HOME=/tmp/huggingface",
            "--env", "HF_HUB_DISABLE_IMPLICIT_TOKEN=1",
        ]
        if self.gpu is not None:
            args.extend(["--gpus", "device=" + str(self.gpu)])
        return args + ["--entrypoint", "python", self.image, "-B", "/input/train.py"]

    def run(self, run_id, input_dir, output_dir, stop):
        name = "neural-lab-" + run_id
        try:
            self._docker(*self.command(name, input_dir, output_dir))
            if stop.is_set():
                return {"status": "interrupted", "error": "Shutdown requested"}
            # Deadline begins before start, including Docker start overhead.
            deadline = time.monotonic() + self.timeout
            self._docker("start", name)
            while True:
                if stop.is_set():
                    return {"status": "interrupted", "error": "Shutdown requested"}
                if time.monotonic() >= deadline:
                    return {"status": "failed", "error": "Experiment exceeded its runtime limit"}
                state = self._docker("inspect", "--format", "{{.State.Running}} {{.State.ExitCode}}", name)
                running, exit_code = state.split()
                if running == "false":
                    if exit_code != "0":
                        return {"status": "failed", "error": "Worker exited with code " + exit_code}
                    return {"status": "succeeded", "result": read_result(output_dir)}
                stop.wait(min(1.0, max(0.0, deadline - time.monotonic())))
        finally:
            # A cleanup failure propagates: do not launch another GPU job if one
            # might still be alive. Startup recovery repeats label-scoped cleanup.
            try:
                self._docker("rm", "-f", name)
            except Exception as exc:
                raise CleanupError("Container cleanup failed; restart to recover") from exc
