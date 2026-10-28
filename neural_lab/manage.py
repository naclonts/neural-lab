"""Local readiness checks and read-only experiment status."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import uuid

from neural_lab.codex_planner import CODEX_IMAGE
from neural_lab.runner import DockerRunner


WORKER_IMAGE = "pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime"


class LowDiskSpace(RuntimeError):
    pass


def available_disk(root):
    path = Path(root).resolve()
    while not path.exists():
        path = path.parent
    return shutil.disk_usage(path).free / (1024 ** 3)


def require_disk_space(root, minimum_gb):
    free = available_disk(root)
    if free < minimum_gb:
        raise LowDiskSpace(f"Only {free:.1f} GiB free; minimum is {minimum_gb:g} GiB")
    return free


def read_status(root, *, limit=5):
    """Use SQLite read-only mode, without taking the supervisor lock."""
    root = Path(root).resolve()
    database = root / "journal.sqlite3"
    if not database.exists():
        return {"state_dir": str(root), "initialized": False, "attempts": 0, "latest_runs": []}
    connection = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True, timeout=5)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA query_only=ON")
        connection.execute("BEGIN")  # Consistent snapshot while the supervisor writes.
        attempts, reserved = connection.execute(
            "SELECT COUNT(*), COALESCE(SUM(reserved_seconds),0) FROM runs"
        ).fetchone()
        counts = dict(connection.execute("SELECT status, COUNT(*) FROM runs GROUP BY status").fetchall())
        rows = connection.execute(
            """SELECT id,attempt,status,hypothesis,result,error,created_at,finished_at
               FROM runs ORDER BY attempt DESC LIMIT ?""", (limit,),
        ).fetchall()
        latest = []
        for row in rows:
            item = dict(row)
            item["result"] = json.loads(item["result"]) if item["result"] else None
            item["output_dir"] = str(root / "outputs" / row["id"])
            latest.append(item)
        return {
            "state_dir": str(root), "initialized": True, "attempts": attempts,
            "reserved_training_seconds": reserved, "status_counts": counts, "latest_runs": latest,
        }
    finally:
        connection.close()


def _command(*args):
    return subprocess.run(
        args, check=True, text=True, capture_output=True, timeout=30,
    ).stdout.strip()


def check_readiness(*, state_dir=".neural-lab", image=WORKER_IMAGE, planner="codex",
                    auth_dir=None, gpu="0", minimum_gb=10, gpu_test=False):
    checks = []
    def check(name, action, fix):
        try:
            detail = action()
            checks.append({"check": name, "ok": True, "detail": str(detail)})
            return True
        except Exception:
            # Do not copy command stderr, endpoint URLs, environment values, or
            # login output into diagnostics.
            checks.append({"check": name, "ok": False, "fix": fix})
            return False

    def user():
        if platform.system() != "Linux" or os.getuid() == 0 or sys.version_info < (3, 10):
            raise RuntimeError("Unsupported host")
        return "Linux, non-root user, Python 3.10+"
    host_ok = check("host", user, "Use Python 3.10+ as a regular Linux user, without sudo.")
    check("disk", lambda: f"{require_disk_space(state_dir, minimum_gb):.1f} GiB free",
          f"Free disk space; this run requires at least {minimum_gb:g} GiB.")

    if not host_ok:
        return checks

    def local_daemon():
        endpoint = os.environ.get("DOCKER_HOST")
        if not endpoint or os.environ.get("DOCKER_CONTEXT"):
            endpoint = _command("docker", "context", "inspect", "--format", "{{.Endpoints.docker.Host}}")
        if not endpoint.startswith("unix://"):
            raise RuntimeError("Remote Docker daemon")
        version = _command("docker", "version", "--format", "{{.Server.Version}}")
        return "Local Docker server " + version
    docker_ok = check("docker", local_daemon, "Start a local Docker daemon and check socket access/context.")
    if not docker_ok:
        return checks
    check("worker_image", lambda: _command("docker", "image", "inspect", "--format", "{{.Id}}", image),
          "Pull the configured worker image first, for example: docker compose pull")

    if gpu is not None:
        def nvidia():
            raw = _command("nvidia-smi", "--query-gpu=index,name,memory.total,temperature.gpu",
                           "--format=csv,noheader,nounits")
            for line in raw.splitlines():
                fields = [part.strip() for part in line.split(",")]
                if len(fields) == 4 and fields[0] == str(gpu):
                    temperature = float(fields[3])
                    if not math.isfinite(temperature) or temperature >= 85:
                        raise RuntimeError("GPU already too hot")
                    return f"{fields[1]}, {fields[2]} MiB VRAM, {temperature:g} C"
            raise RuntimeError("GPU not found")
        check("gpu", nvidia, "Check nvidia-smi, the selected GPU index, and GPU temperature (must be below 85 C).")

    if planner == "codex":
        home = Path(auth_dir or os.environ.get("NEURAL_LAB_CODEX_HOME")
                    or Path.home() / ".local/share/neural-lab/codex").resolve()
        image_ok = check("codex_image", lambda: _command(
            "docker", "image", "inspect", "--format", "{{.Id}}", CODEX_IMAGE),
            "Build the Codex image with: bash scripts/codex-login.sh")
        if image_ok:
            def login():
                if not home.is_dir() or "," in str(home) or "\n" in str(home):
                    raise RuntimeError("Missing or invalid dedicated auth directory")
                name = "neural-lab-doctor-" + uuid.uuid4().hex
                try:
                    DockerRunner._docker(
                        "create", "--pull", "never", "--name", name,
                        "--label", "neural-lab.role=doctor",
                        "--user", f"{os.getuid()}:{os.getgid()}",
                        "--read-only", "--cap-drop", "ALL", "--security-opt", "no-new-privileges:true",
                        "--memory", "1g", "--memory-swap", "1g", "--cpus", "1", "--pids-limit", "128",
                        "--tmpfs", "/tmp:rw,nosuid,nodev,size=256m,mode=1777",
                        "--mount", f"type=bind,src={home},dst=/codex",
                        CODEX_IMAGE, "-c", 'forced_login_method="chatgpt"',
                        "-c", 'cli_auth_credentials_store="file"',
                        "-c", 'check_for_update_on_startup=false', "login", "status",
                    )
                    DockerRunner._docker("start", "-a", name)
                    # docker start -a does not reliably propagate the container's
                    # exit code, so inspect it explicitly.
                    state = DockerRunner._docker("inspect", "--format", "{{.State.Running}} {{.State.ExitCode}}", name)
                    if state != "false 0":
                        raise RuntimeError("ChatGPT login unavailable")
                    return "Cached ChatGPT login is available; account quota is not checked"
                finally:
                    DockerRunner._docker("rm", "-f", name)
            check("codex_login", login, "Sign in with ChatGPT: bash scripts/codex-login.sh")

    if gpu_test and gpu is not None and all(item["ok"] for item in checks):
        def cuda():
            with tempfile.TemporaryDirectory(prefix="neural-lab-gpu-check-") as root:
                inputs, outputs = Path(root) / "input", Path(root) / "output"
                inputs.mkdir()
                outputs.mkdir()
                source = (
                    "import json, torch\n"
                    "from pathlib import Path\n"
                    "assert torch.cuda.is_available(), 'CUDA unavailable'\n"
                    "x = torch.randn(128, 128, device='cuda:0')\n"
                    "assert torch.isfinite(x @ x).all().item()\n"
                    "torch.cuda.synchronize()\n"
                    "Path('/output/result.json').write_text(json.dumps({"
                    "'metrics': {'cuda_ok': 1}, 'summary': 'Trusted CUDA matrix test passed'}))\n"
                )
                (inputs / "train.py").write_text(source)
                runner = DockerRunner("doctor-" + uuid.uuid4().hex, image=image, gpu=gpu, timeout=60)
                result = runner.run(uuid.uuid4().hex, inputs, outputs, threading.Event())
                if result["status"] != "succeeded":
                    raise RuntimeError("CUDA test failed")
                return "Trusted CUDA matrix test passed inside the worker sandbox"
        check("cuda_sandbox", cuda, "Check NVIDIA Container Toolkit and driver compatibility with the worker image.")
    return checks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    status = commands.add_parser("status", help="Inspect existing run records without modifying them")
    status.add_argument("--state-dir", type=Path, default=Path(".neural-lab"))
    status.add_argument("--limit", type=int, default=5)
    doctor = commands.add_parser("doctor", help="Check local readiness without consuming planner attempts")
    doctor.add_argument("--state-dir", type=Path, default=Path(".neural-lab"))
    doctor.add_argument("--planner", choices=("codex", "demo", "api"), default="codex")
    doctor.add_argument("--codex-home", type=Path)
    doctor.add_argument("--image", default=WORKER_IMAGE)
    doctor.add_argument("--gpu", default="0")
    doctor.add_argument("--cpu", action="store_true")
    doctor.add_argument("--gpu-test", action="store_true")
    doctor.add_argument("--min-free-gb", type=float, default=10)
    args = parser.parse_args(argv)
    if args.command == "status":
        if not 1 <= args.limit <= 50:
            parser.error("--limit must be between 1 and 50")
        print(json.dumps(read_status(args.state_dir, limit=args.limit), indent=2))
        return 0
    if not math.isfinite(args.min_free_gb) or args.min_free_gb < 1:
        parser.error("--min-free-gb must be finite and at least 1")
    if args.cpu and args.gpu_test:
        parser.error("--gpu-test cannot be combined with --cpu")
    checks = check_readiness(
        state_dir=args.state_dir, image=args.image, planner=args.planner,
        auth_dir=args.codex_home, gpu=None if args.cpu else args.gpu,
        minimum_gb=args.min_free_gb, gpu_test=args.gpu_test,
    )
    print(json.dumps({"ready": all(item["ok"] for item in checks), "checks": checks}, indent=2))
    return 0 if all(item["ok"] for item in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
