"""Read-only observatory. No credentials, Docker socket, or experiment execution."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import stat
import shutil
import subprocess
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit, urlencode

WEB = Path(__file__).with_name("web")
NOTE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\.md\Z")


def open_job_file(state: Path, job: str, path: str, *, source=False):
    """Open a regular file without following any output directory symlinks."""
    if not re.fullmatch(r"[0-9a-f]{32}", job):
        raise ValueError("Invalid job")
    parts = path.split("/")
    if len(parts) > 5 or any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", part) or part in (".", "..") for part in parts):
        raise ValueError("Invalid artifact path")
    directory = os.open(state, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in ("queue-inputs" if source else "queue-outputs", job, *parts[:-1]):
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
    finally:
        os.close(directory)
    stream = os.fdopen(fd, "rb")
    info = os.fstat(stream.fileno())
    if not stat.S_ISREG(info.st_mode) or info.st_size > 512 * 1024**2:
        stream.close()
        raise ValueError("Artifact must be a regular file smaller than 512 MiB")
    return stream, info.st_size


def artifact_url(job, path):
    return "/api/artifact?" + urlencode({"job": job, "path": path})


def read_artifacts(state: Path, job: str) -> dict:
    result = {"datasets": [], "checkpoints": [], "reports": []}
    manifest = {}
    try:
        with open_job_file(state, job, "artifacts.json")[0] as stream:
            raw = stream.read(1048577)
            if len(raw) <= 1048576:
                manifest = json.loads(raw)
        if not isinstance(manifest, dict):
            manifest = {}
    except (OSError, ValueError):
        pass
    for category in ("datasets", "checkpoints"):
        entries = manifest.get(category) or []
        if not isinstance(entries, list):
            continue
        for entry in entries[:100]:
            if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
                continue
            path = entry["path"]
            if not path.startswith(category + "/"):
                continue
            try:
                stream, size = open_job_file(state, job, path)
                stream.close()
            except (OSError, ValueError):
                continue
            item = {key: entry.get(key) for key in ("label", "path", "sha256", "selection", "source", "format")}
            item.update(url=artifact_url(job, path), bytes=size)
            result[category].append(item)
    for path, label, category in (("result.json", "Result metrics", "reports"),
                                  ("details.json", "Protocol & per-seed details", "reports"),
                                  ("artifacts.json", "Artifact provenance manifest", "reports"),
                                  ("config.json", "Model configuration", "reports"),
                                  ("model-state.json", "Legacy final candidate weights (not best-selected)", "checkpoints")):
        try:
            stream, size = open_job_file(state, job, path)
            stream.close()
            result[category].append({"label": label, "path": path, "url": artifact_url(job, path), "bytes": size})
        except (OSError, ValueError):
            pass
    try:
        stream, size = open_job_file(state, job, "train.py", source=True)
        stream.close()
        result["reports"].append({"label": "Exact queued training source", "path": "train.py", "url": "/api/source?" + urlencode({"job": job}), "bytes": size})
    except (OSError, ValueError):
        pass
    return result


def read_file(path: Path, limit: int = 65536) -> str:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise ValueError("Not a bounded regular file")
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("File exceeds limit")
    return raw.decode("utf-8")


def read_checkpoint(state: Path, name: str) -> dict:
    try:
        value = json.loads(read_file(state / "notes" / name))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def read_queue(state: Path) -> dict:
    database = state / "queue.sqlite3"
    if not database.exists():
        return {"initialized": False, "counts": {}, "jobs": []}
    # Do not use Queue(): its constructor creates directories/tables and writes WAL.
    db = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True, timeout=2)
    db.row_factory = sqlite3.Row
    try:
        db.execute("PRAGMA query_only=ON")
        db.execute("BEGIN")
        counts = dict(db.execute("SELECT status, COUNT(*) FROM jobs GROUP BY status"))
        rows = db.execute("""SELECT id,hypothesis,status,created_at,started_at,finished_at,
            timeout_seconds,result,error FROM jobs ORDER BY created_at DESC,id DESC LIMIT 100""")
        jobs = []
        for row in rows:
            item = dict(row)
            try:
                result = json.loads(item.pop("result") or "null")
            except ValueError:
                result = None
            item["result"] = result if isinstance(result, dict) else None
            item["device"] = None
            item["artifacts"] = read_artifacts(state, item["id"])
            if re.fullmatch(r"[0-9a-f]{32}", item["id"]):
                try:
                    detail = json.loads(read_file(state / "queue-outputs" / item["id"] / "details.json", 1048576))
                    environment = detail.get("environment") or {}
                    device = detail.get("device") or environment.get("device")
                    item["device"] = device if isinstance(device, str) else None
                except (OSError, ValueError, AttributeError):
                    pass
            jobs.append(item)
        return {"initialized": True, "counts": counts, "jobs": jobs}
    finally:
        db.close()


def read_gpu() -> dict:
    try:
        result = subprocess.run([
            "nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ], capture_output=True, text=True, timeout=3, check=True)
        devices = []
        for line in result.stdout.splitlines()[:8]:
            fields = [v.strip() for v in line.split(",")]
            if len(fields) != 7:
                continue
            device = {"index": fields[0], "name": fields[1]}
            for key, value in zip(("utilization", "memory_used", "memory_total", "temperature", "power"), fields[2:]):
                try:
                    number = float(value)
                    device[key] = number if math.isfinite(number) else None
                except ValueError:
                    device[key] = None
            devices.append(device)
        return {"available": bool(devices), "devices": devices}
    except (OSError, subprocess.SubprocessError):
        return {"available": False, "devices": []}


def list_notes(state: Path) -> list[dict]:
    notes = []
    try:
        for path in (state / "notes").iterdir():
            if not NOTE_NAME.fullmatch(path.name):
                continue
            info = path.lstat()
            if stat.S_ISREG(info.st_mode) and info.st_size <= 65536:
                notes.append({"name": path.name, "updated_at": info.st_mtime, "bytes": info.st_size})
    except OSError:
        pass
    return sorted(notes, key=lambda item: item["updated_at"], reverse=True)[:20]


def snapshot(state: Path) -> dict:
    errors = []
    try:
        queue = read_queue(state)
    except (OSError, sqlite3.Error):
        queue = {"initialized": False, "counts": {}, "jobs": []}
        errors.append("Experiment queue is temporarily unavailable.")
    session = read_checkpoint(state, "latest-session.json")
    pause = read_checkpoint(state, "pause-state.json")
    # A historical pause must not override a newer active/completed session.
    if isinstance(pause.get("updated_at"), (int, float)) and pause["updated_at"] > (session.get("updated_at") or 0):
        session = {**pause, "reason": "paused: " + str(pause.get("reason", "unknown"))}
    # Only the known observation fields are served, never arbitrary checkpoint data.
    scientist = {key: session.get(key) for key in (
        "model", "reason", "last_tokens", "model_context_window", "updated_at", "recent_agent_messages",
        "context_handoff_fraction", "context_handoff_tokens",
    )}
    usage = os.statvfs(state if state.exists() else state.parent)
    return {"generated_at": time.time(), "queue": queue, "scientist": scientist,
            "gpu": read_gpu(), "notes": list_notes(state), "errors": errors,
            "disk_free_gb": round(usage.f_bavail * usage.f_frsize / 1024**3, 1)}


def handler_for(state: Path, allowed_hosts=()):
    state = state.resolve()
    hosts = {"localhost", "127.0.0.1", *allowed_hosts}

    class Handler(BaseHTTPRequestHandler):
        def respond(self, code, body: bytes, content_type="application/json; charset=utf-8"):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            # Explicit hosts also make accidental DNS rebinding fail closed.
            hostname = self.headers.get("Host", "").split(":")[0].lower()
            if hostname not in hosts:
                self.respond(403, b'{"error":"Host is not allowed"}')
                return
            url = urlsplit(self.path)
            assets = {"/": ("index.html", "text/html"), "/dashboard.js": ("dashboard.js", "text/javascript"),
                      "/dashboard.css": ("dashboard.css", "text/css")}
            try:
                if url.path in assets:
                    filename, mime = assets[url.path]
                    self.respond(200, (WEB / filename).read_bytes(), mime + "; charset=utf-8")
                elif url.path == "/api/status":
                    self.respond(200, json.dumps(snapshot(state), allow_nan=False).encode())
                elif url.path == "/api/note":
                    name = parse_qs(url.query).get("name", [""])[0]
                    if not NOTE_NAME.fullmatch(name):
                        self.respond(400, b'{"error":"Invalid note name"}')
                        return
                    body = read_file(state / "notes" / name)
                    self.respond(200, json.dumps({"name": name, "text": body}).encode())
                elif url.path in ("/api/artifact", "/api/source"):
                    query = parse_qs(url.query)
                    job = query.get("job", [""])[0]
                    path = query.get("path", [""])[0]
                    source = url.path == "/api/source"
                    if source:
                        path = "train.py"
                    else:
                        index = read_artifacts(state, job)
                        allowed = {entry["path"] for entries in index.values() for entry in entries}
                        if path not in allowed:
                            self.respond(404, b'{"error":"Artifact not available"}')
                            return
                    with open_job_file(state, job, path, source=source)[0] as stream:
                        size = os.fstat(stream.fileno()).st_size
                        self.send_response(200)
                        self.send_header("Content-Type", "application/octet-stream")
                        self.send_header("Content-Disposition", f'attachment; filename="{path.split("/")[-1]}"')
                        self.send_header("Content-Length", str(size))
                        self.send_header("Cache-Control", "no-store")
                        self.send_header("X-Content-Type-Options", "nosniff")
                        self.end_headers()
                        shutil.copyfileobj(stream, self.wfile, length=65536)
                else:
                    self.respond(404, b'{"error":"Not found"}')
            except (OSError, ValueError):
                self.respond(404, b'{"error":"Observation unavailable"}')

        def log_message(self, *_):
            pass  # Keep note names and requests out of logs.

    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, default=Path(".neural-lab"))
    parser.add_argument("--host", choices=("127.0.0.1", "0.0.0.0"), default="127.0.0.1",
                        help="Use 0.0.0.0 only inside the loopback-published container")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--allowed-host", action="append", default=[], help="Exact Tailscale HTTPS hostname")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    server = ThreadingHTTPServer((args.host, args.port), handler_for(args.state_dir, args.allowed_host))
    server.daemon_threads = True
    print(f"Neural Lab observatory: http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
