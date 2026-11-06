#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
if [ "$(id -u)" -eq 0 ]; then
  echo "Run as your regular Linux user, without sudo." >&2
  exit 1
fi
operation="${1:-status}"
case "$operation" in
  status|set-key|clear) ;;
  *) echo "Usage: bash scripts/pi-auth.sh [status|set-key|clear]" >&2; exit 2 ;;
esac
umask 077
auth_dir="${NEURAL_LAB_PI_HOME:-$HOME/.local/share/neural-lab/pi}"
if [ -f .env ]; then
  auth_dir="$(docker compose -f compose.research.yaml config --format json | python3 -c 'import json,sys; print(next(v["source"] for v in json.load(sys.stdin)["services"]["pi-orchestrator"]["volumes"] if v["target"] == "/pi"))')"
fi
mkdir -p -- "$auth_dir"
auth_dir="$(cd -- "$auth_dir" && pwd -P)"
chmod 700 -- "$auth_dir"
case "$auth_dir" in
  *","*) echo "Auth directory cannot contain commas." >&2; exit 2 ;;
esac
docker build -t neural-lab-pi:1.5.0 -f docker/pi.Dockerfile .
case "$operation" in
  # The key arrives on STDIN, never as an argument, so it stays out of the
  # process list and shell history.
  set-key)
    key="$(cat)"
    provider="${PI_PROVIDER:-anthropic}"
    PI_AUTH_KEY="$key" PI_AUTH_PROVIDER="$provider" python3 -c '
import json, os, sys
provider, key = os.environ["PI_AUTH_PROVIDER"], os.environ["PI_AUTH_KEY"].strip()
if not key:
    sys.exit("Provide the API key on STDIN, for example: pbpaste | bash scripts/pi-auth.sh set-key")
fd = os.open(sys.argv[1], os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
with os.fdopen(fd, "w") as stream:
    json.dump({provider: {"type": "api", "key": key}}, stream)
    stream.write("\n")
' "$auth_dir/auth.json"
    chmod 600 -- "$auth_dir/auth.json"
    echo "Stored ${provider} API key in $auth_dir/auth.json"
    ;;
  clear)
    rm -f -- "$auth_dir/auth.json"
    echo "Removed $auth_dir/auth.json"
    ;;
  status)
    name="neural-lab-pi-auth-$$"
    docker create --pull never --name "$name" \
      --label "neural-lab.role=doctor" \
      --user "$(id -u):$(id -g)" \
      --read-only --cap-drop ALL --security-opt no-new-privileges:true \
      --pids-limit 128 --memory 1g --memory-swap 1g --cpus 1 --init \
      --tmpfs /tmp:rw,nosuid,nodev,size=256m,mode=1777 \
      --mount "type=bind,src=$auth_dir,dst=/pi" \
      --env PI_CODING_AGENT_DIR=/pi \
      neural-lab-pi:1.5.0 \
      --print --no-session --offline -ne -ns -np -nc ok >/dev/null
    started=0
    docker start -a "$name" || started=$?
    # docker start -a does not reliably propagate the container's exit code,
    # so inspect it explicitly.
    state="$(docker inspect --format '{{.State.Running}} {{.State.ExitCode}}' "$name")"
    docker rm -f "$name" >/dev/null
    if [ "$started" -ne 0 ] || [ "$state" != "false 0" ]; then
      echo "pi auth check failed; run: printf 'YOUR_KEY' | bash scripts/pi-auth.sh set-key" >&2
      exit 1
    fi
    echo "pi API-key auth is available (provider ${PI_PROVIDER:-anthropic})."
    ;;
esac
