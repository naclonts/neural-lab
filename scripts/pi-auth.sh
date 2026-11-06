#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
if [ "$(id -u)" -eq 0 ]; then
  echo "Run as your regular Linux user, without sudo." >&2
  exit 1
fi
operation="${1:-status}"
case "$operation" in
  status|set-key|clear|build) ;;
  *) echo "Usage: bash scripts/pi-auth.sh [status|set-key|clear|build]" >&2; exit 2 ;;
esac
umask 077
# Read KEY=VALUE pairs from .env (last match wins, surrounding quotes stripped)
# without sourcing it, so a stray line cannot execute arbitrary shell code.
env_get() {
  [ -f .env ] || return 0
  grep -E "^$1=" .env | tail -n1 | cut -d= -f2- | sed -e "s/^['\"]//" -e "s/['\"]\$//"
}
auth_dir="${NEURAL_LAB_PI_HOME:-$HOME/.local/share/neural-lab/pi}"
# Resolve the model/providers from the environment, falling back to .env; the
# status check must probe the exact model the orchestrator will run.
model="${NEURAL_LAB_ORCH_MODEL:-$(env_get NEURAL_LAB_ORCH_MODEL)}"
model="${model:-anthropic/claude-sonnet-4.5}"
# Provider: PI_PROVIDER wins; otherwise derive it from the model prefix
# ("fireworks/accounts/..." -> "fireworks"), else default to anthropic.
provider="${PI_PROVIDER:-}"
if [ -z "$provider" ]; then
  case "$model" in */*) provider="${model%%/*}" ;; *) provider="anthropic" ;; esac
fi
if [ -f .env ]; then
  compose_dir="$(docker compose -f compose.research.yaml config --format json | python3 -c 'import json,sys; print(next(v["source"] for v in json.load(sys.stdin)["services"]["pi-orchestrator"]["volumes"] if v["target"] == "/pi"))')"
  if [ -n "${NEURAL_LAB_PI_HOME:-}" ]; then
    # An explicitly set NEURAL_LAB_PI_HOME wins over the Compose-resolved path;
    # warn on divergence so the CLI and the containers never differ silently.
    if [ "$compose_dir" != "$auth_dir" ]; then
      echo "Warning: NEURAL_LAB_PI_HOME ($auth_dir) differs from the Compose-resolved path ($compose_dir); using NEURAL_LAB_PI_HOME." >&2
    fi
  else
    auth_dir="$compose_dir"
  fi
fi
mkdir -p -- "$auth_dir"
auth_dir="$(cd -- "$auth_dir" && pwd -P)"
chmod 700 -- "$auth_dir"
case "$auth_dir" in
  *","*) echo "Auth directory cannot contain commas." >&2; exit 2 ;;
esac
# Only set-key (and an explicit build) rebuild the image; status and clear work
# with the image that is already present.
if [ "$operation" = "build" ] || [ "$operation" = "set-key" ]; then
  docker build -t neural-lab-pi:1.5.0 -f docker/pi.Dockerfile .
fi
case "$operation" in
  build)
    echo "Built neural-lab-pi:1.5.0."
    ;;
  # The key arrives on STDIN, never as an argument, so it stays out of the
  # process list and shell history.
  set-key)
    key="$(cat)"
    PI_AUTH_KEY="$key" PI_AUTH_PROVIDER="$provider" python3 -c '
import json, os, sys
provider, key = os.environ["PI_AUTH_PROVIDER"], os.environ["PI_AUTH_KEY"].strip()
if not key:
    sys.exit("Provide the API key on STDIN, for example: pbpaste | bash scripts/pi-auth.sh set-key")
fd = os.open(sys.argv[1], os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
with os.fdopen(fd, "w") as stream:
    json.dump({provider: {"type": "api_key", "key": key}}, stream)
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
      --print --no-session --offline -ne -ns -np -nc --model "$model" ok >/dev/null
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
    echo "pi API-key auth is available (provider $provider, model $model)."
    ;;
esac
