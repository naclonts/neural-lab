#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
if [ "$(id -u)" -eq 0 ]; then
  echo "Run as your regular Linux user, without sudo." >&2
  exit 1
fi
operation="${1:-login}"
case "$operation" in
  login) login_args=(login --device-auth) ;;
  status) login_args=(login status) ;;
  logout) login_args=(logout) ;;
  *) echo "Usage: bash scripts/codex-login.sh [login|status|logout]" >&2; exit 2 ;;
esac
umask 077
auth_dir="${NEURAL_LAB_CODEX_HOME:-$HOME/.local/share/neural-lab/codex}"
mkdir -p -- "$auth_dir"
auth_dir="$(cd -- "$auth_dir" && pwd -P)"
chmod 700 -- "$auth_dir"
case "$auth_dir" in
  *","*) echo "Login directory cannot contain commas." >&2; exit 2 ;;
esac
docker build -t neural-lab-codex:0.165.0 -f docker/codex.Dockerfile docker
docker run --rm -i \
  --user "$(id -u):$(id -g)" \
  --read-only --cap-drop ALL --security-opt no-new-privileges:true \
  --pids-limit 128 --memory 1g --memory-swap 1g --cpus 1 --init \
  --tmpfs /tmp:rw,nosuid,nodev,size=256m,mode=1777 \
  --mount "type=bind,src=$auth_dir,dst=/codex" \
  neural-lab-codex:0.165.0 \
  -c 'forced_login_method="chatgpt"' \
  -c 'cli_auth_credentials_store="file"' \
  "${login_args[@]}"
