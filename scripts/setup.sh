#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
if [ "$(id -u)" -eq 0 ]; then
  echo "Run setup as your regular Linux user, without sudo." >&2
  exit 1
fi
command -v docker >/dev/null || { echo "Install Docker and Docker Compose first." >&2; exit 1; }
if [ ! -S /var/run/docker.sock ]; then
  echo "Docker socket /var/run/docker.sock is unavailable; start Docker first." >&2
  exit 1
fi
umask 077
mkdir -p experiments
if [ ! -e .env ]; then
  printf 'LAB_UID=%s\nLAB_GID=%s\nLAB_REPO_DIR=%s\nDOCKER_GID=%s\nNEURAL_LAB_CODEX_HOME=%s\n' \
    "$(id -u)" "$(id -g)" "$PWD" "$(stat -c %g /var/run/docker.sock)" \
    "${NEURAL_LAB_CODEX_HOME:-$HOME/.local/share/neural-lab/codex}" > .env
else
  echo "Keeping existing .env; check LAB_UID and LAB_GID match your regular user."
  if ! grep -q '^LAB_REPO_DIR=' .env; then printf 'LAB_REPO_DIR=%s\n' "$PWD" >> .env; fi
  if ! grep -q '^DOCKER_GID=' .env; then printf 'DOCKER_GID=%s\n' "$(stat -c %g /var/run/docker.sock)" >> .env; fi
  if ! grep -q '^NEURAL_LAB_CODEX_HOME=' .env; then
    printf 'NEURAL_LAB_CODEX_HOME=%s\n' "${NEURAL_LAB_CODEX_HOME:-$HOME/.local/share/neural-lab/codex}" >> .env
  fi
fi
mkdir -p .neural-lab/notes
# Let Compose resolve .env itself; never execute shell code from that file.
auth_dir="$(docker compose -f compose.research.yaml config --format json | python3 -c 'import json,sys; print(next(v["source"] for v in json.load(sys.stdin)["services"]["orchestrator"]["volumes"] if v["target"] == "/codex"))')"
mkdir -p -- "$auth_dir"
chmod 700 -- "$auth_dir"
docker compose config --quiet
docker compose -f compose.research.yaml config --quiet
echo "Configuration is valid. Start research with: bash scripts/research.sh start"
