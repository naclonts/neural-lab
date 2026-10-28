#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
if [ "$(id -u)" -eq 0 ]; then
  echo "Run setup as your regular Linux user, without sudo." >&2
  exit 1
fi
umask 077
mkdir -p experiments
if [ ! -e .env ]; then
  printf 'LAB_UID=%s\nLAB_GID=%s\n' "$(id -u)" "$(id -g)" > .env
else
  echo "Keeping existing .env; check LAB_UID and LAB_GID match your regular user."
fi
docker compose config --quiet
echo "Configuration is valid. Start with: docker compose up -d"
