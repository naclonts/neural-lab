#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
compose=(docker compose -f compose.dashboard.yaml)
case "${1:-status}" in
  start) "${compose[@]}" up -d --build; echo 'Dashboard: http://127.0.0.1:8766' ;;
  stop) "${compose[@]}" down ;;
  status) "${compose[@]}" ps ;;
  logs) "${compose[@]}" logs --tail=50 --follow ;;
  *) echo 'Usage: bash scripts/dashboard.sh [start|status|logs|stop]' >&2; exit 2 ;;
esac
