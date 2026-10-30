#!/usr/bin/env bash
# Host-side lifecycle for the independent worker and Codex orchestrator.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
compose=(docker compose -f compose.research.yaml)
operation="${1:-status}"
case "$operation" in
  start)
    bash scripts/setup.sh
    # Resolve Compose's .env without executing it as shell code.
    export NEURAL_LAB_GPU_BACKEND
    NEURAL_LAB_GPU_BACKEND="$("${compose[@]}" config --format json | python3 -c 'import json,sys; print(json.load(sys.stdin)["services"]["worker"]["environment"]["NEURAL_LAB_GPU_BACKEND"])')"
    # Pull before enqueue/start: a missing training image must not consume jobs.
    docker pull pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
    python3 -m neural_lab.manage doctor --planner demo --gpu-test
    bash scripts/enqueue-starter-jobs.sh
    bash scripts/enqueue-candidate-jobs.sh
    "${compose[@]}" up -d --build worker orchestrator
    "${compose[@]}" ps
    ;;
  status)
    python3 -m neural_lab.queue status
    "${compose[@]}" ps
    "${compose[@]}" logs --tail=15 orchestrator worker
    ;;
  logs) "${compose[@]}" logs --follow --tail=100 orchestrator worker ;;
  stop) "${compose[@]}" down ;;
  pause-research) "${compose[@]}" stop orchestrator ;;
  resume-research) "${compose[@]}" up -d orchestrator ;;
  *)
    echo 'Usage: bash scripts/research.sh [start|status|logs|stop|pause-research|resume-research]' >&2
    exit 2
    ;;
esac
