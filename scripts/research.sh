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
    training_image="$("${compose[@]}" config --format json | python3 -c 'import json,sys; print(json.load(sys.stdin)["services"]["worker"]["environment"]["NEURAL_LAB_WORKER_IMAGE"])')"
    # Prepare before enqueue/start: a missing training image must not consume jobs.
    if [ "$training_image" = neural-lab-training:torch2.8-hf ]; then
      docker build -t "$training_image" -f docker/training.Dockerfile .
    else
      docker pull "$training_image"
    fi
    python3 -m neural_lab.manage doctor --planner demo --gpu-test --image "$training_image"
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
