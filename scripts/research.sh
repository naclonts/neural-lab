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
    # The scientist designs verification jobs after validating the exact checker.
    # Legacy starter/candidate scripts remain available as explicit demos only.
    "${compose[@]}" up -d --build worker orchestrator
    "${compose[@]}" ps
    ;;
  start-pi)
    bash scripts/setup.sh
    export NEURAL_LAB_GPU_BACKEND
    NEURAL_LAB_GPU_BACKEND="$("${compose[@]}" config --format json | python3 -c 'import json,sys; print(json.load(sys.stdin)["services"]["worker"]["environment"]["NEURAL_LAB_GPU_BACKEND"])')"
    training_image="$("${compose[@]}" config --format json | python3 -c 'import json,sys; print(json.load(sys.stdin)["services"]["worker"]["environment"]["NEURAL_LAB_WORKER_IMAGE"])')"
    provider="$("${compose[@]}" config --format json | python3 -c 'import json,sys; print(json.load(sys.stdin)["services"]["pi-orchestrator"]["environment"]["PI_PROVIDER"])')"
    if [ "$provider" = ollama ]; then
      # Both pi auth probes perform inference outside the worker handoff.  A
      # running experiment could already own the GPU, so check host readiness
      # without calling pi; the orchestrator verifies Ollama during handoff.
      python3 -m neural_lab.manage doctor --planner demo --image "$training_image"
    else
      bash scripts/pi-auth.sh status
      python3 -m neural_lab.manage doctor --planner pi --image "$training_image"
    fi
    "${compose[@]}" up -d --build worker pi-orchestrator
    "${compose[@]}" ps
    ;;
  status)
    service="${2:-orchestrator}"
    python3 -m neural_lab.queue status
    "${compose[@]}" ps
    "${compose[@]}" logs --tail=15 "$service" worker
    ;;
  logs)
    service="${2:-orchestrator}"
    "${compose[@]}" logs --follow --tail=100 "$service" worker
    ;;
  stop) "${compose[@]}" down ;;
  pause-research) "${compose[@]}" stop "${2:-orchestrator}" ;;
  resume-research) "${compose[@]}" up -d "${2:-orchestrator}" ;;
  *)
    echo 'Usage: bash scripts/research.sh [start|start-pi|status|logs|stop|pause-research|resume-research]' >&2
    echo '  status, logs, pause-research, and resume-research accept an optional' >&2
    echo '  service name (for example: logs pi-orchestrator); default orchestrator.' >&2
    exit 2
    ;;
esac
