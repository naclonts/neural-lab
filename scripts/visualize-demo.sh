#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
docker compose -f compose.yaml -f compose.visualization.yaml run --rm --no-deps lab \
  python /opt/neural-lab/examples/visualize_models.py --output /workspace/visualizations
echo "Open experiments/visualizations/{mlp,cnn,residual-mlp}/index.html locally."
