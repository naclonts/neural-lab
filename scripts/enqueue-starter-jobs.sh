#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
python3 -m neural_lab.queue enqueue jobs/starter/linear_probe.py --hypothesis "Linear-control architecture on a fixed synthetic classification task."
python3 -m neural_lab.queue enqueue jobs/starter/deep_mlp.py --hypothesis "A three-layer MLP improves held-out accuracy over the linear control."
python3 -m neural_lab.queue enqueue jobs/starter/residual_mlp.py --hypothesis "Residual connections improve a three-layer MLP at the same width and training budget."
