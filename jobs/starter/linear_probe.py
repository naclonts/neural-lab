"""Deterministic linear control for the synthetic classification task."""
from neural_lab.demo_experiment import run

run({"seed": 1001, "width": 16, "depth": 0, "residual": False, "epochs": 25})
