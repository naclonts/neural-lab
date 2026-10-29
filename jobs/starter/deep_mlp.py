"""Depth ablation against the linear control."""
from neural_lab.demo_experiment import run

run({"seed": 1001, "width": 64, "depth": 3, "residual": False, "epochs": 25})
