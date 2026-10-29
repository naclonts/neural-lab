"""Residual branch ablation with the same task, seed, and training budget."""
from neural_lab.demo_experiment import run

run({"seed": 1001, "width": 64, "depth": 3, "residual": True, "epochs": 25})
