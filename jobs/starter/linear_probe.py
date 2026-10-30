"""Deterministic linear control; synthetic dataset/weight artifact contract v1."""
from neural_lab.demo_experiment import run

run({"seed": 1001, "width": 16, "depth": 0, "residual": False, "epochs": 25})
