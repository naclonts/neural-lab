"""Generate portable visualizations for several network families."""
from pathlib import Path
import argparse

import torch
from torch import nn

from neural_lab.visualization import export_visualizations


class ResidualMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(16, 32)
        self.block = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 32))
        self.output = nn.Linear(32, 4)

    def forward(self, x):
        hidden = torch.relu(self.input(x))
        return self.output(hidden + self.block(hidden))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/workspace/visualizations"))
    args = parser.parse_args()
    torch.manual_seed(7)
    models = {
        "mlp": (
            nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4)),
            (torch.randn(2, 16),),
        ),
        "cnn": (
            nn.Sequential(
                nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(),
                nn.MaxPool2d(2), nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(16, 4),
            ),
            (torch.randn(2, 1, 28, 28),),
        ),
        "residual-mlp": (ResidualMLP(), (torch.randn(2, 16),)),
    }
    for name, (model, inputs) in models.items():
        paths = export_visualizations(model, inputs, args.output / name, title=name)
        print(name, "->", paths["index.html"])


if __name__ == "__main__":
    main()
