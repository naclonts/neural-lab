"""Small synthetic classification experiments for the demo planner."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn


class Candidate(nn.Module):
    def __init__(self, width, depth, residual):
        super().__init__()
        self.input = nn.Linear(16, width)
        self.blocks = nn.ModuleList([nn.Linear(width, width) for _ in range(depth)])
        self.output = nn.Linear(width, 4)
        self.residual = residual

    def forward(self, x):
        hidden = torch.relu(self.input(x))
        for block in self.blocks:
            transformed = torch.relu(block(hidden))
            hidden = hidden + transformed if self.residual else transformed
        return self.output(hidden)


def run(config):
    from neural_lab.artifacts import ArtifactWriter, BestCheckpoint
    from neural_lab.visualization import export_visualizations

    output = Path("/output")
    artifacts = ArtifactWriter(output)
    torch.manual_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Fixed, reproducible synthetic task; no downloads or private data.
    features = torch.randn(640, 16)
    teacher = torch.randn(16, 4)
    labels = (features @ teacher + 0.25 * torch.sin(features[:, :4])).argmax(dim=1)
    train_x, val_x = features[:512].to(device), features[512:].to(device)
    train_y, val_y = labels[:512].to(device), labels[512:].to(device)
    dataset = artifacts.save_dataset("synthetic-seed-" + str(config["seed"]),
        {"train_x": features[:512], "train_y": labels[:512],
         "validation_x": features[512:], "validation_y": labels[512:], "teacher": teacher},
        provenance={"kind": "generated", "generator": "neural_lab.demo_experiment.run",
                    "seed": config["seed"], "description": "Gaussian 16-feature classification; linear teacher plus sinusoid; four classes",
                    "split": "first 512 training rows, last 128 validation rows", "downloads": False})

    def train(model, arm):
        model = model.to(device)
        best = BestCheckpoint()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        for step in range(config["epochs"]):
            model.train()
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(train_x), train_y)
            best.observe(model, loss.item(), step)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            best.observe(model, nn.functional.cross_entropy(model(train_x), train_y).item(), config["epochs"])
            accuracy = (model(val_x).argmax(dim=1) == val_y).float().mean().item()
        best.save(artifacts, f"{arm}-seed-{config['seed']}-best", arm=arm, seed=config["seed"],
                  config={**config, "architecture": "linear" if arm == "baseline" else "Candidate"}, datasets=[dataset["path"]])
        return model, accuracy

    torch.manual_seed(config["seed"])
    _, baseline = train(nn.Linear(16, 4), "baseline")
    torch.manual_seed(config["seed"])
    model, accuracy = train(Candidate(config["width"], config["depth"], config["residual"]), "candidate")
    model = model.cpu()
    (output / "config.json").write_text(json.dumps(config, indent=2))
    # Small demo weights use plain JSON rather than a pickle checkpoint.
    weights = {name: tensor.detach().cpu().tolist() for name, tensor in model.state_dict().items()}
    (output / "model-state.json").write_text(json.dumps(weights))
    visualization = "ok"
    try:
        export_visualizations(model, (features[:2],), output / "visualizations", title="Demo candidate")
    except Exception as exc:
        visualization = "failed: " + type(exc).__name__
    summary = (
        f"Synthetic seed={config['seed']}; train=512, held-out validation=128; "
        f"linear baseline={baseline:.4f}, candidate={accuracy:.4f}. "
        f"Visualization: {visualization}. Data and best-training-loss weights: artifacts.json. "
        "Reported metrics remain final-update metrics. Exploratory only; no novelty claim."
    )
    result = {
        "metrics": {"validation_accuracy": accuracy, "baseline_accuracy": baseline,
                    "accuracy_delta": accuracy - baseline},
        "summary": summary,
    }
    (output / "result.json").write_text(json.dumps(result, allow_nan=False, indent=2))
    print(json.dumps(result))
