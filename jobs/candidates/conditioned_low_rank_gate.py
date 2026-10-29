"""Compare a dense residual MLP with input-conditioned low-rank residual gates."""
import json
import os
from pathlib import Path
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import torch
from torch import nn
from torch.nn import functional as F


SEEDS = (1101, 1102, 1103)
WIDTH = 48
RANK = 16
EPOCHS = 55
INPUT_DIM = 16
CLASSES = 4


def task(seed):
    generator = torch.Generator().manual_seed(seed)
    features = torch.randn(2048, INPUT_DIM, generator=generator)
    linear = torch.randn(INPUT_DIM, CLASSES, generator=generator)
    interaction = torch.randn(4, CLASSES, generator=generator)
    # A fixed teacher with linear signal and nonlinear cross-feature interactions.
    logits = 0.55 * (features @ linear)
    logits += 1.7 * ((features[:, :4] * features[:, 4:8]) @ interaction)
    logits += 0.3 * torch.sin(features[:, 8:12]) @ torch.randn(
        4, CLASSES, generator=generator
    )
    labels = logits.argmax(dim=1)
    return features[:1536], labels[:1536], features[1536:], labels[1536:]


class DenseBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.mix = nn.Linear(WIDTH, WIDTH)

    def forward(self, hidden, original):
        return hidden + F.gelu(self.mix(hidden))


class GatedLowRankBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.down = nn.Linear(WIDTH, RANK)
        self.up = nn.Linear(RANK, WIDTH)
        self.gate = nn.Linear(INPUT_DIM, WIDTH)

    def forward(self, hidden, original):
        transformed = self.up(F.gelu(self.down(hidden)))
        return hidden + torch.sigmoid(self.gate(original)) * F.gelu(transformed)


class Network(nn.Module):
    def __init__(self, block_type):
        super().__init__()
        self.input = nn.Linear(INPUT_DIM, WIDTH)
        self.blocks = nn.ModuleList(block_type() for _ in range(2))
        self.output = nn.Linear(WIDTH, CLASSES)

    def forward(self, features):
        hidden = F.gelu(self.input(features))
        for block in self.blocks:
            hidden = block(hidden, features)
        return self.output(hidden)


def fit(model, data, device):
    train_x, train_y, val_x, val_y = (
        tensor.to(device) for tensor in data
    )
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.006, weight_decay=0.001)
    started = time.monotonic()
    for _ in range(EPOCHS):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(train_x), train_y)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        predicted = model(val_x).argmax(dim=1)
        accuracy = (predicted == val_y).float().mean().item()
        heldout_loss = F.cross_entropy(model(val_x), val_y).item()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return accuracy, heldout_loss, time.monotonic() - started


def main():
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    baseline_params = sum(p.numel() for p in Network(DenseBlock).parameters())
    candidate_params = sum(p.numel() for p in Network(GatedLowRankBlock).parameters())
    assert abs(candidate_params / baseline_params - 1) < 0.05
    records = []
    for seed in SEEDS:
        data = task(seed)
        torch.manual_seed(seed)
        baseline = fit(Network(DenseBlock), data, device)
        torch.manual_seed(seed)
        candidate = fit(Network(GatedLowRankBlock), data, device)
        records.append(
            {"seed": seed, "baseline_accuracy": baseline[0], "candidate_accuracy": candidate[0],
             "baseline_loss": baseline[1], "candidate_loss": candidate[1],
             "baseline_seconds": baseline[2], "candidate_seconds": candidate[2]}
        )
    mean = lambda key: sum(row[key] for row in records) / len(records)
    result = {
        "metrics": {
            "baseline_accuracy": mean("baseline_accuracy"),
            "candidate_accuracy": mean("candidate_accuracy"),
            "accuracy_delta": mean("candidate_accuracy") - mean("baseline_accuracy"),
            "baseline_heldout_loss": mean("baseline_loss"),
            "candidate_heldout_loss": mean("candidate_loss"),
            "baseline_parameters": baseline_params,
            "candidate_parameters": candidate_params,
            "baseline_training_seconds": mean("baseline_seconds"),
            "candidate_training_seconds": mean("candidate_seconds"),
        },
        "summary": (
            "Input-conditioned low-rank residual gate vs dense residual MLP on a fixed "
            "synthetic cross-feature classification task; three seeds, 1536 train and "
            "512 held-out examples per seed, matched width/depth/optimizer/epochs, "
            "parameter counts within 5%. Exploratory comparison, not a novelty claim."
        ),
    }
    output = Path("/output")
    output.mkdir(parents=True, exist_ok=True)
    (output / "details.json").write_text(json.dumps({"device": str(device), "records": records}, indent=2))
    (output / "result.json").write_text(json.dumps(result, allow_nan=False, indent=2))
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
