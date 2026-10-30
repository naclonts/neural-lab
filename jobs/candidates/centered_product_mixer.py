"""Exploratory centered product residual versus dense and additive controls.

Self-contained queue payload; importing this module does not run training.
"""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ.setdefault("OMP_NUM_THREADS", "4")

import torch
from torch import nn
from torch.nn import functional as F

SEEDS = (1101, 1102, 1103)
WIDTH, RANK, BLOCKS = 48, 16, 2
EPOCHS, INPUT_DIM, CLASSES = 55, 16, 4
ARMS = ("baseline", "candidate", "additive")


def task(seed):
    generator = torch.Generator().manual_seed(seed)
    features = torch.randn(2048, INPUT_DIM, generator=generator)
    linear = torch.randn(INPUT_DIM, CLASSES, generator=generator)
    interaction = torch.randn(4, CLASSES, generator=generator)
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

    def forward(self, hidden):
        return hidden + F.gelu(self.mix(hidden))


class CenteredBlock(nn.Module):
    def __init__(self, product):
        super().__init__()
        self.a = nn.Linear(WIDTH, RANK)
        self.b = nn.Linear(WIDTH, RANK)
        self.up = nn.Linear(RANK, WIDTH)
        self.product = product

    def forward(self, hidden):
        a, b = self.a(hidden), self.b(hidden)
        a = a - a.mean(dim=-1, keepdim=True)
        b = b - b.mean(dim=-1, keepdim=True)
        mixed = a * b if self.product else (a + b) / math.sqrt(2.0)
        return hidden + F.gelu(self.up(mixed))


class Network(nn.Module):
    def __init__(self, arm):
        super().__init__()
        self.input = nn.Linear(INPUT_DIM, WIDTH)
        self.blocks = nn.ModuleList(
            DenseBlock() if arm == "baseline" else CenteredBlock(arm == "candidate")
            for _ in range(BLOCKS)
        )
        self.output = nn.Linear(WIDTH, CLASSES)

    def forward(self, features):
        hidden = F.gelu(self.input(features))
        for block in self.blocks:
            hidden = block(hidden)
        return self.output(hidden)


def paired_models(seed):
    torch.manual_seed(seed)
    models = {arm: Network(arm) for arm in ARMS}
    # Explicit copies avoid constructor RNG consumption shifting the shared head.
    for arm in ("candidate", "additive"):
        models[arm].input.load_state_dict(copy.deepcopy(models["baseline"].input.state_dict()))
        models[arm].output.load_state_dict(copy.deepcopy(models["baseline"].output.state_dict()))
    # Product and additive differ only in the parameter-free combining operation.
    models["additive"].load_state_dict(copy.deepcopy(models["candidate"].state_dict()))
    for arm in ("candidate", "additive"):
        for name in ("input", "output"):
            assert all(torch.equal(left, right) for left, right in zip(
                getattr(models["baseline"], name).parameters(),
                getattr(models[arm], name).parameters(),
            ))
    return models


def fit(model, data, device):
    train_x, train_y, val_x, val_y = data
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.006, weight_decay=0.001)
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.monotonic()
    for _ in range(EPOCHS):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(train_x), train_y)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        logits = model(val_x)
        accuracy = (logits.argmax(dim=1) == val_y).float().mean().item()
        heldout_loss = F.cross_entropy(logits, val_y).item()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return {"accuracy": accuracy, "heldout_loss": heldout_loss,
            "training_seconds": time.monotonic() - started}


def main():
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    records, counts = [], None
    for seed in SEEDS:
        models = paired_models(seed)
        counts = {arm: sum(p.numel() for p in model.parameters())
                  for arm, model in models.items()}
        assert counts == {"baseline": 5716, "candidate": 5780, "additive": 5780}
        assert all(abs(count / counts["baseline"] - 1) < 0.05 for count in counts.values())
        assert WIDTH * WIDTH + WIDTH == 2352
        assert 3 * WIDTH * RANK + 2 * RANK + WIDTH == 2384
        cpu_data = task(seed)
        fingerprints = {name: hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()
                        for name, tensor in zip(("train_x", "train_y", "heldout_x", "heldout_y"), cpu_data)}
        data = tuple(tensor.to(device) for tensor in cpu_data)
        row = {"seed": seed, "task_tensor_sha256": fingerprints}
        for arm in ARMS:
            torch.manual_seed(seed)
            row[arm] = fit(models[arm], data, device)
        row["candidate_minus_baseline_accuracy"] = row["candidate"]["accuracy"] - row["baseline"]["accuracy"]
        row["candidate_minus_additive_accuracy"] = row["candidate"]["accuracy"] - row["additive"]["accuracy"]
        row["candidate_minus_baseline_loss"] = row["candidate"]["heldout_loss"] - row["baseline"]["heldout_loss"]
        row["candidate_minus_additive_loss"] = row["candidate"]["heldout_loss"] - row["additive"]["heldout_loss"]
        records.append(row)
    metrics = {f"{arm}_{key}": sum(row[arm][key] for row in records) / len(records)
               for arm in ARMS for key in ("accuracy", "heldout_loss", "training_seconds")}
    metrics.update({f"{arm}_parameters": count for arm, count in counts.items()})
    metrics["accuracy_delta"] = metrics["candidate_accuracy"] - metrics["baseline_accuracy"]
    for key in records[0]:
        if key.startswith("candidate_minus_"):
            metrics[key] = sum(row[key] for row in records) / len(records)
    result = {"metrics": metrics, "summary": (
        "Centered signed product residual versus dense residual and identically initialized "
        "centered additive ablation; three paired seeds on synthetic cross-feature classification. "
        "Width 48, two blocks, rank 16, 55 full-batch AdamW updates; parameter budgets within 5%. "
        "Exploratory architecture hypothesis, not a global novelty or superiority claim."
    )}
    details = {
        "config": {"seeds": SEEDS, "width": WIDTH, "rank": RANK, "blocks": BLOCKS,
                   "epochs": EPOCHS, "train_examples": 1536, "heldout_examples": 512,
                   "input_dim": INPUT_DIM, "classes": CLASSES, "optimizer": "AdamW",
                   "learning_rate": 0.006, "weight_decay": 0.001, "batch": "full",
                   "centering": "each projection over rank channels per example",
                   "product": "center(a) * center(b), without post-product centering",
                   "additive": "(center(a) + center(b)) / sqrt(2)",
                   "arm_order": ARMS, "parameters": counts},
        "environment": {"device": str(device), "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
                        "python": platform.python_version(), "torch": str(torch.__version__),
                        "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                        "threads": torch.get_num_threads(), "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                        "cudnn_benchmark": torch.backends.cudnn.benchmark,
                        "cudnn_deterministic": torch.backends.cudnn.deterministic,
                        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
                        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"]},
        "records": records,
        "limitations": "Single synthetic task family, three seeds, fixed arm timing order; deterministic settings do not imply identical results across hardware or library versions.",
    }
    output = Path("/output")
    output.mkdir(parents=True, exist_ok=True)
    (output / "details.json").write_text(json.dumps(details, allow_nan=False, indent=2))
    (output / "result.json").write_text(json.dumps(result, allow_nan=False, indent=2))
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
