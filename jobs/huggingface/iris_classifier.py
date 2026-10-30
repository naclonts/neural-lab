"""Public Iris connectivity/training control, not an architecture novelty claim."""
import argparse
import json
import os
from pathlib import Path
import time

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ.setdefault("OMP_NUM_THREADS", "4")

import torch
from torch import nn
from torch.nn import functional as F

from neural_lab.artifacts import ArtifactWriter, BestCheckpoint
from neural_lab.hf_data import load_public_dataset

FEATURES = ("SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm")
CLASSES = ("Iris-setosa", "Iris-versicolor", "Iris-virginica")
SPLIT_SEED, INIT_SEED = 8101, 8102


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args(argv)
    if not 1 <= args.steps <= 10000:
        parser.error("steps must be between 1 and 10000")
    output = Path("/output")
    writer = ArtifactWriter(output)
    dataset, downloaded = load_public_dataset("scikit-learn/iris", "train", writer)
    if len(dataset) != 150:
        raise ValueError("This Iris control expects the documented 150-row dataset")
    # Identifier columns are excluded, and preprocessing uses training rows only.
    features = torch.tensor([[row[name] for name in FEATURES] for row in dataset], dtype=torch.float32)
    labels = torch.tensor([CLASSES.index(row["Species"]) for row in dataset], dtype=torch.int64)
    generator = torch.Generator().manual_seed(SPLIT_SEED)
    train_rows, validation_rows = [], []
    for label in range(3):
        rows = (labels == label).nonzero().flatten()
        if len(rows) != 50:
            raise ValueError("Expected 50 Iris examples per class")
        rows = rows[torch.randperm(len(rows), generator=generator)]
        train_rows.extend(rows[:40].tolist())
        validation_rows.extend(rows[40:].tolist())
    train_rows = torch.tensor(train_rows)
    validation_rows = torch.tensor(validation_rows)
    mean = features[train_rows].mean(0)
    std = features[train_rows].std(0, unbiased=False).clamp_min(1e-8)
    normalized = (features - mean) / std
    splits = writer.save_dataset("iris-stratified-training-validation", {
        "train_features": normalized[train_rows], "train_labels": labels[train_rows],
        "validation_features": normalized[validation_rows], "validation_labels": labels[validation_rows],
        "train_row_indices": train_rows, "validation_row_indices": validation_rows,
        "training_mean": mean, "training_std": std,
    }, provenance={"kind": "derived-huggingface", "dataset": downloaded["source"],
                   "raw_artifact": downloaded["path"], "raw_sha256": downloaded["sha256"],
                   "split_seed": SPLIT_SEED, "stratification": "40 train / 10 validation per class",
                   "features": FEATURES, "classes": CLASSES,
                   "preprocessing": "standardize with training mean/population std; exclude Id"})
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_x, train_y = normalized[train_rows].to(device), labels[train_rows].to(device)
    val_x, val_y = normalized[validation_rows].to(device), labels[validation_rows].to(device)
    config = {"features": FEATURES, "classes": CLASSES, "split_seed": SPLIT_SEED,
              "init_seed": INIT_SEED, "steps": args.steps, "optimizer": "AdamW",
              "learning_rate": 0.03, "weight_decay": 0.001, "candidate_width": 16,
              "train_rows": 120, "validation_rows": 30,
              "normalization": {"mean": mean.tolist(), "std": std.tolist()},
              "architectures": {"baseline": "Linear(4,3)",
                                "candidate": "Linear(4,16), ReLU, Linear(16,3)"}}
    records = {}
    for arm in ("baseline", "candidate"):
        torch.manual_seed(INIT_SEED)
        model = (nn.Linear(4, 3) if arm == "baseline" else
                 nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 3))).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.03, weight_decay=0.001)
        best = BestCheckpoint()
        started = time.monotonic()
        for step in range(1, args.steps + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(train_x), train_y)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                best.observe(model, F.cross_entropy(model(train_x), train_y).item(), step)
        model.load_state_dict(best.state)
        model.eval()
        with torch.no_grad():
            logits = model(val_x)
            accuracy = (logits.argmax(-1) == val_y).float().mean().item()
            validation_loss = F.cross_entropy(logits, val_y).item()
        checkpoint = best.save(writer, f"iris-{arm}-best-training-loss", arm=arm, seed=INIT_SEED,
                               config=config, datasets=[downloaded["path"], splits["path"]])
        records[arm] = {"accuracy": accuracy, "validation_loss": validation_loss,
                        "best_training_loss": best.value, "selected_step": best.step,
                        "training_seconds": time.monotonic() - started,
                        "checkpoint": checkpoint["path"]}
    result = {"metrics": {"candidate_accuracy": records["candidate"]["accuracy"],
                           "baseline_accuracy": records["baseline"]["accuracy"],
                           "accuracy_delta": records["candidate"]["accuracy"] - records["baseline"]["accuracy"],
                           "candidate_heldout_loss": records["candidate"]["validation_loss"],
                           "baseline_heldout_loss": records["baseline"]["validation_loss"]},
              "summary": "Public Hugging Face Iris connectivity/training control: four measurement features, identifier excluded, stratified 120 training / 30 validation rows, training-only standardization, linear vs width-16 MLP. Best observed training-loss weights saved for both arms. Single split; exploratory control, not a novelty or independent confirmation claim."}
    (output / "details.json").write_text(json.dumps({"config": config, "device": str(device),
        "environment": {"device": str(device), "torch": str(torch.__version__),
                        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"},
        "records": records, "dataset": downloaded["source"]}, allow_nan=False, indent=2))
    (output / "result.json").write_text(json.dumps(result, allow_nan=False, indent=2))
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
