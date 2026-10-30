"""Exploratory development-only product-scale diagnostic; import does not train.

A frozen per-block scalar matches initial aggregate product fused RMS to additive
fused RMS on the additive training trajectory, BEFORE the shared up/GELU.
No per-example normalization, learned gain, or test evaluation is performed.
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

from neural_lab.artifacts import ArtifactWriter, BestCheckpoint

TEACHER_SEEDS = (5101, 5102, 5103)
DATA_SEEDS = {"train": 6101, "dev": 6102}
INIT_SEEDS = (7101, 7102)
WIDTH, RANK, BLOCKS = 48, 16, 2
STEPS, INPUT_DIM, CLASSES = 55, 16, 4
CHECKPOINTS = (0, 10, 25, 55)
EPSILON = 1e-8
ARMS = ("dense", "additive", "product", "scaled_product")


def task(teacher_seed):
    teacher_rng = torch.Generator().manual_seed(teacher_seed)
    linear = torch.randn(INPUT_DIM, CLASSES, generator=teacher_rng)
    interaction = torch.randn(4, CLASSES, generator=teacher_rng)
    sinusoid = torch.randn(4, CLASSES, generator=teacher_rng)
    data = {}
    for split, count in (("train", 1536), ("dev", 512)):
        data_rng = torch.Generator().manual_seed(DATA_SEEDS[split])
        features = torch.randn(count, INPUT_DIM, generator=data_rng)
        logits = 0.55 * (features @ linear)
        logits += 1.7 * ((features[:, :4] * features[:, 4:8]) @ interaction)
        logits += 0.3 * torch.sin(features[:, 8:12]) @ sinusoid
        data[split] = (features, logits.argmax(dim=1))
    return data


def rms(tensor):
    return tensor.square().mean().sqrt().item()


class DenseBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.mix = nn.Linear(WIDTH, WIDTH)

    def forward(self, hidden, diagnostics=None):
        preactivation = self.mix(hidden)
        branch = F.gelu(preactivation)
        if diagnostics is not None:
            diagnostics.append({"input_rms": rms(hidden),
                                "preactivation_rms": rms(preactivation),
                                "branch_rms": rms(branch), "branch_abs_max": branch.abs().max().item(),
                                "output_rms": rms(hidden + branch)})
        return hidden + branch


class CenteredBlock(nn.Module):
    def __init__(self, product):
        super().__init__()
        self.a = nn.Linear(WIDTH, RANK)
        self.b = nn.Linear(WIDTH, RANK)
        self.up = nn.Linear(RANK, WIDTH)
        self.product = product
        self.register_buffer("scale", torch.tensor(1.0))

    def forward(self, hidden, diagnostics=None):
        a, b = self.a(hidden), self.b(hidden)
        a = a - a.mean(dim=-1, keepdim=True)
        b = b - b.mean(dim=-1, keepdim=True)
        fused = a * b if self.product else (a + b) / math.sqrt(2.0)
        # Frozen training-calibrated scalar before up/GELU; no per-example normalization.
        mixed = fused * self.scale
        preactivation = self.up(mixed)
        branch = F.gelu(preactivation)
        if diagnostics is not None:
            diagnostics.append({"input_rms": rms(hidden), "centered_a_rms": rms(a),
                                "centered_b_rms": rms(b), "fused_raw_rms": rms(fused),
                                "fused_to_up_rms": rms(mixed),
                                "fused_raw_abs_max": fused.abs().max().item(),
                                "fused_to_up_abs_max": mixed.abs().max().item(),
                                "preactivation_rms": rms(preactivation),
                                "branch_rms": rms(branch), "branch_abs_max": branch.abs().max().item(),
                                "output_rms": rms(hidden + branch)})
        return hidden + branch


class Network(nn.Module):
    def __init__(self, arm):
        super().__init__()
        self.input = nn.Linear(INPUT_DIM, WIDTH)
        self.blocks = nn.ModuleList(
            DenseBlock() if arm == "dense" else CenteredBlock(
                arm in ("product", "scaled_product"))
            for _ in range(BLOCKS))
        self.output = nn.Linear(WIDTH, CLASSES)

    def forward(self, features, diagnostics=None):
        hidden = F.gelu(self.input(features))
        for block in self.blocks:
            hidden = block(hidden, diagnostics)
        return self.output(hidden)


def digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def paired_models(seed):
    torch.manual_seed(seed)
    models = {arm: Network(arm) for arm in ARMS}
    for arm in ARMS[1:]:
        if arm != "additive":
            models[arm].load_state_dict(copy.deepcopy(models["additive"].state_dict()))
        for name in ("input", "output"):
            getattr(models[arm], name).load_state_dict(
                copy.deepcopy(getattr(models["dense"], name).state_dict()))
    reference = models["additive"].state_dict()
    for arm in ARMS[1:]:
        assert all(torch.equal(value, models[arm].state_dict()[key])
                   for key, value in reference.items())
    for arm in ARMS:
        for name in ("input", "output"):
            assert all(torch.equal(value, getattr(models[arm], name).state_dict()[key])
                       for key, value in getattr(models["dense"], name).state_dict().items())
    return models


@torch.no_grad()
def calibrate(models, train_x):
    """Use one shared initial additive hidden trajectory for both RMS terms."""
    additive = models["additive"]
    hidden = F.gelu(additive.input(train_x))
    calibration = []
    for index, block in enumerate(additive.blocks):
        a, b = block.a(hidden), block.b(hidden)
        a = a - a.mean(-1, keepdim=True)
        b = b - b.mean(-1, keepdim=True)
        numerator, denominator = rms((a + b) / math.sqrt(2.0)), rms(a * b)
        gain = numerator / max(denominator, EPSILON)
        if not all(math.isfinite(value) for value in (numerator, denominator, gain)):
            raise FloatingPointError(f"nonfinite calibration at block {index}")
        # Frozen buffer moves with the model and is recorded in state digests.
        models["scaled_product"].blocks[index].scale.fill_(gain)
        calibration.append({"block": index, "additive_fused_rms": numerator,
                            "product_fused_rms": denominator, "frozen_gain": gain})
        hidden = block(hidden)
    return calibration


@torch.no_grad()
def evaluate(model, data):
    model.eval()
    result = {}
    for split, (features, labels) in data.items():
        blocks = []
        logits = model(features, blocks)
        if not torch.isfinite(logits).all():
            raise FloatingPointError(f"nonfinite {split} logits")
        result[split] = {"loss": F.cross_entropy(logits, labels).item(),
                         "accuracy": (logits.argmax(-1) == labels).float().mean().item(),
                         "logit_rms": rms(logits),
                         "centered_logit_rms": rms(logits - logits.mean(-1, keepdim=True)),
                         "logit_abs_max": logits.abs().max().item(),
                         "logit_mean_abs": logits.abs().mean().item(),
                         "predictive_entropy": -(logits.softmax(-1) * logits.log_softmax(-1)).sum(-1).mean().item(),
                         "mean_max_probability": logits.softmax(-1).max(-1).values.mean().item(),
                         "blocks": blocks}
    return result


def fit(model, data, device, *, artifacts, arm, seed, dataset_path):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.006, weight_decay=0.001)
    checkpoints = {"0": evaluate(model, data)}
    best = BestCheckpoint(scope="best training loss among recorded diagnostic evaluations; held-out data not used for selection")
    best.observe(model, checkpoints["0"]["train"]["loss"], 0)
    train_x, train_y = data["train"]
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.monotonic()
    training_seconds = 0.0
    for step in range(1, STEPS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(train_x), train_y)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite training loss at step {step}")
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError(f"nonfinite gradient at step {step}")
        optimizer.step()
        if step in CHECKPOINTS:
            if device.type == "cuda":
                torch.cuda.synchronize()
            training_seconds += time.monotonic() - started
            checkpoints[str(step)] = evaluate(model, data)
            best.observe(model, checkpoints[str(step)]["train"]["loss"], step)
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.monotonic()
    best.save(artifacts, f"{arm}-{seed}-best", arm=arm, seed=seed,
              config={"script": "product_scale_diagnostic.py", "width": WIDTH, "rank": RANK,
                      "blocks": BLOCKS, "steps": STEPS, "observed_updates": CHECKPOINTS},
              datasets=[dataset_path])
    return {"checkpoints": checkpoints, "training_seconds": training_seconds}


def main():
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    output = Path("/output")
    artifacts = ArtifactWriter(output)
    records = []
    counts = None
    for teacher_seed in TEACHER_SEEDS:
        cpu_data = task(teacher_seed)
        dataset = artifacts.save_dataset(f"cross-feature-teacher-{teacher_seed}",
            {f"{split}_{name}": tensor for split, pair in cpu_data.items()
             for name, tensor in zip(("x", "y"), pair)},
            provenance={"kind": "generated", "generator": "product_scale_diagnostic.py:task",
                        "teacher_seed": teacher_seed, "data_seeds": DATA_SEEDS,
                        "description": "Gaussian 16-feature classification with linear, multiplicative, and sinusoidal teacher; four classes",
                        "split": "1536 training rows and 512 development rows; independent data seeds", "downloads": False})
        hashes = {split: {"x": digest(x), "y": digest(y)} for split, (x, y) in cpu_data.items()}
        data = {split: tuple(t.to(device) for t in pair) for split, pair in cpu_data.items()}
        for init_seed in INIT_SEEDS:
            models = paired_models(init_seed)
            counts = {arm: sum(p.numel() for p in model.parameters()) for arm, model in models.items()}
            assert counts == {arm: 5716 if arm == "dense" else 5780 for arm in ARMS}
            calibration = calibrate(models, cpu_data["train"][0])
            init_hashes = {arm: {key: digest(value) for key, value in model.state_dict().items()}
                           for arm, model in models.items()}
            order = ARMS if len(records) % 2 == 0 else tuple(reversed(ARMS))
            row = {"teacher_seed": teacher_seed, "data_seeds": DATA_SEEDS,
                   "init_seed": init_seed, "split_sha256": hashes,
                   "parameters": counts, "calibration": calibration,
                   "initial_state_sha256": init_hashes, "arm_order": order, "arms": {}}
            for arm in order:
                torch.manual_seed(init_seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(init_seed)
                row["arms"][arm] = fit(models[arm], data, device, artifacts=artifacts, arm=arm,
                    seed=f"teacher-{teacher_seed}-init-{init_seed}", dataset_path=dataset["path"])
            records.append(row)
    metrics = {f"{arm}_{split}_{key}": sum(
        row["arms"][arm]["checkpoints"][str(STEPS)][split][key] for row in records) / len(records)
        for arm in ARMS for split in ("train", "dev") for key in ("loss", "accuracy")}
    metrics.update({f"{arm}_parameters": count for arm, count in counts.items()})
    comparisons = (("scaled_product", "product"), ("product", "additive"),
                   ("scaled_product", "additive"), ("scaled_product", "dense"))
    paired = {}
    for left, right in comparisons:
        teacher_differences = []
        for teacher in TEACHER_SEEDS:
            values = [row["arms"][left]["checkpoints"][str(STEPS)]["dev"]["loss"] -
                      row["arms"][right]["checkpoints"][str(STEPS)]["dev"]["loss"]
                      for row in records if row["teacher_seed"] == teacher]
            accuracy_pp = [100.0 * (row["arms"][left]["checkpoints"][str(STEPS)]["dev"]["accuracy"] -
                                      row["arms"][right]["checkpoints"][str(STEPS)]["dev"]["accuracy"])
                           for row in records if row["teacher_seed"] == teacher]
            teacher_differences.append({"teacher_seed": teacher, "mean_dev_loss_difference": sum(values) / len(values),
                                        "mean_dev_accuracy_difference_pp": sum(accuracy_pp) / len(accuracy_pp)})
        key = f"{left}_minus_{right}"
        values = [entry["mean_dev_loss_difference"] for entry in teacher_differences]
        paired[key] = {"teachers": teacher_differences, "mean": sum(values) / len(values),
                       "min": min(values), "max": max(values),
                       "mean_dev_accuracy_difference_pp": sum(entry["mean_dev_accuracy_difference_pp"] for entry in teacher_differences) / len(teacher_differences)}
        metrics[f"{key}_dev_loss"] = paired[key]["mean"]
    details = {"status": "completed", "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "config": {"teacher_seeds": TEACHER_SEEDS, "data_seeds": DATA_SEEDS,
                          "init_seeds": INIT_SEEDS, "arms": ARMS, "width": WIDTH,
                          "rank": RANK, "blocks": BLOCKS, "steps": STEPS,
                          "checkpoints": CHECKPOINTS, "train_examples": 1536, "dev_examples": 512,
                          "input_dim": INPUT_DIM, "classes": CLASSES, "parameters": counts,
                          "optimizer": "AdamW", "learning_rate": 0.006, "weight_decay": 0.001,
                          "batch": "full", "epsilon": EPSILON,
                          "scaling": "frozen per-block scalar = RMS(additive fused) / max(RMS(product fused), epsilon), on initial additive training hidden trajectory; product multiplied before up/GELU",
                          "primary_endpoint": "scaled_product minus product development cross entropy at update 55; exploratory, no test selection or promotion",
                          "diagnostic_rms": "global tensor sqrt(mean(square)); no epsilon; each block recorded separately"},
               "environment": {"device": str(device), "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
                               "python": platform.python_version(), "torch": str(torch.__version__),
                               "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                               "threads": torch.get_num_threads(), "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                               "cudnn_benchmark": torch.backends.cudnn.benchmark,
                               "cudnn_deterministic": torch.backends.cudnn.deterministic,
                               "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                               "cudnn_tf32": torch.backends.cudnn.allow_tf32,
                               "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"]},
               "records": records, "paired_dev_loss": paired,
               "limitations": "Three teachers in one previously inspected synthetic family; initialization repeats are not independent task replicates. Development-only diagnostic; no confirmation, novelty, superiority, or speed claim. Initial aggregate fused RMS matching does not match variance, per-example amplitude, branch RMS, or later scale. Scaling changes optimization and does not isolate function class. Timings exclude diagnostics, have no warmup, and are descriptive only."}
    result = {"metrics": metrics, "summary": "Development-only centered product scale diagnostic with dense, raw additive/product, and frozen training-calibrated pre-up scalar product control; 55 full-batch updates, three fresh teachers and two paired initializations. No selection or promotion."}
    result["summary"] += " Data and best observed training-loss weights: artifacts.json; reported metrics remain final-update metrics."
    # Serialize both before writing, so a nonfinite diagnostic cannot yield a completed result.
    details_json = json.dumps(details, allow_nan=False, indent=2)
    result_json = json.dumps(result, allow_nan=False, indent=2)
    (output / "details.json").write_text(details_json)
    (output / "result.json").write_text(result_json)
    print(result_json)


if __name__ == "__main__":
    main()
