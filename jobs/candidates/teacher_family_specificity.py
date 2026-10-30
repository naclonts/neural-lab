"""Preregistered development-only rotated teacher-family specificity screen.

Import defines the experiment only; all training belongs in the queued worker.
"""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import platform

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ.setdefault("OMP_NUM_THREADS", "4")

import torch
from torch import nn
from torch.nn import functional as F
from neural_lab.artifacts import ArtifactWriter, BestCheckpoint

TEACHER_SEEDS = (8101, 8102, 8103)
DATA_SEEDS = ((9101, 9102), (9111, 9112), (9121, 9122))
INIT_SEEDS = (10101, 10102)
FAMILIES = ("additive", "interaction")
ARMS = ("dense", "raw_additive", "raw_product", "tanh_additive", "tanh_product")
WIDTH, RANK, BLOCKS = 48, 16, 2
STEPS, INPUT_DIM, CLASSES = 40, 16, 4
CHECKPOINTS = (0, 10, 25, 40)


def task(teacher_seed, data_seeds):
    rng = torch.Generator().manual_seed(teacher_seed)
    q, r = torch.linalg.qr(torch.randn(INPUT_DIM, INPUT_DIM, generator=rng))
    # Fix the QR sign ambiguity, including a deterministic zero convention.
    signs = torch.where(r.diag() < 0, -torch.ones(INPUT_DIM), torch.ones(INPUT_DIM))
    q = q * signs.unsqueeze(0)
    coefficients = {}
    audit = {}
    for family, dimension in (("additive", 8), ("interaction", 4)):
        w = torch.randn(dimension, CLASSES, generator=rng)
        audit[f"teacher_{family}_raw_coefficients"] = w.clone()
        w = w - w.mean(-1, keepdim=True)
        scale = w.square().sum(0).mean().sqrt()
        if not torch.isfinite(scale) or scale <= 0:
            raise FloatingPointError("invalid teacher coefficient normalization")
        audit[f"teacher_{family}_normalizer"] = scale
        w = w / scale
        coefficients[family] = w
    shared = {**audit, "rotation_q": q, **{f"teacher_{k}_coefficients": w for k, w in coefficients.items()}}
    datasets = {family: {} for family in FAMILIES}
    for split, count, seed in zip(("train", "dev"), (1536, 512), data_seeds):
        z = torch.randn(count, INPUT_DIM, generator=torch.Generator().manual_seed(seed))
        x = z @ q
        shared[f"{split}_latent_z"] = z
        for family in FAMILIES:
            basis = z[:, :8] if family == "additive" else z[:, :4] * z[:, 4:8]
            logits = basis @ coefficients[family]
            shared[f"{split}_{family}_teacher_logits"] = logits
            datasets[family][split] = (x, logits.argmax(-1))
    return datasets, shared


def rms(tensor):
    return tensor.square().mean().sqrt().item()


def magnitude(tensor):
    if not torch.isfinite(tensor).all():
        raise FloatingPointError("nonfinite diagnostic activation")
    per_example = tensor.square().mean(-1).sqrt()
    return {"rms": rms(tensor), "abs_max": tensor.abs().max().item(),
            "per_example_rms_quantiles": dict(zip(
                ("min", "p25", "p50", "p75", "max"),
                torch.quantile(per_example, per_example.new_tensor([0., .25, .5, .75, 1.])).tolist()))}


class DenseBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.mix = nn.Linear(WIDTH, WIDTH)

    def forward(self, hidden, diagnostics=None):
        preactivation = self.mix(hidden)
        branch = F.gelu(preactivation)
        if diagnostics is not None:
            diagnostics.append({"magnitudes": {name: magnitude(value) for name, value in
                                (("input", hidden), ("preactivation", preactivation),
                                 ("branch", branch), ("output", hidden + branch))},
                                "input_rms": rms(hidden),
                                "preactivation_rms": rms(preactivation),
                                "branch_rms": rms(branch), "branch_abs_max": branch.abs().max().item(),
                                "output_rms": rms(hidden + branch)})
        return hidden + branch


class CenteredBlock(nn.Module):
    def __init__(self, product, bounded):
        super().__init__()
        self.a = nn.Linear(WIDTH, RANK)
        self.b = nn.Linear(WIDTH, RANK)
        self.up = nn.Linear(RANK, WIDTH)
        self.product = product
        self.bounded = bounded

    def forward(self, hidden, diagnostics=None):
        a, b = self.a(hidden), self.b(hidden)
        a = a - a.mean(dim=-1, keepdim=True)
        b = b - b.mean(dim=-1, keepdim=True)
        p, q = (torch.tanh(a), torch.tanh(b)) if self.bounded else (a, b)
        fused = p * q if self.product else (p + q) / math.sqrt(2.0)
        mixed = fused
        preactivation = self.up(mixed)
        branch = F.gelu(preactivation)
        if diagnostics is not None:
            diagnostics.append({"magnitudes": {name: magnitude(value) for name, value in
                                (("input", hidden), ("centered_a", a), ("centered_b", b),
                                 ("factor_a", p), ("factor_b", q), ("fusion", fused),
                                 ("preactivation", preactivation), ("branch", branch),
                                 ("output", hidden + branch))},
                                "input_rms": rms(hidden), "centered_a_rms": rms(a),
                                "centered_b_rms": rms(b),
                                "factor_a_abs_over_2_fraction": (a.abs() > 2).float().mean().item(),
                                "factor_b_abs_over_2_fraction": (b.abs() > 2).float().mean().item(),
                                "factor_a_tanh_abs_over_095_fraction": (a.tanh().abs() > 0.95).float().mean().item(),
                                "factor_b_tanh_abs_over_095_fraction": (b.tanh().abs() > 0.95).float().mean().item(), "fused_raw_rms": rms(fused),
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
                arm.endswith("product"), arm.startswith("tanh_"))
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
        if arm != "raw_additive":
            models[arm].load_state_dict(copy.deepcopy(models["raw_additive"].state_dict()))
        for name in ("input", "output"):
            getattr(models[arm], name).load_state_dict(
                copy.deepcopy(getattr(models["dense"], name).state_dict()))
    reference = models["raw_additive"].state_dict()
    for arm in ARMS[1:]:
        assert all(torch.equal(value, models[arm].state_dict()[key])
                   for key, value in reference.items())
    for arm in ARMS:
        for name in ("input", "output"):
            assert all(torch.equal(value, getattr(models[arm], name).state_dict()[key])
                       for key, value in getattr(models["dense"], name).state_dict().items())
    return models


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
        # Inspect gradients at this exact diagnostic state without changing weights.
        with torch.enable_grad():
            probe_loss = F.cross_entropy(model(features), labels)
            gradients = torch.autograd.grad(probe_loss, tuple(model.parameters()))
        if any(not torch.isfinite(gradient).all() for gradient in gradients):
            raise FloatingPointError(f"nonfinite {split} diagnostic gradient")
        result[split]["gradient_l2_norm"] = math.sqrt(sum(
            gradient.square().sum().item() for gradient in gradients))
        result[split]["gradients_finite"] = True
    json.dumps(result, allow_nan=False)
    return result


def model_config(arm):
    return {"script": "teacher_family_specificity.py", "arm": arm,
            "input_dim": INPUT_DIM, "classes": CLASSES, "width": WIDTH,
            "rank": RANK, "blocks": BLOCKS, "steps": STEPS,
            "factor_centering": "subtract channel mean per example before optional tanh",
            "fusion": "dense GELU linear branch" if arm == "dense" else ("product" if arm.endswith("product") else "sum / sqrt(2)"),
            "bounded_factors": arm.startswith("tanh_"),
            "branch": "linear then GELU then residual addition" if arm == "dense" else "up projection then GELU then residual addition",
            "optimizer": "AdamW", "learning_rate": 0.006, "weight_decay": 0.001,
            "betas": (0.9, 0.999), "epsilon": 1e-8,
            "batch": "full", "best_observed_updates": list(range(STEPS + 1)),
            "diagnostic_updates": CHECKPOINTS}


def fit(model, data, device, *, artifacts, arm, seed, dataset_path):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.006, weight_decay=0.001)
    diagnostics = {"0": evaluate(model, data)}
    best = BestCheckpoint(scope="minimum training cross entropy over initial state and every post-update state 0..40; no held-out selection")
    losses = [diagnostics["0"]["train"]["loss"]]
    best.observe(model, losses[0], 0)
    train_x, train_y = data["train"]
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
        # The loss must correspond to these weights, not the pre-update weights.
        model.eval()
        with torch.no_grad():
            observed_logits = model(train_x)
            if not torch.isfinite(observed_logits).all():
                raise FloatingPointError(f"nonfinite post-update logits at step {step}")
            observed_loss = F.cross_entropy(observed_logits, train_y).item()
        losses.append(observed_loss)
        best.observe(model, observed_loss, step)
        if step in CHECKPOINTS:
            diagnostics[str(step)] = evaluate(model, data)
    assert best.observations == STEPS + 1
    assert best.observed_steps == list(range(STEPS + 1))
    artifact = best.save(artifacts, f"{seed}-{arm}-best", arm=arm, seed=seed,
                         config=model_config(arm), datasets=[dataset_path])
    # Saving the selected state never replaces the final-update model.
    return {"checkpoints": diagnostics, "training_loss_every_state": losses,
            "best_training_checkpoint": artifact["path"],
            "best_training_selection": artifact["selection"],
            "final_update_metrics": diagnostics[str(STEPS)]}


def aggregate(records):
    metrics, comparisons = {}, {}
    pairs = (("tanh_product", "tanh_additive"), ("raw_product", "raw_additive"),
             ("tanh_product", "dense"), ("raw_product", "dense"),
             ("tanh_additive", "dense"), ("raw_additive", "dense"),
             ("tanh_product", "raw_product"), ("tanh_additive", "raw_additive"))
    for family in FAMILIES:
        rows = [row for row in records if row["family"] == family]
        for arm in ARMS:
            for split in ("train", "dev"):
                for key in ("loss", "accuracy"):
                    metrics[f"{family}_{arm}_{split}_{key}"] = sum(
                        row["arms"][arm]["final_update_metrics"][split][key] for row in rows) / len(rows)
        comparisons[family] = {}
        for left, right in pairs:
            teachers = []
            for teacher in TEACHER_SEEDS:
                repeated = [row for row in rows if row["teacher_seed"] == teacher]
                values = [row["arms"][left]["final_update_metrics"]["dev"]["loss"] -
                          row["arms"][right]["final_update_metrics"]["dev"]["loss"] for row in repeated]
                accuracy = [100 * (row["arms"][left]["final_update_metrics"]["dev"]["accuracy"] -
                                  row["arms"][right]["final_update_metrics"]["dev"]["accuracy"]) for row in repeated]
                teachers.append({"teacher_seed": teacher, "init_loss_differences": values,
                                 "mean_dev_loss_difference": sum(values) / len(values),
                                 "mean_dev_accuracy_difference_pp": sum(accuracy) / len(accuracy)})
            values = [t["mean_dev_loss_difference"] for t in teachers]
            key = f"{left}_minus_{right}"
            comparisons[family][key] = {"teachers": teachers, "mean": sum(values) / len(values),
                                        "min": min(values), "max": max(values)}
            metrics[f"{family}_{key}_dev_loss"] = sum(values) / len(values)
    key = "tanh_product_minus_tanh_additive"
    primary = comparisons["interaction"][key]["mean"] - comparisons["additive"][key]["mean"]
    teacher_primary = [
        {"teacher_seed": teacher,
         "difference": comparisons["interaction"][key]["teachers"][index]["mean_dev_loss_difference"] -
                       comparisons["additive"][key]["teachers"][index]["mean_dev_loss_difference"]}
        for index, teacher in enumerate(TEACHER_SEEDS)]
    values = [row["difference"] for row in teacher_primary]
    comparisons["primary_specificity"] = {"teachers": teacher_primary, "mean": primary,
                                           "min": min(values), "max": max(values)}
    metrics["primary_specificity_dev_loss"] = primary
    return metrics, comparisons


def main():
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    output = Path("/output")
    artifacts = ArtifactWriter("/output")
    records = []
    prepared = []
    script_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for teacher_seed, split_seeds in zip(TEACHER_SEEDS, DATA_SEEDS):
        families, shared = task(teacher_seed, split_seeds)
        for family in FAMILIES:
            cpu_data = families[family]
            arrays = {f"{split}_{name}": tensor for split, pair in cpu_data.items()
                      for name, tensor in zip(("x", "y"), pair)}
            arrays.update(shared)
            if any(not torch.isfinite(tensor).all() for tensor in arrays.values()):
                raise FloatingPointError("nonfinite generated dataset")
            hashes = {name: digest(tensor) for name, tensor in arrays.items()}
            assert hashes["train_latent_z"] != hashes["dev_latent_z"]
            provenance = {"kind": "generated", "generator": "teacher_family_specificity.py:task",
                          "script_sha256": script_sha256,
                          "draw_order": "sign-fixed QR from 16x16 normal; additive 8x4 coefficients; interaction 4x4 coefficients; independent split generators",
                          "teacher_seed": teacher_seed, "data_seeds": dict(zip(("train", "dev"), split_seeds)),
                          "family": family, "downloads": False,
                          "noise": "none; hard argmax labels; lowest-index tie convention",
                          "input": "z iid standard normal; x=z@Q; QR columns multiplied by sign(diag(R)), zero sign replaced by +1; Q shared across families",
                          "teacher": "z[:8]@W_additive or (z[:4]*z[4:8])@W_interaction; no bias or mixed terms",
                          "coefficient_normalization": "row class-center W then divide by sqrt(mean_c sum_j W[j,c]^2); average population class logit variance 1",
                          "split": "1536 train / 512 development; no final test generated",
                          "array_sha256": hashes}
            dataset = artifacts.save_dataset(f"{family}-teacher-{teacher_seed}", arrays, provenance=provenance)
            prepared.append((family, teacher_seed, cpu_data, shared, hashes, provenance, dataset))
    assert len(artifacts.manifest["datasets"]) == 6
    for teacher in TEACHER_SEEDS:
        paired = [entry for entry in prepared if entry[1] == teacher]
        assert len(paired) == 2
        for name in ("train_x", "dev_x", "train_latent_z", "dev_latent_z", "rotation_q"):
            assert paired[0][4][name] == paired[1][4][name]
    for family, teacher_seed, cpu_data, shared, hashes, provenance, dataset in prepared:
        data = {split: tuple(t.to(device) for t in pair) for split, pair in cpu_data.items()}
        for init_seed in INIT_SEEDS:
            models = paired_models(init_seed)
            counts = {arm: sum(p.numel() for p in model.parameters()) for arm, model in models.items()}
            assert counts == {arm: 5716 if arm == "dense" else 5780 for arm in ARMS}
            order = ARMS if len(records) % 2 == 0 else tuple(reversed(ARMS))
            row = {"family": family, "teacher_seed": teacher_seed, "init_seed": init_seed,
                   "dataset": dataset["path"], "data_seeds": provenance["data_seeds"],
                   "array_sha256": hashes, "parameters": counts, "arm_order": order,
                   "arm_order_rule": "forward/reverse alternating by family-teacher-initialization fit replicate in fixed loop order",
                   "class_counts": {split: labels.bincount(minlength=CLASSES).tolist()
                                    for split, (_, labels) in cpu_data.items()},
                   "teacher_logit_summaries": {
                       split: {"mean_class_variance": shared[f"{split}_{family}_teacher_logits"].var(0, unbiased=False).mean().item(),
                               "top_two_margin": magnitude(
                                   shared[f"{split}_{family}_teacher_logits"].topk(2, dim=-1).values.diff(dim=-1).abs())}
                       for split in ("train", "dev")},
                   "initial_state_sha256": {arm: {name: digest(value) for name, value in model.state_dict().items()}
                                            for arm, model in models.items()}, "arms": {}}
            for arm in order:
                torch.manual_seed(init_seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(init_seed)
                seed = f"{family}-teacher-{teacher_seed}-init-{init_seed}"
                row["arms"][arm] = fit(models[arm], data, device, artifacts=artifacts,
                                        arm=arm, seed=seed, dataset_path=dataset["path"])
            records.append(row)
    metrics, comparisons = aggregate(records)
    assert len(artifacts.manifest["datasets"]) == 6
    assert len(artifacts.manifest["checkpoints"]) == 60
    details = {"status": "completed", "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "config": {"teacher_seeds": TEACHER_SEEDS, "data_seeds": DATA_SEEDS, "init_seeds": INIT_SEEDS,
                          "families": FAMILIES, "arms": {arm: model_config(arm) for arm in ARMS},
                          "primary_endpoint": "final-update40 dev CE difference (tanh_product-tanh_additive)_interaction minus (tanh_product-tanh_additive)_additive; negative favors relative interaction specificity",
                          "aggregation": "average two initialization differences within each teacher, then average three teachers per family; subtract family means",
                          "stopping_rule": "exactly40 optimizer updates each fit; fail on nonfinite values; no early stopping or selected dev state"},
               "environment": {"device": str(device), "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
                               "python": platform.python_version(), "torch": str(torch.__version__), "cuda": torch.version.cuda,
                               "cudnn": torch.backends.cudnn.version(), "threads": torch.get_num_threads(),
                               "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                               "cudnn_benchmark": torch.backends.cudnn.benchmark, "cudnn_deterministic": torch.backends.cudnn.deterministic,
                               "matmul_tf32": torch.backends.cuda.matmul.allow_tf32, "cudnn_tf32": torch.backends.cudnn.allow_tf32,
                               "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"]},
               "records": records, "paired_final_dev_loss": comparisons,
               "full_aggregate_metrics": metrics,
               "limitations": "Exploratory development-only screen: three teachers per family, two initialization repeats, no final test, tuning, promotion, novelty, speed or general superiority claim. Shared Q/z couples inputs across families but does not make different labels the same task. Negative primary contrast can coexist with worse product loss in both families; inspect within-family additive and dense comparisons. Equal population logit variance does not imply equal task difficulty. Tanh changes scale, gradients and function class; diagnostics do not identify a causal stabilization mechanism. Historical55-update results have a different budget."}
    # The runner accepts at most 32 metrics. Keep all aggregates in details;
    # this fixed 25-key envelope is independent of observed metric values.
    result_metric_keys = [
        f"{family}_{arm}_dev_{metric}"
        for family in FAMILIES for arm in ARMS for metric in ("loss", "accuracy")]
    result_metric_keys.append("primary_specificity_dev_loss")
    result_metric_keys.extend(
        f"{family}_tanh_product_minus_{control}_dev_loss"
        for family in FAMILIES for control in ("tanh_additive", "dense"))
    assert len(result_metric_keys) == len(set(result_metric_keys)) == 25
    result_metrics = {key: metrics[key] for key in result_metric_keys}
    result = {"metrics": result_metrics, "summary": "Development-only rotated additive/interaction teacher specificity: five arms, 3 teachers x 2 initializations per family, 40 full-batch updates. Final-update dev endpoints independent of saved minimum-training-loss checkpoints (every state0..40 observed). Six exact datasets and60 checkpoints in artifacts.json. No promotion."}
    details_json = json.dumps(details, allow_nan=False, indent=2)
    result_json = json.dumps(result, allow_nan=False, indent=2)
    (output / "details.json").write_text(details_json)
    (output / "result.json").write_text(result_json)
    print(result_json)


if __name__ == "__main__":
    main()
