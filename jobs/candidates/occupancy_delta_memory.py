"""Frozen development-only associative recall screen; execute only in queued worker."""
import copy
import hashlib
import json
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

REGIMES = ("unique", "overwrite")
ARMS = ("dense", "delta_fixed", "delta_constant", "delta_occupancy")
DATASET_SEEDS = (12101, 12102, 12103)
SPLIT_SEEDS = ((13101, 13102), (13111, 13112), (13121, 13122))
INIT_SEEDS = (14101, 14102)
K = V = T = 8
DIM = 16
STEPS = 60
DIAGNOSTIC_STEPS = (0, 20, 40, 60)
SPLIT_SIZE = 512
RHO = 0.9
LOG_EPS = 1e-4


def digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def finite(tensor, label):
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f"nonfinite {label}")


def dense_features(keys, values, query):
    writes = torch.cat((F.one_hot(keys, K), F.one_hot(values, V)), dim=-1)
    return torch.cat((writes.flatten(1), F.one_hot(query, K)), dim=-1).float()


def task(regime, split_seeds):
    """CPU integer RNG; train then dev, rejecting duplicate complete examples."""
    seen = set()
    arrays, rejection_counts = {}, {}
    for split, seed in zip(("train", "dev"), split_seeds):
        rng = torch.Generator().manual_seed(seed)
        examples = []
        rejections = 0
        while len(examples) < SPLIT_SIZE:
            if regime == "unique":
                keys = torch.randperm(K, generator=rng)
            else:
                subset = torch.randperm(K, generator=rng)[:4]
                keys = subset[torch.randint(4, (T,), generator=rng)]
            values = torch.randint(V, (T,), generator=rng)
            distinct = keys.unique(sorted=True)
            query = distinct[torch.randint(len(distinct), (), generator=rng)]
            identity = tuple(keys.tolist() + values.tolist() + [query.item()])
            if identity in seen:
                rejections += 1
                continue
            seen.add(identity)
            last = (keys == query).nonzero().flatten()[-1]
            examples.append((keys, values, query, values[last], last))
        for index, name in enumerate(("keys", "values", "query", "labels", "last_answer_position")):
            arrays[f"{split}_{name}"] = torch.stack([row[index] for row in examples])
        arrays[f"{split}_dense_features"] = dense_features(
            arrays[f"{split}_keys"], arrays[f"{split}_values"], arrays[f"{split}_query"])
        rejection_counts[split] = rejections
    assert len(seen) == 2 * SPLIT_SIZE
    return arrays, rejection_counts


class Dense(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(T * (K + V) + K, 32)
        self.blocks = nn.ModuleList(nn.Linear(32, 32) for _ in range(2))
        self.output = nn.Linear(32, V)

    def forward(self, data, diagnostics=None):
        hidden = F.gelu(self.input(data["dense_features"]))
        for block in self.blocks:
            hidden = hidden + F.gelu(block(hidden))
        logits = self.output(hidden)
        finite(logits, "dense logits")
        return logits


class Delta(nn.Module):
    def __init__(self, arm):
        super().__init__()
        self.arm = arm
        self.key = nn.Embedding(K, DIM)
        self.value = nn.Embedding(V, DIM)
        self.output = nn.Linear(DIM, V)
        if arm != "delta_fixed":
            self.b = nn.Parameter(torch.zeros(()))
        if arm == "delta_occupancy":
            self.a = nn.Parameter(torch.zeros(()))

    def forward(self, data, diagnostics=None):
        keys = self.key(data["keys"])
        keys = keys / keys.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        values = self.value(data["values"])
        query = self.key(data["query"])
        query = query / query.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        memory = keys.new_zeros((keys.shape[0], DIM, DIM))
        covariance = torch.zeros_like(memory)
        beta_trace, occupancy_trace = [], []
        for position in range(T):
            key, value = keys[:, position], values[:, position]
            prior = torch.bmm(covariance, key.unsqueeze(-1)).squeeze(-1)
            # The PSD quadratic form can be slightly negative from roundoff.
            occupancy = (key * prior).sum(-1).clamp_min(0)
            if self.arm == "delta_fixed":
                beta = occupancy.new_full(occupancy.shape, 0.5)
            elif self.arm == "delta_constant":
                beta = self.b.sigmoid().expand_as(occupancy)
            else:
                beta = (self.b + self.a * (LOG_EPS + occupancy).log()).sigmoid()
            prediction = torch.bmm(memory, key.unsqueeze(-1)).squeeze(-1)
            memory = memory + beta[:, None, None] * (value - prediction).unsqueeze(-1) * key.unsqueeze(1)
            covariance = RHO * covariance + (1 - RHO) * key.unsqueeze(-1) * key.unsqueeze(1)
            if diagnostics is not None:
                beta_trace.append(beta)
                occupancy_trace.append(occupancy)
        read = torch.bmm(memory, query.unsqueeze(-1)).squeeze(-1)
        logits = self.output(read)
        finite(memory, "delta value memory")
        finite(covariance, "delta key covariance")
        finite(logits, "delta logits")
        if diagnostics is not None:
            diagnostics.update({"beta": distribution(torch.stack(beta_trace, dim=1)),
                                "occupancy": distribution(torch.stack(occupancy_trace, dim=1)),
                                "memory_rms": memory.square().mean().sqrt().item(),
                                "covariance_rms": covariance.square().mean().sqrt().item(),
                                "a": self.a.item() if hasattr(self, "a") else None,
                                "b": self.b.item() if hasattr(self, "b") else None})
        return logits


def distribution(value):
    finite(value, "diagnostic distribution")
    return {"mean": value.mean().item(), "min": value.min().item(), "max": value.max().item(),
            "quantiles": torch.quantile(value.flatten(), value.new_tensor([0., .25, .5, .75, 1.])).tolist()}


def paired_models(seed):
    torch.manual_seed(seed)
    models = {"dense": Dense(), **{arm: Delta(arm) for arm in ARMS[1:]}}
    reference = models["delta_fixed"].state_dict()
    for arm in ARMS[2:]:
        models[arm].load_state_dict(copy.deepcopy(reference), strict=False)
        assert all(torch.equal(value, models[arm].state_dict()[name]) for name, value in reference.items())
        assert models[arm].b.item() == 0
    assert models["delta_occupancy"].a.item() == 0
    return models


@torch.no_grad()
def evaluate(model, data):
    model.eval()
    result = {}
    for split, batch in data.items():
        diagnostics = {}
        logits = model(batch, diagnostics)
        loss = F.cross_entropy(logits, batch["labels"])
        finite(loss, f"{split} loss")
        result[split] = {"loss": loss.item(),
                         "accuracy": (logits.argmax(-1) == batch["labels"]).float().mean().item(),
                         "logit_rms": logits.square().mean().sqrt().item(),
                         "diagnostics": diagnostics}
    return result


def model_config(arm):
    return {"script": "occupancy_delta_memory.py", "arm": arm, "keys": K, "values": V,
            "writes": T, "memory_dimension": DIM, "dense_hidden": 32, "dense_blocks": 2,
            "dense_input": "ordered flattened per-write concatenated key/value one-hots, then query one-hot",
            "memory_update": "M += beta * (v - M k) k^T; zero M; normalized keys/query, norm floor1e-6",
            "occupancy_update": "prior k^T C k; C=.9*C+.1*k*k^T after M; zero C; no detach; quadratic roundoff clamped >=0",
            "gate": {"delta_fixed": "0.5", "delta_constant": "sigmoid(b), b initially0",
                     "delta_occupancy": "sigmoid(b+a*log(1e-4+occupancy)), a=b initially0"}.get(arm),
            "rho": RHO, "log_epsilon": LOG_EPS, "optimizer": "AdamW", "learning_rate": 0.003,
            "weight_decay": 0.001, "betas": [0.9, 0.999], "epsilon": 1e-8, "steps": STEPS,
            "batch": "full", "diagnostic_steps": DIAGNOSTIC_STEPS,
            "selection_observed_steps": list(range(STEPS + 1)), "query_updates_state": False,
            "per_example_persistent_memory_elements": 0 if arm == "dense" else 2 * DIM * DIM,
            "per_example_persistent_memory_bytes_float32": 0 if arm == "dense" else 2 * DIM * DIM * 4}


def fit(model, data, device, *, writer, arm, fit_id, dataset):
    model.to(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    fit_started = time.monotonic()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.001)
    diagnostics = {"0": evaluate(model, data)}
    losses = [diagnostics["0"]["train"]["loss"]]
    best = BestCheckpoint(scope="minimum full-training CE at actual states0..60; earliest tied minimum; no held-out selection")
    best.observe(model, losses[0], 0)
    for step in range(1, STEPS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(data["train"]), data["train"]["labels"])
        finite(loss, f"training loss at update{step}")
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                finite(parameter.grad, f"gradient {name} update{step}")
        optimizer.step()
        model.eval()
        with torch.no_grad():
            observed = F.cross_entropy(model(data["train"]), data["train"]["labels"])
        finite(observed, f"post-update training CE{step}")
        losses.append(observed.item())
        best.observe(model, observed.item(), step)
        if step in DIAGNOSTIC_STEPS:
            diagnostics[str(step)] = evaluate(model, data)
    assert best.observations == STEPS + 1
    assert best.observed_steps == list(range(STEPS + 1))
    config = {**model_config(arm), "fit_id": fit_id, "dataset_sha256": dataset["sha256"]}
    saved = best.save(writer, f"{fit_id}-{arm}-best", arm=arm, seed=fit_id,
                      config=config, datasets=[dataset["path"]])
    assert saved["selection"]["value"] == min(losses)
    assert saved["selection"]["step"] == losses.index(min(losses))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.monotonic() - fit_started
    return {"diagnostic_states": diagnostics, "training_loss_every_state": losses,
            "best_training_checkpoint": saved["path"], "best_training_selection": saved["selection"],
            "final_update": STEPS, "final_update_metrics": diagnostics[str(STEPS)],
            "gradients_finite_every_update": True, "parameters": sum(p.numel() for p in model.parameters()),
            "elapsed_seconds_including_diagnostics_and_checkpoint": elapsed}


def aggregate(records):
    metrics, comparisons = {}, {}
    pairs = (("delta_occupancy", "delta_constant"), ("delta_occupancy", "delta_fixed"),
             ("delta_constant", "delta_fixed"), ("delta_occupancy", "dense"))
    for regime in REGIMES:
        rows = [row for row in records if row["regime"] == regime]
        for arm in ARMS:
            for metric in ("loss", "accuracy"):
                metrics[f"{regime}_{arm}_dev_{metric}"] = sum(
                    row["arms"][arm]["final_update_metrics"]["dev"][metric] for row in rows) / len(rows)
        comparisons[regime] = {}
        for left, right in pairs:
            replicates = []
            for seed in DATASET_SEEDS:
                repeated = [row for row in rows if row["dataset_seed"] == seed]
                values = [row["arms"][left]["final_update_metrics"]["dev"]["loss"] -
                          row["arms"][right]["final_update_metrics"]["dev"]["loss"] for row in repeated]
                accuracies = [100 * (row["arms"][left]["final_update_metrics"]["dev"]["accuracy"] -
                                    row["arms"][right]["final_update_metrics"]["dev"]["accuracy"]) for row in repeated]
                replicates.append({"dataset_seed": seed, "init_loss_differences": values,
                                   "mean_dev_loss_difference": sum(values) / len(values),
                                   "mean_dev_accuracy_difference_pp": sum(accuracies) / len(accuracies)})
            means = [row["mean_dev_loss_difference"] for row in replicates]
            accuracy = [row["mean_dev_accuracy_difference_pp"] for row in replicates]
            comparisons[regime][f"{left}_minus_{right}"] = {
                "replicates": replicates, "mean": sum(means) / len(means), "min": min(means), "max": max(means),
                "mean_accuracy_difference_pp": sum(accuracy) / len(accuracy),
                "min_accuracy_difference_pp": min(accuracy), "max_accuracy_difference_pp": max(accuracy)}
        for control in ("delta_constant", "dense"):
            metrics[f"{regime}_occupancy_minus_{control}_dev_loss"] = comparisons[regime][f"delta_occupancy_minus_{control}"]["mean"]
    assert len(metrics) == 20
    return metrics, comparisons


def main():
    job_started = time.monotonic()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    writer = ArtifactWriter("/output")
    source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    prepared, records = [], []
    for regime_index, regime in enumerate(REGIMES):
        for dataset_seed, split_seeds in zip(DATASET_SEEDS, SPLIT_SEEDS):
            streams = tuple(seed + 100000 * regime_index for seed in split_seeds)
            arrays, rejections = task(regime, streams)
            hashes = {name: digest(value) for name, value in arrays.items()}
            for value in arrays.values():
                finite(value, "dataset tensor")
            provenance = {"kind": "generated", "generator": "occupancy_delta_memory.py:task",
                          "script_sha256": source_sha, "dataset_replicate_id_seed": dataset_seed,
                          "base_split_seeds": dict(zip(("train", "dev"), split_seeds)),
                          "actual_split_seeds": dict(zip(("train", "dev"), streams)),
                          "stream_derivation": "base split seed +100000*regime_index; unique0 overwrite1",
                          "regime": regime, "keys": K, "values": V, "writes": T,
                          "draw_order": "train then dev separate CPU generators; each candidate keys (unique randperm8, overwrite randperm8[:4] then randint4[8]), randint8[8] values, randint among sorted distinct keys for query; reject duplicate complete example across both splits",
                          "label_rule": "value at last occurrence of query key; no answer token supplied",
                          "split": {"train": SPLIT_SIZE, "dev": SPLIT_SIZE, "test": "not generated"},
                          "duplicate_complete_example_rejections": rejections,
                          "array_sha256": hashes, "downloads": False,
                          "generator_software": {"torch": str(torch.__version__), "device": "cpu"}}
            dataset = writer.save_dataset(f"{regime}-dataset-{dataset_seed}", arrays, provenance=provenance)
            prepared.append((regime, dataset_seed, streams, arrays, dataset, hashes))
    assert len(writer.manifest["datasets"]) == 6
    assert len(writer.manifest["checkpoints"]) == 0
    for regime, dataset_seed, streams, arrays, dataset, hashes in prepared:
        data = {split: {name: arrays[f"{split}_{name}"].to(device) for name in
                       ("keys", "values", "query", "labels", "last_answer_position", "dense_features")}
                for split in ("train", "dev")}
        for init_seed in INIT_SEEDS:
            models = paired_models(init_seed)
            counts = {arm: sum(p.numel() for p in model.parameters()) for arm, model in models.items()}
            assert counts == {"dense": 6760, "delta_fixed": 392, "delta_constant": 393, "delta_occupancy": 394}
            initial_hashes = {arm: {name: digest(value) for name, value in model.state_dict().items()} for arm, model in models.items()}
            for model in models.values():
                model.to(device)
            with torch.no_grad():
                for split in ("train", "dev"):
                    reference = models["delta_fixed"](data[split])
                    for arm in ARMS[2:]:
                        assert torch.equal(reference, models[arm](data[split])), "initial delta logits differ"
            order = ARMS if len(records) % 2 == 0 else tuple(reversed(ARMS))
            fit_id = f"{regime}-data-{dataset_seed}-init-{init_seed}"
            row = {"regime": regime, "dataset_seed": dataset_seed, "init_seed": init_seed,
                   "actual_split_seeds": dict(zip(("train", "dev"), streams)),
                   "dataset": dataset["path"], "dataset_sha256": dataset["sha256"], "array_sha256": hashes,
                   "parameters": counts, "initial_state_sha256": initial_hashes,
                   "initial_delta_logits_exactly_equal": True, "arm_order": order,
                   "arm_order_rule": "forward/reverse alternating regime/dataset/init loop index",
                   "distributions": {split: {
                       "label_counts": arrays[f"{split}_labels"].bincount(minlength=V).tolist(),
                       "query_counts": arrays[f"{split}_query"].bincount(minlength=K).tolist(),
                       "distinct_key_counts": torch.tensor([len(x.unique()) for x in arrays[f"{split}_keys"]]).bincount(minlength=K + 1).tolist(),
                       "last_answer_position_counts": arrays[f"{split}_last_answer_position"].bincount(minlength=T).tolist(),
                       "last_write_only_heuristic_accuracy": (arrays[f"{split}_values"][:, -1] == arrays[f"{split}_labels"]).float().mean().item()}
                       for split in ("train", "dev")}, "arms": {}}
            for arm in order:
                torch.manual_seed(init_seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(init_seed)
                row["arms"][arm] = fit(models[arm], data, device, writer=writer, arm=arm,
                                        fit_id=fit_id, dataset=dataset)
            records.append(row)
    assert len(records) == 12
    assert len(writer.manifest["datasets"]) == 6
    assert len(writer.manifest["checkpoints"]) == 48
    metrics, comparisons = aggregate(records)
    primary = comparisons["overwrite"]["delta_occupancy_minus_delta_constant"]
    unique = comparisons["unique"]["delta_occupancy_minus_delta_constant"]
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    details = {"status": "completed", "script_sha256": source_sha,
               "elapsed_seconds_through_all_fits": time.monotonic() - job_started,
               "config": {"regimes": REGIMES, "dataset_seeds": DATASET_SEEDS, "split_seeds": SPLIT_SEEDS,
                          "init_seeds": INIT_SEEDS, "arms": {arm: model_config(arm) for arm in ARMS},
                          "primary_endpoint": "final60 overwrite dev CE occupancy-minus-constant; average initializations within dataset then three datasets equally",
                          "screening_rule": "primary mean<0 and all three dataset means<0, unique mean<=0.01; no significance/promotion claim",
                          "stopping_rule": "exactly60 full-batch updates, no tuning or held-out checkpoint selection; abort incomplete/nonfinite jobs",
                          "chance_accuracy": 1 / V},
               "environment": {"device": str(device), "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
                               "python": platform.python_version(), "torch": str(torch.__version__), "cuda": torch.version.cuda,
                               "cudnn": torch.backends.cudnn.version(), "threads": torch.get_num_threads(),
                               "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                               "cudnn_benchmark": torch.backends.cudnn.benchmark, "cudnn_deterministic": torch.backends.cudnn.deterministic,
                               "matmul_tf32": torch.backends.cuda.matmul.allow_tf32, "cudnn_tf32": torch.backends.cudnn.allow_tf32,
                               "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"]},
               "records": records, "paired_final_dev_contrasts": comparisons,
               "screening_criterion_met": primary["mean"] < 0 and primary["max"] < 0 and unique["mean"] <= 0.01,
               "limitations": "Development-only fixed-budget synthetic recall screen; no final test or promotion. Initialization repeats cluster within three datasets. Dense is generous in parameters but not compute matched; no competitive recurrent/attention baseline. Occupancy has one more scalar than constant, two more than fixed; gate can change learned scale. Regimes differ in repetition and difficulty. No causal collision protection, global novelty, efficiency, convergence, real-world or cross-task superiority claim. Favorable screen requires future history-shuffled/token-only controls, competitive sequence baseline and fresh confirmation. Failure stops without outcome-driven tuning."}
    summary = "Development-only associative recall: unique/overwrite, four arms, 3 datasets x 2 initializations, 60 full-batch updates (48 fits). Final60 dev endpoints separate from every-state0..60 minimum-training-CE checkpoints. Six exact datasets and48 checkpoints. No test or promotion."
    assert len(metrics) <= 32 and len(summary) <= 2000
    result = {"metrics": metrics, "summary": summary}
    output = Path("/output")
    (output / "details.json").write_text(json.dumps(details, allow_nan=False, indent=2))
    result_json = json.dumps(result, allow_nan=False, indent=2)
    (output / "result.json").write_text(result_json)
    print(result_json)


if __name__ == "__main__":
    main()
