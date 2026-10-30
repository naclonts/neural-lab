"""Frozen Fourier-feature development screen; queued worker execution only."""
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

REGIMES = ("periodic", "affine")
MATCHED_ARMS = ("fourier", "relu", "linear")
ARMS = MATCHED_ARMS + ("raw_dense",)
DATASET_SEEDS = (25101, 25102, 25103)
SPLIT_SEEDS = ((26101, 26102), (26111, 26112), (26121, 26122))
INIT_SEEDS = (27101, 27102)
STEPS = 120
SIZE = 512
DIAGNOSTIC_STEPS = (0, 40, 80, 120)


def digest(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def finite(value, name):
    if not torch.isfinite(value).all():
        raise FloatingPointError("nonfinite " + name)


def task(regime, seeds):
    arrays, rejections, seen = {}, {}, set()
    for split, seed in zip(("train", "dev"), seeds):
        rng = torch.Generator().manual_seed(seed)
        rows, rejected = [], 0
        while len(rows) < SIZE:
            x = 2 * torch.rand((2,), generator=rng, dtype=torch.float32) - 1
            identity = tuple(x.tolist())
            if identity in seen:
                rejected += 1
                continue
            seen.add(identity)
            rows.append(x)
        x = torch.stack(rows)
        score = (torch.sin(2 * torch.pi * (1.7 * x[:, 0] + .8 * x[:, 1]) + .3)
                 + .65 * torch.cos(2 * torch.pi * (-.6 * x[:, 0] + 2.3 * x[:, 1]) - .7)) if regime == "periodic" else x[:, 0] + .6 * x[:, 1] - .1
        arrays[split + "_inputs"] = x
        arrays[split + "_teacher_score"] = score
        arrays[split + "_labels"] = (score > 0).long()
        rejections[split] = rejected
    assert len(seen) == 2 * SIZE
    return arrays, rejections


def prepare_features(arrays, init_index):
    result = dict(arrays)
    generator = torch.Generator().manual_seed(28100 + init_index)
    projection = torch.randn((16, 2), generator=generator, dtype=torch.float32) * 1.5
    result["projection_B"] = projection
    for split in ("train", "dev"):
        z = 2 * torch.pi * (arrays[split + "_inputs"] @ projection.T)
        result[split + "_z"] = z
        for arm in MATCHED_ARMS:
            raw = {"fourier": lambda: torch.cat((z.sin(), z.cos()), -1),
                   "relu": lambda: torch.cat((z.relu(), (-z).relu()), -1),
                   "linear": lambda: torch.cat((z, -z), -1)}[arm]()
            result[split + "_" + arm + "_raw"] = raw
    for arm in MATCHED_ARMS:
        raw = result["train_" + arm + "_raw"]
        mean = raw.mean(0)
        std = raw.std(0, unbiased=False).clamp_min(1e-6)
        result[arm + "_train_mean"] = mean
        result[arm + "_train_std"] = std
        for split in ("train", "dev"):
            result[split + "_" + arm + "_features"] = (result[split + "_" + arm + "_raw"] - mean) / std
    return result


class RawDense(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(2, 48)
        self.blocks = nn.ModuleList([nn.Linear(48, 48) for _ in range(2)])
        self.output = nn.Linear(48, 2)

    def forward(self, x):
        h = F.gelu(self.input(x))
        for block in self.blocks:
            h = h + F.gelu(block(h))
        return self.output(h)


def head(raw=False):
    return RawDense() if raw else nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 2))


@torch.no_grad()
def evaluate(model, batches, known_train_loss=None):
    model.eval()
    result = {}
    for split, (features, labels) in batches.items():
        logits = model(features)
        loss = logits.new_tensor(known_train_loss) if split == "train" and known_train_loss is not None else F.cross_entropy(logits, labels)
        finite(logits, split + " logits")
        finite(loss, split + " loss")
        result[split] = {"loss": loss.item(), "accuracy": (logits.argmax(-1) == labels).float().mean().item(),
                         "logit_rms": logits.square().mean().sqrt().item()}
    return result


def config(arm, source_sha):
    return {"script": "fourier_feature_classifier.py", "source_sha256": source_sha, "arm": arm,
            "feature_map": {"fourier": "cat(sin(z),cos(z))", "relu": "cat(relu(z),relu(-z))", "linear": "cat(z,-z)", "raw_dense": "raw unstandardized input coordinates"}[arm],
            "projection": "none" if arm == "raw_dense" else "z=2*pi*x@B.T; B16x2 iid Normal(0,1.5^2), fixed independent CPU generator",
            "standardization": "none; raw input" if arm == "raw_dense" else "per feature training mean/population std floor1e-6; no labels/dev statistics",
            "external_fixed_state": "none" if arm == "raw_dense" else "projection_B and arm_train_mean/arm_train_std in checkpoint-linked dataset path/hash; combine these with head weights for inference",
            "head": "Linear2,48 GELU; twice residual Linear48,48 GELU; Linear48,2" if arm == "raw_dense" else "Linear(32,32),ReLU,Linear(32,2)", "trainable_parameters": 4946 if arm == "raw_dense" else 1122,
            "optimizer": "AdamW", "learning_rate": .003, "weight_decay": .001,
            "betas": [.9, .999], "epsilon": 1e-8, "steps": STEPS, "batch": "full",
            "device": "cpu", "threads": 4, "selection_observed_steps": list(range(STEPS + 1))}


def fit(model, batches, writer, arm, fit_id, dataset, source_sha, init_seed, projection_seed):
    start = time.monotonic()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.001, betas=(.9, .999), eps=1e-8)
    x, y = batches["train"]
    loss = F.cross_entropy(model(x), y)
    finite(loss, "state0 loss")
    diagnostics = {"0": evaluate(model, batches, known_train_loss=loss.item())}
    losses = [loss.item()]
    best = BestCheckpoint(scope="minimum full-training CE at actual states0..120; earliest tied minimum; no held-out selection")
    best.observe(model, losses[0], 0)
    for step in range(1, STEPS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        finite(loss, "pre-update loss")
        loss.backward()
        for name, parameter in model.named_parameters():
            finite(parameter.grad, "gradient " + name)
        optimizer.step()
        for name, parameter in model.named_parameters():
            finite(parameter, "post-update parameter " + name)
        model.eval()
        observed = F.cross_entropy(model(x), y)
        finite(observed, "post-update loss")
        losses.append(observed.item())
        best.observe(model, observed.item(), step)
        loss = observed
        if step in DIAGNOSTIC_STEPS:
            diagnostics[str(step)] = evaluate(model, batches, known_train_loss=observed.item())
    assert best.observations == STEPS + 1 and best.observed_steps == list(range(STEPS + 1))
    configuration = {**config(arm, source_sha), "fit_id": fit_id, "init_seed": init_seed,
                     "projection_seed": projection_seed, "dataset_sha256": dataset["sha256"]}
    checkpoint = best.save(writer, fit_id + "-" + arm + "-best", arm=arm, seed=init_seed,
                           config=configuration, datasets=[dataset["path"]])
    assert checkpoint["selection"]["value"] == min(losses)
    assert checkpoint["selection"]["step"] == losses.index(min(losses))
    return {"training_loss_every_state": losses, "diagnostic_states": diagnostics,
            "best_training_checkpoint": checkpoint["path"], "best_training_selection": checkpoint["selection"],
            "final_update": STEPS, "final_update_metrics": diagnostics[str(STEPS)],
            "elapsed_seconds": time.monotonic() - start, "gradients_finite_every_update": True}


def aggregate(records):
    metrics, comparisons = {}, {}
    for regime in REGIMES:
        rows = [r for r in records if r["regime"] == regime]
        for arm in ARMS:
            for endpoint in ("loss", "accuracy"):
                metrics[f"{regime}_{arm}_dev_{endpoint}"] = sum(r["arms"][arm]["final_update_metrics"]["dev"][endpoint] for r in rows) / len(rows)
        comparisons[regime] = {}
        for control in ARMS[1:]:
            replicates = []
            for dataset_seed in DATASET_SEEDS:
                paired = [r for r in rows if r["dataset_seed"] == dataset_seed]
                differences = [r["arms"]["fourier"]["final_update_metrics"]["dev"]["loss"] - r["arms"][control]["final_update_metrics"]["dev"]["loss"] for r in paired]
                replicates.append({"dataset_seed": dataset_seed, "init_loss_differences": differences,
                                   "mean_dev_loss_difference": sum(differences) / len(differences)})
            means = [r["mean_dev_loss_difference"] for r in replicates]
            comparisons[regime][control] = {"replicates": replicates, "mean": sum(means) / len(means), "min": min(means), "max": max(means)}
            metrics[f"{regime}_fourier_minus_{control}_dev_loss"] = sum(means) / len(means)
    passed = all(comparisons["periodic"][arm]["mean"] <= -.02 and comparisons["periodic"][arm]["max"] < 0
                 and comparisons["affine"][arm]["mean"] <= .02 for arm in MATCHED_ARMS[1:])
    return metrics, comparisons, passed


def main():
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    writer = ArtifactWriter("/output")
    source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    prepared = []
    for regime_index, regime in enumerate(REGIMES):
        for dataset_seed, base_seeds in zip(DATASET_SEEDS, SPLIT_SEEDS):
            seeds = tuple(s + 100000 * regime_index for s in base_seeds)
            arrays, rejections = task(regime, seeds)
            for init_index, init_seed in enumerate(INIT_SEEDS):
                payload = prepare_features(arrays, init_index)
                for name, value in payload.items():
                    finite(value, name)
                hashes = {name: digest(value) for name, value in payload.items()}
                provenance = {"kind": "generated", "generator": "fourier_feature_classifier.py:task/prepare_features",
                              "script_sha256": source_sha, "dataset_replicate_id_seed": dataset_seed,
                              "base_split_seeds": dict(zip(("train", "dev"), base_seeds)),
                              "actual_split_seeds": dict(zip(("train", "dev"), seeds)),
                              "stream_derivation": "base+100000*regime_index; periodic0 affine1",
                              "projection_seed": 28100 + init_index, "init_seed": init_seed, "regime": regime,
                              "draw_order": "train then dev separate CPU generators; one float32 torch.rand(2) input candidate at a time, x=2*u-1; reject exact complete input duplicates across splits",
                              "teacher": "periodic: sin(2*pi*(1.7*x+.8*y)+.3)+.65*cos(2*pi*(-.6*x+2.3*y)-.7); affine: x+.6*y-.1; label=(score>0).long",
                              "split": {"train": SIZE, "dev": SIZE, "test": "not generated"},
                              "duplicate_input_rejections": rejections, "array_sha256": hashes,
                              "feature_preparation": "fixed independent B; raw z/maps saved; training-only mean/population std floor1e-6; transformed train/dev saved",
                              "downloads": False, "generator_software": {"torch": str(torch.__version__), "device": "cpu"}}
                dataset = writer.save_dataset(f"{regime}-data-{dataset_seed}-init-{init_seed}", payload, provenance=provenance)
                prepared.append((regime, dataset_seed, seeds, init_seed, init_index, payload, dataset, hashes))
    assert len(writer.manifest["datasets"]) == 12 and not writer.manifest["checkpoints"]
    records = []
    for regime, dataset_seed, seeds, init_seed, init_index, payload, dataset, hashes in prepared:
        torch.manual_seed(init_seed)
        reference = head()
        models = {arm: copy.deepcopy(reference) for arm in MATCHED_ARMS}
        torch.manual_seed(init_seed)
        models["raw_dense"] = head(raw=True)
        assert {a: sum(p.numel() for p in m.parameters()) for a,m in models.items()} == {a: 4946 if a == "raw_dense" else 1122 for a in ARMS}
        initial_hashes = {name: digest(value) for name, value in reference.state_dict().items()}
        assert all(all(torch.equal(value, models[arm].state_dict()[name]) for name, value in reference.state_dict().items()) for arm in MATCHED_ARMS)
        order = ARMS if len(records) % 2 == 0 else tuple(reversed(ARMS))
        fit_id = f"{regime}-data-{dataset_seed}-init-{init_seed}"
        row = {"regime": regime, "dataset_seed": dataset_seed, "init_seed": init_seed, "projection_seed": 28100 + init_index,
               "actual_split_seeds": dict(zip(("train", "dev"), seeds)), "dataset": dataset["path"],
               "dataset_sha256": dataset["sha256"], "array_sha256": hashes, "initial_head_state_sha256": initial_hashes,
               "projection_frequency_norms": payload["projection_B"].norm(dim=1).tolist(),
               "feature_diagnostics": {a:{"train_feature_mean":payload["train_"+a+"_features"].mean(0).tolist(),
                   "train_feature_std":payload["train_"+a+"_features"].std(0,unbiased=False).tolist(),
                   "train_feature_min":payload["train_"+a+"_features"].min(0).values.tolist(),
                   "train_feature_max":payload["train_"+a+"_features"].max(0).values.tolist()} for a in MATCHED_ARMS},
               "initial_head_exactly_shared_matched_arms": True, "raw_head_initial_state_sha256": {n:digest(v) for n,v in models["raw_dense"].state_dict().items()},
               "fixed_projection_elements": 32, "standardization_elements_per_matched_arm": 64,
               "trainable_parameters": {a:4946 if a == "raw_dense" else 1122 for a in ARMS}, "arm_order": order, "arms": {},
               "label_counts": {s: payload[s + "_labels"].bincount(minlength=2).tolist() for s in ("train", "dev")}}
        for arm in order:
            torch.manual_seed(init_seed)
            batches = {s: (payload[s + "_inputs"] if arm == "raw_dense" else payload[s + "_" + arm + "_features"], payload[s + "_labels"]) for s in ("train", "dev")}
            row["arms"][arm] = fit(models[arm], batches, writer, arm, fit_id, dataset, source_sha, init_seed, 28100 + init_index)
        records.append(row)
    assert len(records) == 12 and len(writer.manifest["checkpoints"]) == 48
    metrics, comparisons, passed = aggregate(records)
    details = {"status": "completed", "script_sha256": source_sha, "elapsed_seconds": time.monotonic() - started,
               "config": {"regimes": REGIMES, "dataset_seeds": DATASET_SEEDS, "split_seeds": SPLIT_SEEDS, "init_seeds": INIT_SEEDS,
                          "arms": {arm: config(arm, source_sha) for arm in ARMS},
                          "primary_endpoint": "final120 periodic dev CE Fourier-minus-matched relu/linear; mean2inits within dataset then mean3datasets; raw_dense contextual only",
                          "screening_rule": "both controls periodic mean<=-.02 and all3dataset means<0; affine mean<=.02 both",
                          "stopping_rule": "exactly120 fullbatch updates; no tuning, test or heldout selection"},
               "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu",
                               "threads": torch.get_num_threads(), "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                               "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"]},
               "records": records, "paired_final_dev_contrasts": comparisons, "screening_criterion_met": passed,
               "limitations": "Development-only synthetic in-distribution screening with deliberately periodic and affine teachers; no test/OOD, significance, novelty, promotion, causal mechanism or broad superiority claim. Three input-dataset clusters and two paired head/projection repeats. Standardization matches marginal feature scale, not joint geometry or optimization; nonlinear ReLU head means linear features are not a linear classifier. Twelve payloads contain six input datasets duplicated across two fixed projection realizations. Raw dense contextual baseline reuses the existing residual GELU family adapted to2 inputs/2classes,4946 parameters versus1122; its initialization shape differs and is seeded independently. Equal updates/parameters do not equal compute. Failure stops without tuning."}
    summary = "Frozen Fourier-feature classification screen: periodic/affine teachers, Fourier/ReLU/linear fixed maps, shared 1122-parameter matched heads plus4946-parameter raw dense residual GELU baseline, 3 datasets x2 paired projection/head initializations,48 fits of120 CPU fullbatch updates. Final120 dev separate from121-state minimum-training-CE checkpoints. Exact inputs/maps/statistics saved; no test, tuning or promotion."
    assert 1 <= len(metrics) <= 32 and len(summary) <= 2000
    output = Path("/output")
    (output / "details.json").write_text(json.dumps(details, allow_nan=False, indent=2))
    result = json.dumps({"metrics": metrics, "summary": summary}, allow_nan=False, indent=2)
    (output / "result.json").write_text(result)
    print(result)


if __name__ == "__main__":
    main()
