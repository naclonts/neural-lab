"""Portable, pickle-free experiment datasets, weights, and provenance manifests.

Use inside the training container. Reading these files never executes model code;
reconstruct the reviewed model architecture before loading its state dictionary.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re


def _slug(label: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_-]+", "-", label).strip("-")[:100]
    if not value:
        raise ValueError("Artifact label must contain a filename character")
    return value


def _json(path: Path, value) -> None:
    path.write_text(json.dumps(value, allow_nan=False, indent=2))


class ArtifactWriter:
    """Save numeric arrays and state dictionaries with a discoverable manifest."""

    def __init__(self, output):
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.manifest = {"version": 1, "datasets": [], "checkpoints": []}
        previous = self.output / "artifacts.json"
        if previous.exists():
            value = json.loads(previous.read_text())
            if value.get("version") != 1 or any(not isinstance(value.get(key), list) for key in ("datasets", "checkpoints")):
                raise ValueError("Existing artifact manifest has an unsupported schema")
            self.manifest = value

    def _record(self, category, path, label, **metadata):
        entry = {"label": label, "path": path.relative_to(self.output).as_posix(),
                 "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                 "bytes": path.stat().st_size, **metadata}
        self.manifest[category].append(entry)
        temporary = self.output / ".artifacts.json.tmp"
        _json(temporary, self.manifest)
        temporary.replace(self.output / "artifacts.json")
        return entry

    def register_file(self, category, path, label, **metadata):
        """Register an already-saved dataset/checkpoint without interpreting it."""
        path = Path(path).resolve()
        if category not in ("datasets", "checkpoints") or not path.is_relative_to(self.output.resolve()) or not path.is_file():
            raise ValueError("Artifact must be a file within this job's output")
        return self._record(category, path, label, **metadata)

    def save_dataset(self, label, arrays, *, provenance):
        """Save actual split tensors as NPZ, loadable with allow_pickle=False."""
        import numpy as np

        path = self.output / "datasets" / (_slug(label) + ".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        numeric = {}
        for name, value in arrays.items():
            array = value.detach().cpu().contiguous().numpy() if hasattr(value, "detach") else np.asarray(value)
            if array.dtype.kind not in "buifc":
                raise ValueError("Dataset arrays must be numeric")
            numeric[name] = array
        np.savez_compressed(path, **numeric)
        schema = {name: {"shape": list(array.shape), "dtype": str(array.dtype)} for name, array in numeric.items()}
        return self._record("datasets", path, label, format="npz", source=provenance, arrays=schema)

    def save_checkpoint(self, label, state, *, arm, seed, selection, config=None, datasets=()):
        """Save an architecture-independent typed JSON state dictionary."""
        path = self.output / "checkpoints" / (_slug(label) + ".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        state = state.state_dict() if hasattr(state, "state_dict") else state
        tensors = {name: {"dtype": str(tensor.dtype).removeprefix("torch."),
                          "shape": list(tensor.shape), "data": tensor.detach().cpu().tolist()}
                   for name, tensor in state.items()}
        _json(path, {"format": "neural-lab-state-dict-v1", "tensors": tensors,
                     "arm": arm, "seed": seed, "selection": selection, "config": config or {}})
        return self._record("checkpoints", path, label, format="json-state-dict", arm=arm,
                            seed=seed, selection=selection, config=config or {}, datasets=list(datasets))


class BestCheckpoint:
    """Track weights with minimum training loss, without selecting on held-out data."""

    def __init__(self, scope="best observed training loss; held-out data not used for selection"):
        self.state = None
        self.value = math.inf
        self.step = None
        self.observations = 0
        self.observed_steps = []
        self.scope = scope

    def observe(self, model, value: float, step: int):
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("Checkpoint selection requires a finite loss")
        self.observations += 1
        self.observed_steps.append(int(step))
        if value < self.value:
            self.value, self.step = value, int(step)
            self.state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}

    def save(self, writer, label, *, arm, seed, config=None, datasets=()):
        if self.state is None:
            raise ValueError("No checkpoint observations")
        return writer.save_checkpoint(label, self.state, arm=arm, seed=seed,
            selection={"metric": "training_cross_entropy", "mode": "min", "value": self.value,
                       "step": self.step, "observations": self.observations,
                       "observed_steps": self.observed_steps, "scope": self.scope},
            config=config, datasets=datasets)


def load_json_state(path):
    """Load tensors only; the caller supplies the reviewed model architecture."""
    import torch

    value = json.loads(Path(path).read_text())
    if value.get("format") != "neural-lab-state-dict-v1":
        raise ValueError("Unsupported checkpoint format")
    supported = {str(dtype).removeprefix("torch."): dtype for dtype in (
        torch.float16, torch.float32, torch.float64, torch.int8, torch.int16,
        torch.int32, torch.int64, torch.uint8, torch.bool, torch.bfloat16)}
    state = {}
    for name, tensor in value["tensors"].items():
        dtype = supported.get(tensor["dtype"])
        if dtype is None:
            raise ValueError("Unsupported tensor dtype")
        result = torch.tensor(tensor["data"], dtype=dtype)
        if list(result.shape) != tensor["shape"]:
            raise ValueError("Checkpoint tensor shape does not match metadata")
        state[name] = result
    return state
