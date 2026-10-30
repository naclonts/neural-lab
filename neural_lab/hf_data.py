"""Download public, revision-pinned Hugging Face datasets inside a worker.

No account login or token is used. Dataset repositories requiring executable
loading scripts are unsupported by the pinned datasets>=4 training image.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


REPO_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*(?:/[A-Za-z0-9][A-Za-z0-9_.-]*)?\Z")


def load_public_dataset(repo_id: str, split: str, writer, *, config: str | None = None,
                        revision: str = "main", max_rows: int | None = None,
                        label: str | None = None):
    """Return (Dataset, artifact entry), saving selected data and Hub provenance.

    `writer` is an ArtifactWriter rooted at this job's /output directory. The
    optional row cap is applied after loading; it is not a download-size quota.
    Set a bounded HF split such as ``train[:1000]`` for a small experiment.
    """
    if not REPO_ID.fullmatch(repo_id) or ".." in repo_id:
        raise ValueError("Use a public Hugging Face dataset ID, not a URL or local path")
    if not split or len(split) > 200:
        raise ValueError("Select one explicit dataset split")
    if max_rows is not None and (type(max_rows) is not int or max_rows < 1):
        raise ValueError("max_rows must be a positive integer")
    # Lazy imports keep the host supervisor and dashboard dependency-free.
    import datasets
    from datasets import load_dataset
    from huggingface_hub import HfApi

    metadata = HfApi(endpoint="https://huggingface.co", token=False).dataset_info(
        repo_id, revision=revision, timeout=20)
    resolved = metadata.sha
    if not isinstance(resolved, str) or not re.fullmatch(r"[0-9a-f]{40}", resolved):
        raise ValueError("Hugging Face did not report an immutable dataset revision")
    dataset = load_dataset(repo_id, name=config, split=split, revision=resolved,
                           token=False, cache_dir=str(writer.output / ".cache" / "datasets"))
    if max_rows is not None and len(dataset) > max_rows:
        dataset = dataset.select(range(max_rows))
    card = metadata.card_data
    license_name = card.get("license") if card is not None else None
    label = label or f"{repo_id.replace('/', '-')}-{config or 'default'}-{split}"
    filename = re.sub(r"[^A-Za-z0-9_-]+", "-", label).strip("-")[:120]
    if not filename:
        raise ValueError("Dataset label must contain a filename character")
    path = writer.output / "datasets" / (filename + ".parquet")
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(str(path))
    source = {"kind": "huggingface", "repo_id": repo_id, "revision": resolved,
              "requested_revision": revision, "config": config, "split": split,
              "url": f"https://huggingface.co/datasets/{repo_id}/tree/{resolved}",
              "license": license_name, "rows": len(dataset), "row_cap": max_rows,
              "features": dataset.features.to_dict(), "datasets_version": datasets.__version__,
              "cache": ".cache/datasets", "authentication": "public; token=False"}
    entry = writer.register_file("datasets", path, label, format="parquet", source=source)
    return dataset, entry


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_id")
    parser.add_argument("--split", required=True)
    parser.add_argument("--config")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--output", type=Path, default=Path("/output"))
    args = parser.parse_args(argv)
    from neural_lab.artifacts import ArtifactWriter

    _, entry = load_public_dataset(args.repo_id, args.split, ArtifactWriter(args.output),
                                  config=args.config, revision=args.revision, max_rows=args.max_rows)
    print(json.dumps(entry, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
