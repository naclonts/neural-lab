# Training data and Hugging Face

The initial experiments use generated data, not a downloaded benchmark:

- Starter controls: 640 Gaussian vectors with 16 features; labels from a seeded
  four-class linear teacher plus a sinusoidal term, split 512 train / 128 held-out.
- Original candidate comparisons: 2,048 Gaussian vectors, four-class labels from
  linear, pairwise-interaction, and sinusoidal teacher terms; 1,536 train / 512
  held-out, seeds 1101–1103.
- Product-scale diagnostic: the same task family with separate teacher, data,
  and initialization seeds, and training/development splits.

These are small exploratory screens. A completed script is not evidence of a
successful hypothesis or a general improvement on real datasets.

`jobs/huggingface/iris_classifier.py` is a small real-data connection control.
It downloads Iris, makes a seeded stratified 120/30 training/validation split,
standardizes with training statistics, and compares a linear classifier with a
width-16 MLP. It saves the source Parquet, actual normalized split tensors, and
both arms' best observed training-loss weights. The checkpoints include feature
order, class names, normalization statistics, and architecture descriptions.

## Public Hugging Face access

The training image pins `datasets==4.8.5`, `huggingface_hub==1.3.7`,
`pyarrow==21.0.0`, and `pandas==2.3.3`:

```bash
docker build -t neural-lab-training:torch2.8-hf -f docker/training.Dockerfile .
```

Use this image for queued jobs that import the data loader. It extends the
existing PyTorch CUDA image and retains the worker's network, user, mount, and
resource boundaries. Workers receive no host home, Codex account, or HF login.
Public datasets need no account token. The helper explicitly uses `token=False`
for both metadata and data requests, so it does not borrow an ambient login.
Private/gated data requires a separately designed scoped read-token path;
neural-lab does not currently inject those credentials into training jobs.

Inside a queued training script:

```python
from neural_lab.artifacts import ArtifactWriter
from neural_lab.hf_data import load_public_dataset

writer = ArtifactWriter('/output')
dataset, source = load_public_dataset(
    'scikit-learn/iris',
    split='train',
    writer=writer,
    revision='main',  # resolved to an immutable commit before data is loaded
)
# Use dataset columns to create the experiment's tensors and its own split.
# This repository's single split contains all 150 examples; it is not an
# independent training/evaluation protocol by itself.
```

The loader resolves the requested revision with the official Hub API, then loads
that exact commit. It saves a Parquet snapshot of the selected rows under
`/output/datasets/` and registers it in `artifacts.json`, including SHA-256,
repository URL, immutable commit, configuration, split expression, row count,
feature schema, reported license, and library version. Subsequent preprocessing
and train/dev/test partitioning should also be recorded in the experiment's data
artifacts; the raw source snapshot alone does not capture transformations.
[Hugging Face loading reference](https://huggingface.co/docs/datasets/package_reference/loading_methods),
[Hub dataset metadata](https://huggingface.co/docs/huggingface_hub/package_reference/hf_api).

`datasets` version 4 removed support for repository loading scripts. Use
supported data formats such as Parquet, CSV, or JSON. The helper accepts Hub
dataset identifiers, not executable scripts, arbitrary URLs, or host paths.
For image/audio datasets, preserve referenced external assets as well as the
Parquet snapshot when the selected representation contains file references.

The public [Iris repository](https://huggingface.co/datasets/scikit-learn/iris)
reports 150 rows and CC0-1.0 metadata. Choose the dataset, license, data budget,
split, and baseline appropriate to each new research question. `max_rows` limits
the saved selection **after loading**, not download size; even a slice expression
can require downloading a larger source shard. Worker timeouts and memory limits
still apply. The per-job cache is `.cache/datasets/` under the output directory;
HF's underlying download cache uses the worker's ephemeral `/tmp/huggingface`.

For a container-only connectivity smoke check, with a writable output directory:

```bash
docker run --rm --user "$(id -u):$(id -g)" --read-only --cap-drop ALL \
  --security-opt no-new-privileges:true --tmpfs /tmp:rw,size=256m,mode=1777 \
  --mount "type=bind,src=$PWD/neural_lab,dst=/opt/tooling/neural_lab,readonly" \
  --mount "type=bind,src=$PWD/.neural-lab/hf-smoke,dst=/output" \
  -e PYTHONPATH=/opt/tooling neural-lab-training:torch2.8-hf \
  python -m neural_lab.hf_data scikit-learn/iris --split train
```

Create `.neural-lab/hf-smoke` first. This smoke check downloads and snapshots data;
it does not perform training or establish a new experiment result. No files are
published back to Hugging Face.
