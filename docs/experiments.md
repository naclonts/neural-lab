# Candidate architecture experiments

The scripts in `jobs/candidates/` are self-contained PyTorch training jobs.
Each compares a candidate with an ordinary dense residual MLP on the **same**
seeded synthetic classification task. Both train on 1,536 examples and measure
accuracy and cross-entropy on 512 held-out examples for each of three seeds.
The teacher combines a linear signal with nonlinear feature interactions.

| Job | Candidate change | Matched control |
| --- | --- | --- |
| `conditioned_low_rank_gate.py` | Rank-16 transform whose residual update is gated by the original input | Dense residual block |
| `energy_normalized_mixer.py` | Normalize mixed-feature energy per example and learn a channel scale | Dense residual block |

Both arms use width 48, two residual blocks, 55 full-batch AdamW updates per
seed, the same optimizer settings, and the same held-out data. Candidate and
control parameter counts differ by less than 5%. The scripts report mean
held-out accuracy and loss, parameter counts, and measured training seconds in
`/output/result.json`; per-seed details go to `/output/details.json`.
Training uses deterministic PyTorch algorithms and a fixed seed for each arm.
The 300-second job timeout bounds each queued script on the target worker.

Queue both once with:

```bash
bash scripts/enqueue-candidate-jobs.sh
python3 -m neural_lab.queue status
```

The enqueue script checks script SHA-256 and hypothesis in the existing queue
before adding a job. Running it again with unchanged scripts does not duplicate
them. An edited script gets a new digest and can be queued as a new revision.
Execute queued jobs with `python3 -m neural_lab.queue worker` or the long-running
Compose worker in [research stack setup](orchestrator.md).

These are architecture **hypotheses**, not claims of globally novel networks or
superiority. The synthetic task is small and uses one fixed teacher per seed.
Results need an independent evaluator, broader datasets, more seeds, and runtime
matching before drawing stronger conclusions. GPU results can still vary across
hardware or library versions despite deterministic settings.

## Centered product screening experiment

`jobs/candidates/centered_product_mixer.py` compares the same dense control
with rank-16 centered projection products and a centered additive ablation.
Each projection subtracts its own per-example mean over rank channels. The
product is projected back to width 48 before GELU and residual addition.
Product and additive arms share every initial tensor; all arms share initial
input and output tensors. Parameter totals are 5,716 (dense) and 5,780
(product/additive). It retains seeds 1101–1103 and the original 55-update
protocol for exploratory continuity, and records task fingerprints, paired
accuracy/loss differences, and software/device settings.

The additive branch can collapse to a low-rank affine map before GELU.
Multiplication changes both polynomial degree and activation scale; dividing
the additive sum by sqrt(2) does not guarantee matched activation variance.
This is an operator ablation, with no isolated claim about centering,
parameter efficiency, or broad architectural novelty. Low-rank bilinear
products have established prior art, including
[Kim et al.](https://arxiv.org/abs/1610.04325). Existing inspected seeds are
development screening data; confirmation requires fresh seeds and tasks.
Timing has fixed arm order and supports no speed claim.

## Saved training data and checkpoint weights

New revisions of the starter, gate, energy, centered-product, and product-scale jobs write an
`artifacts.json` manifest in their output directory. Dataset files under
`datasets/` contain the actual split tensors in compressed NPZ form, with numeric
dtypes only. The manifest records tensor shapes/dtypes, generator, seeds, split
description, file size, and SHA-256. These jobs use generated Gaussian features
and synthetic four-class teachers, not downloaded natural-language or image data.
The starter uses 512 train/128 validation examples; the candidate screen uses
1,536 train/512 held-out examples per seed.

Each arm and seed saves weights under `checkpoints/`. Selection minimizes observed
**training** cross-entropy. Starter/gate/energy/centered-product jobs observe the
initialized model and every training update; the product-scale diagnostic saves
the best among its existing diagnostic evaluations at updates 0, 10, 25, and 55.
Held-out examples are not used to choose weights. Checkpoint metadata states the
selection metric/value, update, observation count, arm, seed, and configuration.
Result metrics still describe the final update, so they should not be presented
as validation metrics of the selected checkpoint. Checkpoint copying adds training
overhead; these timings do not establish speed comparisons with historical runs.

Weights use the pickle-free `neural-lab-state-dict-v1` JSON format, which includes
each tensor's dtype, shape, and values. In a training/review container, reconstruct
the architecture from the reviewed script, then load the tensor dictionary with
`neural_lab.artifacts.load_json_state(checkpoint_path)` and `model.load_state_dict`.
Load NPZ datasets with `numpy.load(dataset_path, allow_pickle=False)`.

Historical output directories are preserved. A manifest or weights absent from an
older run were not saved by that run; changes to a script produce a new queue
revision rather than retroactively changing its result. The original starter
`model-state.json` files contain final candidate weights, not selected best weights.
The shared `ArtifactWriter` and `BestCheckpoint` helpers are available for future
scientist-authored experiments to follow the same contract.

Queue the self-contained script without running training locally:

```bash
python3 -m neural_lab.queue enqueue jobs/candidates/centered_product_mixer.py --hypothesis "Centered low-rank product residuals improve held-out loss over dense residual and same-parameter centered additive controls under a fixed three-seed cross-feature protocol."
```

## Frozen product-scale diagnostic

`jobs/candidates/product_scale_diagnostic.py` compares dense, centered additive,
raw centered product, and a product with one frozen scalar per block. At shared
initialization, each scalar is the additive fused RMS divided by product fused
RMS (denominator floored at 1e-8), measured on the additive network's training
hidden trajectory. It multiplies product features before the up projection and
GELU. This matches initial aggregate second moments on that reference, without
normalizing each example or guaranteeing matching later in training.

The 24 fits use three fresh teacher seeds (5101–5103), two initialization seeds
(7101–7102), and separate training/development generators (6101/6102). Teacher
weights are drawn independently from features. Every teacher uses the same
feature draws deliberately; initializations are repeated within each teacher.
Width, rank, depth, parameter counts, optimizer and 55-update budget match the
centered-product screen. Checkpoints at 0/10/25/55 record training/development
performance and activation/logit magnitudes. Per-teacher paired differences
are the task-level comparisons; individual predictions are not replicates.

The primary endpoint is step-55 development loss, scaled minus raw product.
Dense and additive comparisons provide context. No final test is generated;
this diagnostic supports no promotion, general superiority or novelty claim.
Bilinear fusion and scale stabilization have established prior art, including
[Multi-modal Factorized Bilinear Pooling](https://arxiv.org/abs/1708.01471).
No architecture or gain sweep is part of the job. Queue without local training:

```bash
python3 -m neural_lab.queue enqueue jobs/candidates/product_scale_diagnostic.py --hypothesis "Frozen training-initialized pre-up product scale calibration reduces development cross entropy versus raw centered products under a preregistered fresh three-teacher, two-initialization protocol; dense and additive controls contextualize the diagnostic."
```
