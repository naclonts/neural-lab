# Candidate architecture experiments

## Overnight exploratory screens

The following completed scripts retain exact synthetic data, training-selected
weights, final development endpoints, and their frozen comparison criteria via
`artifacts.json`. Private result files are accessible in the local dashboard;
they are not stored in the public repository.

| Script | Research question | Historical training protocol |
| --- | --- | --- |
| `teacher_family_specificity.py` | Bounded products versus additive controls across teacher families | 60 fits, 40 full-batch updates |
| `occupancy_delta_memory.py` | History-dependent delta-memory write rates on associative recall | 48 fits, 60 updates, CUDA |
| `occupancy_delta_controls.py` | Constant, timestep, and shuffled-history controls for occupancy gates | 60 fits, 60 updates, CUDA |
| `fourier_feature_classifier.py` | Fixed Fourier features on periodic and affine targets | 48 fits, 120 CPU updates |
| `input_gated_fourier_classifier.py` | Input-dependent blending versus fixed gates and concatenation | 108 fits, 120 CPU updates |
| `permutation_pool_classifier.py` | Established DeepSets pooling versus order-sensitive controls | 72 fits, 120 CPU updates |

These historical budgets are preserved for reproducibility. The current
scientist policy requests larger, longer CUDA campaigns with stronger data and
training budgets; see [research operations](orchestrator.md#monitor-and-control).
CPU-only synthetic screens are preliminary diagnostics. None of these scripts
establishes broad superiority or architectural novelty.

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

## Rotated teacher-family specificity screen

`jobs/candidates/teacher_family_specificity.py` compares dense, raw additive,
raw product, tanh additive and tanh product residuals with 40 full-batch AdamW
updates each. The four factor arms share exactly initialized projection tensors;
the tanh additive control isolates fusion from the transformed-factor change.
Both teachers use the same independently drawn orthogonal rotation and Gaussian
inputs per replicate, with the preregistered `x = z @ Q` convention. Additive logits are a latent linear map; interaction logits
use four disjoint latent coordinate products without linear terms. Coefficients
are class-centered and globally normalized to average population logit variance
one. This does not equalize task difficulty.

Three fresh teacher seeds (8101–8103), two initializations (10101/10102), and
independent train/dev data streams (9101/9102, 9111/9112, 9121/9122) give 60 fits.
The primary endpoint is final-update development loss difference
`(tanh_product - tanh_additive)_interaction - (tanh_product - tanh_additive)_additive`,
averaging initializations within teacher before averaging teachers. Negative
values indicate a relative family effect; actual within-family additive and dense
comparisons remain necessary to establish a within-family advantage. Diagnostics
at 0/10/25/40 include factor saturation, residual magnitudes and class balance.

`ArtifactWriter('/output')` stores six exact numeric NPZ datasets with train/dev
features and labels, latent draws, rotations, raw and normalized teacher coefficients,
normalizers and teacher logits. All six datasets are saved before fitting, with seeds,
array hashes and script-hashed generator provenance. Per-example activation RMS
quantiles, exact-state gradient finiteness and teacher logit variance/margins are
recorded alongside the three teacher-level primary contrasts and their range. Sixty unique best-training-loss checkpoints
observe every state 0 through 40, with metric, value, step, scope, model config and
dataset links. Reported final-update development metrics remain independent of
checkpoint selection. No final test, tuning or promotion is part of this screen.
The 40-update budget differs from earlier 55-update runs. Bounded factor products
and multiplicative residuals have established precedent, including
[Hadamard Product for Low-rank Bilinear Pooling](https://arxiv.org/abs/1610.04325)
and [Multimodal Residual Learning](https://arxiv.org/abs/1606.01455).

## Occupancy-conditioned delta memory screen

`jobs/candidates/occupancy_delta_memory.py` starts a fresh synthetic sequence
experiment after the negative bounded-product screen. Eight key/value writes and
one query ask for the last value written to that key. Unique sequences permute
all eight keys; overwrite sequences sample eight keys from a random four-key
subset. Queries sample uniformly among distinct written keys. Both regimes use
independent uniform values and the same last-occurrence label rule. The generator
rejects complete-example duplicates across each train/dev pair and stores the
ordered tokens, query, labels, answer positions and actual flattened dense inputs.

Four arms compare a generous ordered-input dense residual network (6,760
parameters) with fixed-rate delta memory (392), learned constant-rate delta memory
(393), and an occupancy-conditioned rate (394). Delta keys and queries are learned
16-dimensional embeddings normalized with a norm floor of 1e-6. Each write updates
`M += beta * (v - M k) k^T`; query output is `Linear(M q)`. All three delta arms
compute zero-initialized value memory M and key covariance C, each 16 by16 per
example (2,048 persistent-state bytes total in float32). C updates after each
memory write as `.9*C + .1*k*k^T`. Occupancy reads prior `k^T C k`, clamped at zero
only for PSD roundoff. The conditioned rate is
`sigmoid(b + a*log(1e-4 + occupancy))`; constant is `sigmoid(b)`; fixed is0.5.
With a=b=0, all delta arms copy identical embeddings/output weights and assert
exactly equal initial logits on every actual train/dev dataset. C remains in the
autograd graph; the query changes no state. Dense sees the ordered concatenated
key/value one-hots and query, so positions and all writes remain available.

Three independent dataset replicates (IDs12101–12103) use training/development
RNG streams13101/13102,13111/13112,13121/13122. Add100000 to each stream in
the overwrite regime. IDs121xx identify replicates and do not seed an unused
teacher. Two model initialization seeds14101/14102 yield48 fits. Every fit uses
512 train/512 development examples and exactly60 full-batch AdamW updates with
lr0.003, weight decay0.001, betas(0.9,0.999), epsilon1e-8. No final test is generated.

The primary endpoint is final60 overwrite development CE, occupancy minus
constant, averaging initializations within dataset before averaging datasets.
The screen is favorable only if the primary mean and all three dataset means are
negative, and unique-regime mean occupancy-minus-constant CE is no greater than
0.01. These are exploratory screening criteria. All occupancy/constant/fixed/dense
contrasts, dataset ranges and accuracy differences remain in details.json.
Diagnostics at0/20/40/60 include gate and occupancy distributions and memory RMS.
The result envelope has a fixed20 finite metrics, independent of observed results.

`ArtifactWriter('/output')` saves all six exact datasets before fitting with
source hash, exact streams, draw order, rejection counts, per-tensor hashes,
schema, file hashes and software provenance. Each of48 selected checkpoints
minimizes actual full-training CE over all61 states0..60; metadata links the exact
dataset and its hash. Final60 development metrics are evaluated independently and
never select weights. Only the trusted queue may train this script after review.

This tests fixed-budget synthetic recall, not novelty, efficiency or superiority
over competitive recurrent/attention models. Dense has more parameters but is
not compute matched. Occupancy has one more learned scalar than constant, and
regimes differ in difficulty. A favorable screen needs history-shuffled/token-only
gate controls, a competitive sequence baseline and fresh final-test confirmation.
Failure stops this frozen protocol without tuning it into success. Relevant
precedent includes [DeltaNet](https://arxiv.org/abs/2406.06484) and
[Gated Delta Networks](https://arxiv.org/abs/2412.06464).
