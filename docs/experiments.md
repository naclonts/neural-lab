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

Queue the self-contained script without running training locally:

```bash
python3 -m neural_lab.queue enqueue jobs/candidates/centered_product_mixer.py --hypothesis "Centered low-rank product residuals improve held-out loss over dense residual and same-parameter centered additive controls under a fixed three-seed cross-feature protocol."
```
