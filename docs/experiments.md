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
