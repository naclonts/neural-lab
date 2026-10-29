# Persistent deterministic jobs

The queue stores reviewed training scripts and results in `.neural-lab/`, which Git
ignores. Enqueue snapshots each script and records its SHA-256 digest. The trusted
host worker runs one queued job at a time through the existing disposable Docker
runner. No Codex call is made while queued jobs execute, so the worker can continue
while the ChatGPT planner waits for account capacity.

```bash
bash scripts/enqueue-starter-jobs.sh
python3 -m neural_lab.queue status
python3 -m neural_lab.queue worker --once
python3 -m neural_lab.queue worker
```

The starter jobs compare a small control, a deeper MLP, and a residual MLP on the
same seeded synthetic dataset. They are exploratory controls, not novelty claims.
To queue another reviewed script:

```bash
python3 -m neural_lab.queue enqueue path/to/train.py \
  --hypothesis 'A specific testable claim' --timeout-seconds 300
```

Scripts must be UTF-8 regular files at most 64 KiB. They must write
`/output/result.json` in the worker contract documented in [agent loop](agent-loop.md).
Input files are read-only in the container. The queue records completed outputs in
`.neural-lab/queue-outputs/<job-id>/`. The worker checks Docker, GPU, image, and
disk readiness before taking a job. If the host is not ready, queued work remains
queued. A crash interrupts the running job but leaves later jobs queued. The next
worker start removes stale containers for this queue and marks that job interrupted.

Use `--cpu` for CPU-only development. The worker defaults to the GPU. Its Docker
isolation and host limitations are the same as the existing supervisor, including
normal Docker bridge networking and no enforced output disk quota.
