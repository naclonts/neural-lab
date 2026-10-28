# Local readiness and run status

## Check the laptop before running experiments

After setup and Codex login, run:

```bash
python3 -m neural_lab.manage doctor --gpu-test
```

Doctor checks:

- Linux, Python 3.10+, and a non-root host user.
- Free space on the state filesystem (10 GiB minimum by default).
- Access to a local Docker daemon and presence of the worker image.
- The selected NVIDIA GPU, its VRAM, and a temperature below 85 C.
- For the Codex planner, its image and cached ChatGPT login via the official CLI.
- With --gpu-test, a trusted CUDA matrix calculation inside the worker sandbox.

It does not call a model, reserve an experiment attempt, inspect credential files,
or report raw login output. A cached login does not establish remaining account
allowance or guarantee that the next model request succeeds.

The CUDA probe uses a disposable container with the usual worker limits and
cleans it up after the test. It is a setup test, not an ongoing GPU temperature
monitor. Doctor exits nonzero if a required check fails.

Use --planner demo to omit the Codex checks, --gpu N to select another GPU, or
--cpu for CPU-only checks. --image selects a trusted worker image; its presence
alone does not prove that it contains PyTorch. The CUDA probe validates PyTorch
and the container's GPU access.

The supervisor runs the normal readiness checks automatically after startup
recovery and before planning. Run doctor --gpu-test explicitly once to validate
CUDA execution; the supervisor does not repeat that probe for every invocation.
Missing images, unavailable Docker/GPU, and missing cached Codex login are reported
without consuming a planner attempt.

## Inspect progress and results

```bash
python3 -m neural_lab.manage status
python3 -m neural_lab.manage status --limit 10
```

Status works while the supervisor is running and needs neither Docker nor a GPU.
It opens the existing database in SQLite read-only mode and reports attempt count,
reserved training seconds, status counts, recent findings/errors, and output paths.

It does not acquire the supervisor lock, recover runs, reserve attempts, or change
run records. On a new checkout it reports uninitialized state without creating it.
Use --state-dir with either command if your supervisor uses another state directory.
Budget thresholds come from supervisor arguments; status shows recorded usage,
not an inferred remaining balance.

JSON output escapes control characters from worker summaries rather than printing
them as terminal commands. It is suitable for a future local dashboard.

## Disk admission guard

The supervisor checks free disk space before each planner call and pauses if it
falls below --min-free-gb (default 10). The threshold must be at least 1 GiB.
No new attempt is reserved when this check blocks a run.

This checks the state/output filesystem. Docker's separate image-storage filesystem
may need its own monitoring. The guard cannot stop a running worker from filling
the disk: enforced quotas and retention remain necessary for unattended operation.

## Suggested first session

```bash
python3 -m neural_lab.manage doctor --gpu-test
python3 -m neural_lab.supervisor --planner codex --once
python3 -m neural_lab.manage status
```

Inspect the run's result and visualization artifacts before starting a longer batch.
Automatic model publishing, thermal shutdown, egress restrictions, and hard disk
quotas remain separate work.
