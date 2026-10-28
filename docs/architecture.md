# Intended architecture

## Research loop

1. A trusted supervisor asks a planner for a hypothesis and experiment code.
2. The supervisor allocates a run ID and a dedicated working directory.
3. A fresh GPU container executes the experiment within fixed resource and time limits.
4. A separate evaluation step compares results against a baseline.
5. The supervisor records findings and gives them to the planner.
6. A separate publisher uploads qualifying artifacts to Hugging Face.
7. Repeat with bounded retries and backoff.

The Compose service is an idle development sandbox. The first supervisor now
creates a disposable container per run, exposes only that run's input/output and
reviewed tooling, and journals worker-reported outcomes. It supports deterministic
demo planning and API-backed experiment proposals. Independent evaluation and
publishing remain planned. See [agent-loop usage](agent-loop.md).

A worker cannot choose its own mounts, Docker flags, or limits.

## Trust boundaries

The host supervisor owns Docker control, API credentials, budgets, and the
research journal. Generated code and downloaded datasets are untrusted.
Never execute generated code, model-loading code, or package installation on
the host. Planner output is data until executed in a worker.

Keep planner and Hugging Face credentials outside worker mounts and environments.
Treat metrics and files from the worker as untrusted input. Validate output paths,
reject symlinks and special files, cap artifact sizes, and never load arbitrary
pickle files on the host. Prefer safetensors for weights.

## Before unattended operation

The supervisor implements runtime deadlines and cleanup, journaled reservations,
failure backoff/circuit breaking, and single-instance locking. Remaining work:

- Enforce storage quotas and a retention policy; bound stdout and stderr.
- Restrict outbound traffic through an enforced gateway that blocks private,
  loopback, link-local, and metadata destinations. Proxy environment variables
  alone do not enforce a network policy.
- Monitor GPU temperature and usage; define cooldown and shutdown thresholds.
  CPU/RAM cgroups do not enforce VRAM or GPU power limits.
- Define allowed data sources, dataset licenses, and download size limits.
- Evaluate on held-out data against reproducible baselines with recorded seeds.
- Publish model cards documenting data provenance, evaluation, and limitations.
- Default initial uploads to private repositories with a narrowly scoped token.
- Add an operator-installed service for unattended restarts after hardening.
  The current supervisor already handles graceful shutdown and restart recovery.

Architecture variation is not evidence of research novelty. Describe results
as exploratory until comparisons and evaluation support stronger claims.
