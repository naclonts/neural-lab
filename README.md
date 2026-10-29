# Neural Lab

A starting point for an autonomous neural-network research lab on a Linux laptop
with an NVIDIA GPU.

**Current status:** GPU sandbox, offline visualizations, a bounded experiment loop,
and a persistent queue for deterministic training jobs. The headless research stack
uses subscription-backed GPT-6.1 Sol, subagents, live context handoff above 250k
tokens, and a 75% subscription usage pause.
Independent evaluation, Hugging Face publishing, and unattended-operation
hardening remain to be built.

## Requirements

- Linux with a supported NVIDIA driver.
- Docker Engine and Docker Compose v2.
- NVIDIA Container Toolkit configured for Docker.
- A driver compatible with the image's CUDA 12.8 runtime. Select a compatible
  PyTorch CUDA image if your driver cannot support it.
- A regular, non-root Linux account. Prefer a dedicated account for this project.

Docker daemon access is privileged; give it to the trusted supervisor, never to
generated experiment code.

## Quick start

Run from a clone of this repository:

```bash
bash scripts/setup.sh
docker compose pull
docker compose up -d
bash scripts/smoke-test.sh
```

Setup creates a private local `.env` containing your numeric UID/GID and an
`experiments/` directory. Both are ignored by Git. Never put account tokens or
personal files inside the experiment directory.

Inspect or stop the container:

```bash
docker compose logs --tail=100 lab
docker compose exec lab bash
docker compose down
```

Stopping the container preserves `experiments/`. The `unless-stopped` policy
restarts this idle environment after daemon restarts; it does not recover
experiments or implement a research loop.

## Local checks and progress

After setup and Codex login, validate the laptop:

```bash
python3 -m neural_lab.manage doctor --gpu-test
```

Inspect results at any time, including while the loop runs:

```bash
python3 -m neural_lab.manage status
```

The supervisor checks readiness before consuming planner attempts and pauses
before planning if free disk space drops below 10 GiB. See
[local operations](docs/local-operations.md) for checks and remaining limits.

## Agent loop

Use your ChatGPT account through the isolated Codex CLI planner:

```bash
bash scripts/codex-login.sh
python3 -m neural_lab.supervisor --planner codex --once
```

See [Codex subscription setup and limits](docs/codex.md). GPU workers never receive
the Codex login. Codex usage is subject to your account's allowance and credits.

Or run one synthetic research experiment without an account or API key:

```bash
python3 -m neural_lab.supervisor --once
```

The supervisor creates a disposable GPU worker, compares an MLP candidate against
a linear baseline, exports visualizations, journals results, and removes the
container. Run without `--once` for a bounded sequence. API-backed planning is
available with `--planner api --model YOUR_MODEL_NAME`.

See [agent loop usage, budgets, and recovery](docs/agent-loop.md). This is a
development loop; network egress and disk quotas still need hardening.

## Persistent research queue

Queue the three reproducible starter comparisons, then run or inspect GPU jobs:

```bash
bash scripts/enqueue-starter-jobs.sh
python3 -m neural_lab.queue status
python3 -m neural_lab.queue worker --once
```

For the research stack after ChatGPT sign-in:

```bash
bash scripts/codex-login.sh
bash scripts/research.sh start
bash scripts/research.sh status
```

The queue worker uses no OpenAI subscription while it trains. Startup queues three
controls and two candidate architectures once. Read [research stack setup](docs/orchestrator.md),
[candidate experiments](docs/experiments.md), and [queue operations](docs/queue.md)
for account setup, job recovery, and results. Docker and NVIDIA access must be
validated on the target laptop.

## Network visualizations

Generate interactive graphs, SVG layer diagrams, and JSON/DOT exports for an MLP,
CNN, and residual network inside a disposable sandbox:

```bash
bash scripts/visualize-demo.sh
```

Open `experiments/visualizations/mlp/index.html` locally. The view shows tensor
shapes, parameter counts, real branch edges, and sampled activation heatmaps.
See [visualization usage and limitations](docs/visualizations.md) to export your
own models.

## Sandbox boundaries

The worker runs without root privileges, with a read-only root filesystem,
no added Linux capabilities, and no permission to gain new privileges.
Only `experiments/` is bind-mounted from the host. Temporary files and shared
memory use bounded, ephemeral storage. CPU, RAM, process count, and Docker logs
are limited. GPU index 0 is exposed; adjust the Compose configuration as needed.

The worker has normal outbound Docker networking. There are no published ports,
but this does **not** prevent it from reaching host services or the LAN.
There is no disk quota on `experiments/`, GPU memory limit, thermal policy,
API budget, or per-experiment timeout yet. RAM limits are separate from GPU VRAM.

Containers share the host kernel and NVIDIA driver. This setup is a development
foundation, not a hardened boundary for unattended execution of arbitrary code.
Do not mount your home directory, Docker socket, SSH agent, or credentials.
Do not use privileged mode or host networking. Before unattended use, add
enforced egress restrictions, storage quotas, and the supervisor controls described
in [the architecture notes](docs/architecture.md).

The PyTorch image is version-tagged, not digest-pinned. Pin a reviewed digest
before a long-running deployment.

## Validation

`scripts/setup.sh` invokes `docker compose config --quiet`.
`scripts/smoke-test.sh` checks the non-root user, read-only root mount, writable
workspace/tmp, and one real GPU training step.

GitHub Actions validates Compose configuration, network visualization exports,
persistent supervisor state, and real CPU Docker isolation/timeout behavior.
The NVIDIA driver and GPU configuration still need validation on the target laptop.
