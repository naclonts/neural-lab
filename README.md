# Neural Lab

A starting point for an autonomous neural-network research lab on a Linux laptop
with an NVIDIA GPU.

**Current status:** GPU sandbox, offline visualizations, and a bounded persistent
experiment loop with demo and API-backed planners. Independent evaluation,
Hugging Face publishing, and unattended-operation hardening remain to be built.
The Compose service stays idle; the supervisor launches disposable workers.

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

## Agent loop

Run one synthetic research experiment without an API key:

```bash
python3 -m neural_lab.supervisor --once
```

The supervisor creates a disposable GPU worker, compares an MLP candidate against
a linear baseline, exports visualizations, journals results, and removes the
container. Run without `--once` for a bounded sequence. API-backed planning is
available with `--planner api --model YOUR_MODEL_NAME`.

See [agent loop usage, budgets, and recovery](docs/agent-loop.md). This is a
development loop; network egress and disk quotas still need hardening.

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
