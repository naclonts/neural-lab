# Headless research stack

The stack runs an independent GPU job worker and a Codex research orchestrator.
The orchestrator researches architectures, delegates to subagents, implements
training scripts, and queues them. Queued jobs make no OpenAI calls and continue
while the orchestrator waits for subscription capacity.

## Start on the laptop

Use a regular Linux account with access to the local Docker daemon and a working
NVIDIA driver plus NVIDIA Container Toolkit:

```bash
bash scripts/setup.sh
bash scripts/codex-login.sh
bash scripts/research.sh start
```

The login script opens the official ChatGPT device login flow. Complete that in
your browser once. Credentials stay in a dedicated directory outside the public
checkout, defaulting to `~/.local/share/neural-lab/codex`. The startup script pulls
the GPU image, checks real GPU execution, builds both research containers, and
queues the starter jobs. Repeated startup does not duplicate existing starter or
candidate jobs.

Setup preserves existing `.env` values and adds missing paths and Docker group
settings. Check `LAB_REPO_DIR`, `DOCKER_GID`, and `NEURAL_LAB_CODEX_HOME` after
moving a checkout or changing Docker permissions. `.env`, `.neural-lab/`, logs,
notes, and authentication are excluded from Git. The Docker build context also
excludes private state. Public commits must contain only reviewed source files.

This task's managed shell may not expose the laptop's Docker socket or NVIDIA
devices. A passing static Compose check does not establish that GPU training is
running; the startup GPU check must pass on the laptop.

### NVIDIA CDI startup failures

If Docker reports `Error updating ldcache: execveat: no such file or directory`
before Python starts, test NVIDIA's explicit runtime path:

```bash
docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime \
  python -c 'import torch; print(torch.ones(2, device="cuda"))'
```

On the validated Debian 6.1 / Toolkit 1.20.1 / Docker 29 host, the CDI library-cache
hook fails but the explicit NVIDIA runtime works with `mode = "legacy"` in
`/etc/nvidia-container-runtime/config.toml`. Back up that host configuration before
changing it with `sudo nvidia-ctk config --set nvidia-container-runtime.mode=legacy
--in-place`. This setting affects other users of the NVIDIA runtime too. Restore
the backup to undo it. See [NVIDIA runtime configuration](https://github.com/NVIDIA/nvidia-container-toolkit/blob/v1.20.1/cmd/nvidia-container-runtime/README.md).

After the explicit-runtime CUDA test passes, set `NEURAL_LAB_GPU_BACKEND=nvidia`
in the private `.env`. Startup passes it to both the host GPU probe and the queue
worker. For direct Python commands, set it in the command environment:

```bash
NEURAL_LAB_GPU_BACKEND=nvidia python3 -m neural_lab.manage doctor --planner demo --gpu-test
```

The default `docker` backend uses `--gpus device=0`; the `nvidia` backend uses
`--runtime=nvidia` and selects the GPU with `NVIDIA_VISIBLE_DEVICES`. Both retain
the same user, read-only filesystem, capability, mount, and resource limits.

## Monitor and control

The scientist now targets substantive CUDA training campaigns rather than a
sequence of tiny CPU screens. Typical useful training should take about 10–20
minutes per job within a 30-minute timeout. Runtime is a planning target, not an
instruction to pad experiments. The controller requests 2–4 justified, reviewed
jobs per campaign when appropriate, larger data/model/training budgets, fair
controls, and recorded CUDA placement, training time, and peak GPU memory.
Public Hugging Face benchmarks are available in the training image. The pending
relational proposal must be revised and reviewed under this policy before use.
Existing completed experiments and their frozen protocols are preserved.

```bash
bash scripts/research.sh status
bash scripts/research.sh logs
python3 -m neural_lab.queue status
```

The status command shows queue counts, job metrics, service state, and recent
logs. Outputs live in `.neural-lab/queue-outputs/<job-id>/`. The initial jobs compare
three baseline architectures and two candidate designs; see
[experiment descriptions](experiments.md). These are exploratory experiments,
with no established novelty or improvement claim.

```bash
bash scripts/research.sh pause-research
bash scripts/research.sh resume-research
bash scripts/research.sh stop
```

Pausing research leaves the GPU worker running. Stopping the stack preserves all
queue state, results, and notes. Both services restart with Docker unless manually
stopped. GPU jobs have individual timeouts and run one at a time. See
[queue recovery](queue.md) for interrupted jobs.

## Model, handoff, and subscription capacity

The default is `gpt-6.1-sol`, which the current Codex catalog reports with a nominal
500,000-token context window. The orchestrator uses the official app-server
protocol with ChatGPT login and the OpenAI model provider. It does not use API
keys or automatically switch to API billing. [Codex models](https://learn.chatgpt.com/docs/models),
[app-server protocol](https://learn.chatgpt.com/docs/app-server).

Each session starts a new thread and delegates research, experiment planning,
implementation, and independent review to subagents. The controller watches
`thread/tokenUsage/updated` and hands off when the latest model call reports
usage **strictly above 60% of its reported context window**. A 258,400-token
window hands off above 155,040 tokens; a 475,000-token window hands off above
285,000 tokens. The exact 60% boundary continues. Smaller reported windows are
accepted without overriding or fabricating model metadata.

Context handoff is an expected successful session outcome: the controller saves
the actual window, token count, dynamic threshold, session IDs, and recent agent
messages before interruption. It shuts down that app-server and starts a fresh
thread after the configured pause, carrying the prior handoff forward. Subagents
must write durable findings throughout work so the next session can continue.
Telemetry arrives between model calls, so crossing is detected at the next
update; this cannot stop a model call partway through its token stream. Missing
or invalid telemetry still causes a conservative pause.

On this host a timed-out Codex model-catalog refresh produced a 258,400-token
report for `gpt-6.1-sol`, while earlier sessions reported 475,000. This no longer
fails the controller: the threshold adapts to the active reported window. Fixed
catalog-error categories are retained for invalid-telemetry diagnostics without
storing raw CLI logs or credentials. The independent worker continues queued jobs.

Handoff notes are private under `.neural-lab/notes/`. The controller records token
usage, session identifiers, interruption reasons, and recent agent messages in an
atomic checkpoint. The prompt instructs the orchestrator and its subagents to
write findings incrementally so work survives context rollover or restart.
[Codex subagents](https://learn.chatgpt.com/docs/agent-configuration/subagents).

Before a session, the controller verifies ChatGPT authentication and reads
`account/rateLimits/read`. It pauses new model work when any reported primary or
secondary usage window reaches **75%**, when included usage is explicitly blocked,
or when usable quota information is unavailable. It also checks quota during
active work and interrupts the turn if the threshold is reached. It rechecks
capacity periodically and resumes when the account reports capacity below the
threshold. The independent GPU worker continues processing queued jobs throughout.
Usage reads and interruptions are not billing transactions; a model call already
in flight can finish before a quota update arrives.

ChatGPT subscription allowances and purchased ChatGPT credits are controlled by
the account. This project does not buy credits, redeem reset credits, or implement
an API fallback. Keep account credit settings consistent with subscription-only
use. [App-server account and quota methods](https://learn.chatgpt.com/docs/app-server).

## Container permissions

The orchestrator has a writable container filesystem, internet access, unrestricted
commands, and passwordless `sudo` inside the container. Routine files are written
as the laptop UID/GID. Only the repository and dedicated Codex login directory
are mounted; the host home and Docker socket are absent. The Docker container
provides the filesystem boundary. The repository and login mounts are intentional
exceptions and must be treated as trusted inputs to the research agent.

The separate trusted worker receives Docker daemon access so it can launch
training containers. Its controller code is baked into its image. Training
containers receive only that job's input/output directories and the shared
experiment helper package; they receive no Codex credentials or Docker socket.
The current training boundary still lacks disk quotas and enforced network egress
restrictions. See [architecture](architecture.md) for those limits.
