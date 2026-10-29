# Headless research stack

`compose.research.yaml` separates two processes. The `worker` polls the persistent
SQLite queue and launches disposable GPU training containers. The `orchestrator`
uses Codex with ChatGPT sign-in to research, implement experiment scripts, and
enqueue them. The orchestrator prompt asks it to delegate independent work to
Codex subagents. Worker jobs are deterministic and make no OpenAI calls.

## Set up the local worker

Run from this repository on a Linux host with Docker Compose and NVIDIA Container
Toolkit installed:

```bash
bash scripts/setup.sh
docker compose pull
bash scripts/smoke-test.sh
python3 -m neural_lab.manage doctor --gpu-test
bash scripts/enqueue-starter-jobs.sh
docker compose -f compose.research.yaml up -d --build worker
```

If `.env` existed before this research stack was added, `scripts/setup.sh` adds
the missing `LAB_REPO_DIR`, `DOCKER_GID`, and `NEURAL_LAB_CODEX_HOME` values.
Check those paths after moving the checkout.

Inspect the queue and container logs:

```bash
python3 -m neural_lab.queue status
docker compose -f compose.research.yaml logs --tail=100 worker
docker compose -f compose.research.yaml ps
```

The three starter scripts compare a linear control, a deep MLP, and a residual
MLP on the same seeded synthetic task. They establish the workflow; their results
do not establish architectural novelty. See [queue operations](queue.md) to add
a reviewed script and inspect saved outputs in `.neural-lab/queue-outputs/`.

## Codex sign-in and research sessions

The dedicated sign-in directory is outside this public Git checkout. Authenticate
locally and confirm that Codex reports ChatGPT sign-in:

```bash
bash scripts/codex-login.sh
bash scripts/codex-login.sh status
```

The directory defaults to `~/.local/share/neural-lab/codex` and is bind-mounted
only into the orchestrator. It contains bearer credentials: keep it private, never
commit it, and never mount it into a training worker. `.env` and `.neural-lab/`
are Git-ignored, but inspect `git status` before any public push.
Codex CLI supports ChatGPT account sign-in and headless `codex exec`; its usage
counts toward ChatGPT plan limits and applicable ChatGPT credits, without using
OpenAI API-key billing. [Codex CLI](https://learn.chatgpt.com/docs/codex/cli),
[non-interactive mode](https://learn.chatgpt.com/docs/non-interactive-mode).

**The orchestrator is intentionally gated today.** As of October 28, 2026, OpenAI
documents GPT-6.1 Sol with a default **500,000-token** context window in Codex.
The API model's 1,050,000-token specification does not verify that full window
for ChatGPT subscription-backed Codex. GPT-5.4's experimental 1M Codex support
ended for ChatGPT sign-in on August 31, 2026. No current subscription-backed
Codex model has a documented, verified million-token window. Setting
`model_context_window` in Codex config only declares client metadata; verify
actual model access and context capacity before setting
`NEURAL_LAB_ORCH_MODEL` and `NEURAL_LAB_CONTEXT_TOKENS`. [Codex models](https://learn.chatgpt.com/docs/models),
[API models](https://developers.openai.com/api/docs/models),
[configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference).

Once a model and its context limit are verified for the signed-in account, set
both variables in the local `.env`, for example using the verified model ID and
token count, then start:

```bash
docker compose -f compose.research.yaml up -d --build orchestrator
docker compose -f compose.research.yaml logs -f orchestrator
```

Do not substitute an API key to satisfy the gate. The orchestrator checks
`account/read` for `chatgpt` auth and calls the official app-server
`account/rateLimits/read` method before each Codex session. If quota metadata is
unavailable, it pauses planning. It compares available primary and secondary
`usedPercent` windows with the 75% threshold; training jobs already queued keep
running in the separate worker service. Limit windows and reset times are account
specific. [Codex app-server protocol](https://learn.chatgpt.com/docs/app-server).

The code starts a fresh `codex exec` session on each cycle. **This does not
guarantee a live shutdown exactly when context use reaches 50%.** The JSON event
stream reports usage when a turn completes, so live occupancy and timely handoff
still need implementation and validation. The orchestrator may also require
account-specific validation of app-server quota fields before it can run.
Review `.neural-lab/notes/` for private handoff notes; they are never committed.

Stop the research services with:

```bash
docker compose -f compose.research.yaml down
```

Stopping services preserves the queue, completed results, and notes in
`.neural-lab/`. The research worker has Docker daemon access so it can launch
training containers; only run it from a trusted local checkout. The current
training boundary still lacks enforced egress and disk quotas, GPU thermal
monitoring, and independent evaluation. See [architecture](architecture.md) and
[agent loop](agent-loop.md) before unattended operation.
