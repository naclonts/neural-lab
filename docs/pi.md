# Use the pi coding agent with API-key billing

The pi backend runs the official [pi coding agent](https://www.npmjs.com/package/@earendil-works/pi-coding-agent)
as both the single-proposal planner and the headless research orchestrator. It is
the direct counterpart of the ChatGPT-subscription Codex path; the security
boundaries, untrusted plan validation, and GPU worker sandbox are identical.

**Billing disclaimer:** pi uses pay-per-token **API billing**. There is no ChatGPT
(or other subscription) included usage, and no API fallback exists on the Codex
path — each backend is strictly separate. Because a subscription pause cannot
protect you here, the orchestrator replaces quota pausing with a local **budget
gate** (see below). You are responsible for your provider's API spend.

## Image build

The planner/orchestrator image is pinned to pi 1.5.0 on node:24-bookworm-slim.
`scripts/pi-auth.sh set-key` (and an explicit `scripts/pi-auth.sh build`) build
it automatically; `status` and `clear` reuse the existing image. To build it
manually (never as generated code):

```bash
docker build -t neural-lab-pi:1.5.0 -f docker/pi.Dockerfile .
```

## Configure the API key

```bash
printf 'YOUR_API_KEY' | bash scripts/pi-auth.sh set-key
bash scripts/pi-auth.sh status
bash scripts/pi-auth.sh clear
```

- `set-key` reads the key from **STDIN**, never from argv, so it stays out of the
  process list and shell history. It writes `auth.json` inside the dedicated auth
  directory with mode 0600 (directory 0700).
- The provider object is selected by `PI_PROVIDER` (default `anthropic`).
- The auth directory defaults to `~/.local/share/neural-lab/pi`; set
  `NEURAL_LAB_PI_HOME` to override. Setup records it in the private `.env`, and
  the doctor/planner/orchestrator containers resolve the same path from the Compose
  configuration. An explicitly set `NEURAL_LAB_PI_HOME` takes precedence over the
  Compose-resolved path; `scripts/pi-auth.sh` warns when the two diverge.
- `status` runs a hardened one-shot `pi --print --offline` container and inspects
  the exit code explicitly. A clean exit means a real provider round-trip through
  `pi --print` succeeded; `--offline` only suppresses update and catalog
  refreshes, not provider inference.

## Planner (single experiments)

```bash
python3 -m neural_lab.supervisor --planner pi --model anthropic/claude-sonnet-4.5 --once
```

`--model` is required for the pi planner. `--pi-home` overrides the auth
directory for one run, like `--codex-home`. The planner container receives only
the request mount, an answer mount, and the auth directory at `/pi`; it runs
`pi --print --no-session --no-tools --offline` with a read-only root, dropped
capabilities, and CPU/memory/process limits. No API key is ever passed through
the environment. The returned plan is validated exactly like the Codex and API
planner outputs: untrusted data, never executed on the host.

## Orchestrator (headless research)

```bash
bash scripts/pi-auth.sh status
bash scripts/research.sh start-pi
```

`start-pi` normally validates readiness with `doctor --planner pi`, then starts
the worker and the `pi-orchestrator` service. With `PI_PROVIDER=ollama`, it
uses the non-inference demo readiness checks instead, so startup cannot load a
model while a worker may be using the GPU. The orchestrator container runs
`python3 -m neural_lab.orchestrator --backend pi`; `--backend` defaults to
`NEURAL_LAB_ORCH_BACKEND` (default `codex`), so the Codex path is unchanged.
The default model is `NEURAL_LAB_ORCH_MODEL` (`anthropic/claude-sonnet-4.5` in
the pi service). The orchestrator drives `pi --mode rpc --no-session` over JSONL
records, prepends the previous private handoff, and writes handoff notes to
`.neural-lab/notes/latest-session.json` with `"backend": "pi"`.

### Sharing a GPU with Ollama

When `PI_PROVIDER=ollama`, the GPU queue worker and the pi orchestrator coordinate
through the shared `.neural-lab` state directory. Before launching a queued GPU
job, the worker requests the GPU and waits. The orchestrator interrupts and
closes its pi session, asks Ollama to unload its loaded models, and verifies that
Ollama reports no loaded models before acknowledging the request. The worker
does not launch the job if this confirmation is unavailable. The orchestrator
does not start another pi session while the worker holds the request. CPU-only
workers do not use this handoff.

Set `OLLAMA_BASE_URL` in `.env` to the base URL of the **same Ollama server**
used by the pi provider. The Compose default is `http://127.0.0.1:11434`.
The pi orchestrator uses host networking so it can reach a loopback-only Ollama
listener. This also permits it to reach other services listening on the host's
loopback interfaces; take that into account when configuring those services.
Changing
`OLLAMA_BASE_URL` configures the handoff, not pi's provider; configure pi to use
the same server separately. The handoff unloads all models reported by that
server, including models loaded by other clients, so use a dedicated server when
other clients must remain running. Other clients must not load models on the
shared GPU during a worker job.

Set `PI_PROVIDER=ollama` for both processes when running them outside Compose,
and give them the same state directory. If Ollama is unreachable, the URL is
incorrect, or the pi orchestrator is stopped, the job remains queued until the
handoff succeeds. Check the orchestrator logs and server configuration before
retrying. Recreate both Compose services after changing `.env`; an existing
container retains its original environment. The standalone `pi-auth.sh status`
and `manage doctor --planner pi` commands perform inference outside this
handooff; do not run them while a GPU experiment is active.

### Context handoff

Each turn is interrupted — and treated as a successful session outcome — when the
reported usage strictly exceeds **60% of the reported context window**
(`totalTokens * 5 > contextWindow * 3`), mirroring the Codex behavior. A session
that settles without verified usage telemetry, or whose telemetry is missing when
an update arrives, is aborted conservatively instead of being trusted.

### Budget gate (replaces subscription quota pausing)

There is no provider-side quota pause for API billing, so the pi orchestrator
pauses on local spend/session budgets stored in
`.neural-lab/notes/pi-budget.json`. The gate is conservative: missing or
unreadable state pauses the orchestrator.

| Knob | Default | Meaning |
|---|---|---|
| `NEURAL_LAB_PI_BUDGET_SECONDS` / `--pi-budget-seconds` | 14400 | Active session seconds per rolling 24-hour window |
| `NEURAL_LAB_PI_BUDGET_SESSIONS` / `--pi-budget-sessions` | 8 | Settled sessions per rolling 24-hour window |

The window resets automatically when `window_start` is older than a day. The
read-only gate treats a missing, invalid, or stale `window_start` as a fresh
window, so the gate reopens after a day even if no further session runs to
persist the reset (the reset is written back by the recorder on the next
session). Set both knobs in `.env` to make the Compose service enforce them.

## Security

- The auth directory is mounted into the planner, orchestrator, and doctor
  containers at `/pi`, never into the trusted GPU worker or a training container.
  The orchestrator can read the API key from this mount; the GPU worker cannot.
- No API key is passed through container environment variables; the pi CLI reads
  its own auth state from the dedicated mount (`PI_CODING_AGENT_DIR=/pi`).
- The planner and orchestrator containers use a read-only root filesystem (planner),
  dropped capabilities, no-new-privileges, CPU/memory/process limits, and bounded
  logs. The orchestrator keeps the same writable research container boundary as
  the Codex service.
- Untrusted plan validation, the worker sandbox, and the queue boundaries are
  unchanged from the Codex path. Generated code never receives credentials.
- The RPC client stores only fixed, sanitized diagnostic categories; raw CLI
  logs and secrets are never copied into handoff notes or journals.
