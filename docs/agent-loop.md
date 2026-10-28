# Agent loop

The first supervisor supports three planners:

- demo: a deterministic MLP architecture search on a fixed synthetic task, with
  held-out evaluation against a linear baseline.
- codex: the official Codex CLI with ChatGPT login in a separate planning container.
  See [subscription setup](codex.md); no API key is needed.
- api: an OpenAI-compatible Chat Completions endpoint proposes hypotheses and
  standalone training scripts using recent experiment outcomes.

The host supervisor uses Python 3.10+ and stdlib only. PyTorch runs in the worker.
The API planner is a one-request experiment planner, not an interactive coding
agent with an iterative shell session. Failed scripts inform the next proposal.

## Start small

After installing Docker, NVIDIA Container Toolkit, and running the GPU smoke test:

```bash
git pull
docker compose pull
python3 -m neural_lab.supervisor --once
```

The image must already be pulled. The runner uses pull=never so a lengthy image
download is not hidden inside a timed Docker control command.

This starts one disposable GPU container, trains a small candidate and a linear
baseline, saves plain JSON demo weights/config, attempts visualization, records
the outcome in SQLite, and removes the container. It needs no API key.

For CPU-only development, add --cpu. That disables the GPU device grant but does
not change the image; use the default PyTorch image or another trusted image
containing Python, PyTorch, and the standard library.

Run a bounded sequence:

```bash
python3 -m neural_lab.supervisor \
  --planner demo \
  --max-attempts 20 \
  --timeout-seconds 300 \
  --budget-seconds 7200 \
  --idle-seconds 60
```

Ctrl-C or SIGTERM requests shutdown and container removal. A planner HTTP call
may take up to its 120-second timeout to return before shutdown completes.
Docker control operations have 30-second timeouts; the worker runtime deadline
includes start overhead and is checked between control calls. It is not a
real-time hardware cutoff.

The loop runs in the foreground. No laptop service is installed or started by
this repository. The current defaults deliberately stop after a modest number
of attempts rather than running forever.

## Use an API planner

Keep the token in the host environment, outside experiment mounts:

```bash
read -rs -p 'Planner API key: ' NEURAL_LAB_API_KEY
export NEURAL_LAB_API_KEY
python3 -m neural_lab.supervisor \
  --planner api \
  --model YOUR_MODEL_NAME \
  --max-attempts 10
```

Choose a model supporting Chat Completions and max_tokens. The default endpoint is
https://api.openai.com/v1/chat/completions. Override --api-url with a compatible
HTTPS endpoint, or a loopback HTTP server such as
http://127.0.0.1:8000/v1/chat/completions. --api-key-env changes the environment
variable name; local servers can use a non-secret placeholder if required.

API output must be a JSON object containing exactly hypothesis and code.
No generated Docker commands or host-side tool calls are accepted. The host
never imports or executes the returned code. The planner has a 6000-output-token
cap and a 1 MiB HTTP response limit. Code is capped at 64 KiB.

Recent hypotheses, worker-reported metrics, and summaries are sent to the configured
planner endpoint. Use a local model or review experiment data if those observations
are private. Account tokens are not included in prompts or worker environments.

## State, recovery, and limits

The default state directory is .neural-lab/, ignored by Git:

```text
.neural-lab/
  journal.sqlite3       # host-only state, reservations, results, errors
  supervisor.lock      # one supervisor per state directory
  inputs/<run-id>/     # generated script, mounted read-only for one worker
  outputs/<run-id>/    # only this run's writable output mount
```

The worker also gets a read-only mount of the reviewed neural_lab package.
It receives neither the database nor other runs' directories.

A run transitions through planning, planned, running, and then succeeded, failed,
or interrupted. Before each planner call, the supervisor commits an attempt and
a full runtime reservation. Failed planning/training and crashes do not refund
these reservations. All limits apply across restarts of the same state directory,
not per day. They are conservative admission budgets, not a GPU-hour meter or an
exact-dollar API spending cap.

On startup, the supervisor acquires a file lock, removes containers bearing that
state directory's persisted owner label, and marks unfinished runs interrupted.
After repeated failures it backs off exponentially and pauses after three
consecutive failures by default. Cleanup failure halts immediately so another
worker is not launched while the previous one may still be alive.

Increasing --max-attempts, --budget-seconds, or --max-consecutive-failures is an
explicit operator decision. Deleting or changing the state directory resets its
budgets and recovery identity. Never delete state while its containers may exist.
Use one state directory per GPU to avoid separate supervisors competing for it.

JSON status records are printed to stdout. Training artifacts and generated
scripts remain available in state. Docker logs rotate while a container exists
and are deleted with it; scripts should save useful diagnostics under /output.
The journal stores bounded error types for planner/launcher exceptions, without
copying HTTP response bodies or credentials.

## Worker contract

A successful script exits zero and writes /output/result.json:

```json
{
  "metrics": {"validation_accuracy": 0.81, "baseline_accuracy": 0.73},
  "summary": "Held-out comparison, data provenance, seed, and limitations."
}
```

The supervisor rejects symlinks, special files, oversized results, non-finite
metrics, and invalid schemas. It does not load worker checkpoints on the host.
Metrics remain worker-reported observations; this is not an independent evaluation
service. API-generated code must implement its own baseline and held-out split.

The demo saves visualizations under outputs/<run-id>/visualizations/. The API
planner is instructed to attempt the same export. Export failure does not erase a
valid training result; the script should record the failure in its summary.

## Remaining work before unattended operation

The loop already enforces non-root workers, limited mounts, read-only roots,
CPU/RAM/process/log limits, runtime deadlines, single-instance locking,
persistent budgets, restart recovery, and a failure circuit breaker.

It still needs enforced internet egress restrictions, per-run storage quotas,
retention, GPU thermal/power monitoring, independent evaluation, and a separately
credentialed Hugging Face publisher. Default Docker networking can reach the
host/LAN. CPU/RAM limits do not cap VRAM, GPU power, or output disk usage.

Treat this as a bounded development loop. Do not call it a hardened 24/7
deployment or grant workers personal files or account credentials.

## Tests

GitHub Actions runs unit tests and real CPU Docker checks for mounts, credential
isolation, timeout removal, state recovery, and result validation. It also trains
a trusted tiny demo and checks its visualization outputs. NVIDIA behavior still
needs validation on the target laptop.
