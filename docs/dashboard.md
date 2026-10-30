# Experiment observatory

Start the local, read-only dashboard independently of the research stack:

```bash
bash scripts/dashboard.sh start
```

Open **http://127.0.0.1:8766** on the lab machine. It refreshes every five
seconds and fits phone screens. Use `status`, `logs`, or `stop` with the same
script to manage it. Its `unless-stopped` policy resumes it after Docker restarts.
Stopping the dashboard does not stop the scientist or experiment worker.

The dashboard shows queue counts, the latest 100 experiments, hypotheses,
within-job baseline comparisons, all reported result metrics, timestamps,
recorded training devices when available, scientist messages/context, GPU
utilization/VRAM/temperature/power, free storage, and the latest 20 Markdown notes.
Search or filter experiments, then expand a row for details.

“Completed” means a script exited successfully and produced a valid result.
It does not mean its hypothesis held up. The starter jobs and candidate jobs
use different protocols; compare each experiment against its own controls.
Metrics are worker-reported observations, not an independent evaluation.

## Where observations come from

- `.neural-lab/queue.sqlite3`: job state and result summaries, opened read-only.
- `.neural-lab/queue-outputs/<job-id>/result.json`: saved metrics and summary.
- `.neural-lab/queue-outputs/<job-id>/details.json`: optional per-seed records,
  environment/device, and protocol details. The dashboard extracts the device.
- `.neural-lab/queue-outputs/<job-id>/visualizations/`: optional training visuals.
- `.neural-lab/notes/latest-session.json`: last scientist checkpoint and messages.
- `.neural-lab/notes/pause-state.json`: pause reason, when newer than the session.
- `.neural-lab/notes/*.md`: research notes, displayed as plain text.
- `nvidia-smi`: machine-wide GPU samples; it is not per-job attribution.

The scientist runs Codex in `neural-lab-research-orchestrator-1`. The worker
controller in `neural-lab-research-worker-1` makes no Codex calls and launches
separate disposable `neural-lab-<job-id>` training containers. Small experiments
can finish between GPU samples. Zero utilization while the queue is empty is
normal. CPU data generation/calibration and small CUDA kernels may also result
in low GPU utilization during a job. The saved training device is better evidence
of execution placement than one instantaneous utilization sample.

Scientist state is the **last reported checkpoint**, not process health.
A completed scientist session normally waits five minutes before the next one.
Old checkpoints are labeled with their age. Lost dashboard connectivity keeps
the last rendered observations and displays a warning.

## Local boundary

The container receives only a read-only state mount and NVIDIA utility access;
it has no Codex credentials, repository mount, Docker socket, or write endpoints.
Its filesystem is read-only, capabilities are dropped, and CPU/RAM/process limits
are set. The host publishes the dashboard only on `127.0.0.1:8766`.
Other trusted local processes and Docker network peers are outside its access
boundary. Research notes are private data and should stay off public hosting.

No dashboard password or public endpoint is configured. Remote access should go
through the authenticated private network described below. The dashboard does
not trust identity headers as its own authentication mechanism.

Without Docker, run `python3 -m neural_lab.dashboard` from the checkout for a
foreground loopback server. It uses only Python's standard library. NVIDIA
telemetry is optional. The container uses the explicit NVIDIA runtime, so this
machine's runtime compatibility configuration also applies; see
[research setup](orchestrator.md#nvidia-cdi-startup-failures).

## Later: private phone access with Tailscale

The intended path is **phone → Tailscale → HTTPS Serve → loopback dashboard**.
[Tailscale Serve](https://tailscale.com/docs/features/tailscale-serve) exposes a
local service within your tailnet. Install Tailscale on the lab machine and phone,
sign both into your tailnet, and complete any required device approval.
Tailscale is not installed or configured by the dashboard scripts.

Once both devices are enrolled:

1. Find the machine's exact MagicDNS HTTPS hostname, for example
   `lab-machine.your-tailnet.ts.net`.
2. Add `NEURAL_LAB_DASHBOARD_HOST=lab-machine.your-tailnet.ts.net` to private
   `.env` and run `bash scripts/dashboard.sh start`. This permits that HTTP Host
   header through the dashboard's hostname check; it is not an authorization rule.
3. Run `tailscale serve --bg http://127.0.0.1:8766` on the host. Follow its HTTPS
   enablement flow if needed. Use the HTTPS URL printed by Serve on the phone,
   with Tailscale connected. See the [Serve CLI reference](https://tailscale.com/docs/reference/tailscale-cli/serve).
4. Configure tailnet access rules for the phone and HTTPS port 443. For a
   phone-only rule, use the phone's Tailscale IP, not its changing cellular or
   Wi-Fi address. A user/email source instead permits that user's eligible devices.

Example grant fragment using placeholder device IPs, to adapt in your existing
tailnet policy (not paste over your entire policy):

```json
{
  "grants": [
    {
      "src": ["100.101.102.103/32"],
      "dst": ["100.104.105.106/32"],
      "ip": ["tcp:443"]
    }
  ]
}
```

Here `src` is the phone and `dst` is the lab machine. Review existing broad ACLs
and grants as well: adding a narrow allow rule does not revoke other allowed
access. Test the final policy with the phone allowed and an unrelated tailnet
device denied. [Grants syntax and selectors](https://tailscale.com/docs/reference/syntax/grants).

This provides authenticated-device access without a separate web login. Use
**Serve**, not public Funnel, for this private monitoring setup. No router port
forwarding or public IP exposure is needed. To disable this specific Serve
endpoint later: `tailscale serve --https=443 off`.
