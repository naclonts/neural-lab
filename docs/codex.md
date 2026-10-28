# Use Codex with your ChatGPT subscription

The codex planner runs the official Codex CLI in a separate, CPU-only container.
It signs in with ChatGPT and requests a structured experiment proposal.
The supervisor then sends the resulting code to a separate GPU worker.

This uses Codex account access, not the OpenAI API planner. Codex usage counts
against your account's allowance and any applicable credits; it is not unlimited
24/7 model access. The project does not change account billing or credit settings.

## First-time setup on your laptop

```bash
git pull
bash scripts/setup.sh
docker compose pull
bash scripts/codex-login.sh
```

The login script builds the pinned Codex 0.165.0 image, then starts the official
device-code login flow. Open the displayed link in your browser and authenticate
with your ChatGPT account. Device-code login may need enabling in account/workspace
security settings. This interactive account login must happen on your own machine.

Check or clear the dedicated login with:

```bash
bash scripts/codex-login.sh status
bash scripts/codex-login.sh logout
```

Login data is stored in ~/.local/share/neural-lab/codex, outside the repository.
Treat that directory like a password. Do not commit it, copy it into experiments,
or share its contents. It is separate from your regular ~/.codex setup.
NEURAL_LAB_CODEX_HOME can select another dedicated directory for both the script
and supervisor; --codex-home overrides it for the supervisor.

The CLI uses writable file-based auth storage so token refresh can persist.
Credentials are handled by Codex itself; this project does not parse tokens or
call private ChatGPT endpoints.

## Run an experiment

```bash
python3 -m neural_lab.supervisor --planner codex --once
```

Run a bounded sequence:

```bash
python3 -m neural_lab.supervisor \
  --planner codex \
  --max-attempts 10 \
  --timeout-seconds 300 \
  --planning-timeout-seconds 600 \
  --idle-seconds 120
```

Without --model, Codex chooses its default available model. Set --model only to a
model available to your account. Planning and training have separate timeouts;
attempt and reserved-training-runtime limits persist in the existing journal.

## Permissions and account limits

The planner receives only a schema, selected experiment observations through stdin,
a temporary answer directory, and its dedicated login directory. It has no GPU,
Docker socket, personal home mount, training outputs, or database mount.

It ignores user config, uses a read-only Codex sandbox, disables approvals and
shell/image/browser/computer-use/plugin/app/subagent/code-mode tools, disables
web search, and returns JSON. The outer container has a read-only root and resource
limits. The first implementation uses Codex for planning only; it does not grant
Codex an interactive training shell or Docker control.

The writable login mount is trusted CLI state and must never be exposed to training
code. The worker uses the existing sandbox and never receives Codex credentials.
The planner container is labeled for the same restart recovery as workers.
Container cleanup failure halts the supervisor.

Login expiration, quota exhaustion, CLI errors, invalid output, and timeouts are
recorded as failed planning attempts. Existing exponential backoff and the
persistent failure circuit apply. This version does not reliably classify account
reset times or automatically resume at quota reset; check Codex usage/login and
follow the [loop recovery instructions](agent-loop.md) before resuming.

There is no automatic switch to API billing, account rotation, or quota bypass.
Normal Docker networking still permits LAN access. General storage, network, and
GPU hardening requirements described in agent-loop.md remain.

## Verification

CI tests command boundaries, structured response validation, shutdown, timeout,
cleanup failure, and absence of an API fallback using a simulated CLI.
It builds the real pinned Codex image and checks its flags/config without login.
It never uploads ChatGPT login data to public GitHub Actions or consumes your
subscription there. A live authenticated run remains to be tested on your laptop.

## Official references

- [Codex authentication](https://developers.openai.com/codex/auth)
- [Codex non-interactive execution](https://developers.openai.com/codex/noninteractive)
- [Using Codex with a ChatGPT plan](https://help.openai.com/en/articles/11369540-using-codex-with-your-chatgpt-plan)
