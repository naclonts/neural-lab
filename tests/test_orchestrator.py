import hashlib
import json
import io
from collections import deque
from contextlib import contextmanager, redirect_stdout
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

from neural_lab.research_direction import RESEARCH_DIRECTION

from neural_lab.orchestrator import (AppServer, PiSession, PROMPT, budget_ok, capacity_ok,
                                     context_fraction, main, pi_provider_error,
                                     pi_usage, record_pi_budget,
                                     resolve_pi_context_window, rpc_snapshot,
                                     run_pi_session, run_session)


class OrchestratorTests(unittest.TestCase):
    def test_research_direction_includes_gpu_pipeline_policy(self):
        for keyword in ('GPU pipeline policy', 'RTX 2070', 'digest-pinned',
                        'Never enqueue an unvalidated candidate'):
            self.assertIn(keyword, RESEARCH_DIRECTION)
        self.assertIn(RESEARCH_DIRECTION, PROMPT)

    def test_catalog_diagnostics_store_only_fixed_safe_categories(self):
        server = AppServer.__new__(AppServer)
        server.model_catalog_error = None
        server.process = SimpleNamespace(stderr=io.BytesIO(
            b'unrelated private-sentinel\n'
            b'failed to refresh available models: request timed out private-sentinel\n'
        ))
        server._read_diagnostics()
        self.assertEqual(server.model_catalog_error, 'model catalog refresh timed out')
        self.assertNotIn('private-sentinel', server.model_catalog_error)

    def test_usage_gate_is_conservative_at_75_percent(self):
        snapshot = {
            "ordinaryUsageAllowed": True,
            "rateLimitsByLimitId": {
                "codex": {"primary": {"usedPercent": 74}, "secondary": {"usedPercent": 20}},
                "other": {"primary": {"usedPercent": 40}},
            },
        }
        self.assertTrue(capacity_ok(snapshot)[0])
        snapshot["rateLimitsByLimitId"]["codex"]["primary"]["usedPercent"] = 75
        self.assertFalse(capacity_ok(snapshot)[0])
        snapshot["rateLimitsByLimitId"]["codex"]["primary"]["usedPercent"] = 10
        snapshot["ordinaryUsageAllowed"] = None
        self.assertTrue(capacity_ok(snapshot)[0])
        del snapshot["ordinaryUsageAllowed"]
        self.assertTrue(capacity_ok(snapshot)[0])
        snapshot["ordinaryUsageAllowed"] = True
        snapshot["rateLimitsByLimitId"]["codex"]["primary"]["usedPercent"] = True
        self.assertTrue(capacity_ok(snapshot)[0])  # other valid windows remain
        snapshot["rateLimitsByLimitId"] = {"codex": {"primary": {"usedPercent": float("nan")}}}
        self.assertFalse(capacity_ok(snapshot)[0])

    def test_weekly_gate_uses_95_percent_independently_of_five_hour_gate(self):
        snapshot = {"rateLimits": {"primary": {"usedPercent": 7},
                                   "secondary": {"usedPercent": 94}}}
        self.assertTrue(capacity_ok(snapshot)[0])
        snapshot["rateLimits"]["secondary"]["usedPercent"] = 95
        permitted, reason = capacity_ok(snapshot)
        self.assertFalse(permitted)
        self.assertIn("weekly usage 95% (pause at 95%)", reason)
        snapshot["rateLimits"]["secondary"]["usedPercent"] = 76
        self.assertTrue(capacity_ok(snapshot)[0])
        snapshot["rateLimits"]["primary"]["usedPercent"] = 75
        permitted, reason = capacity_ok(snapshot)
        self.assertFalse(permitted)
        self.assertIn("5-hour usage 75% (pause at 75%)", reason)

    def test_rpc_waits_for_initialize_then_parses_batched_replies(self):
        server = r'''
import json, sys
init = json.loads(sys.stdin.readline())
assert init["method"] == "initialize"
print(json.dumps({"id": 1, "result": {}}), flush=True)
requests = [json.loads(sys.stdin.readline()) for _ in range(3)]
assert [item["method"] for item in requests] == [
    "initialized", "account/read", "account/rateLimits/read"]
sys.stdout.write(json.dumps({"id": 2, "result": {"account": {"type": "chatgpt"}}}) + "\n")
sys.stdout.write(json.dumps({"id": 3, "result": {
    "ordinaryUsageAllowed": True,
    "rateLimits": {"primary": {"usedPercent": 15}}
}}) + "\n")
sys.stdout.flush()
'''
        real_popen = subprocess.Popen

        def fake_popen(_command, **kwargs):
            return real_popen([sys.executable, "-u", "-c", server], **kwargs)

        with patch("neural_lab.orchestrator.subprocess.Popen", side_effect=fake_popen):
            snapshot = rpc_snapshot(timeout=5)
        self.assertTrue(capacity_ok(snapshot)[0])

    def test_context_accepts_smaller_windows_and_validates_telemetry(self):
        def event(tokens, window=500_000):
            return {"params": {"tokenUsage": {"modelContextWindow": window,
                                                "last": {"totalTokens": tokens}}}}
        self.assertEqual(context_fraction(event(155_040, 258_400)), 0.6)
        self.assertGreater(context_fraction(event(155_041, 258_400)), 0.6)
        self.assertEqual(context_fraction(event(300_000)), 0.6)
        for window in (None, 0, -1, True, "500000"):
            with self.assertRaises(RuntimeError):
                context_fraction(event(100, window))
        for tokens in (-1, None, True):
            with self.assertRaises(RuntimeError):
                context_fraction(event(tokens))

    def test_live_context_rollover_interrupts_turn(self):
        class FakeServer:
            def __init__(self):
                self.calls = []
                self.messages = [
                    {"method": "item/completed", "params": {"threadId": "thread",
                     "turnId": "turn", "item": {"type": "agentMessage", "text": "Found a useful ablation."}}},
                    {"method": "thread/tokenUsage/updated", "params": {"threadId": "thread",
                     "turnId": "turn", "tokenUsage": {"modelContextWindow": 258_400,
                                                     "last": {"totalTokens": 155_040}}}},
                    {"method": "thread/tokenUsage/updated", "params": {"threadId": "thread",
                     "turnId": "turn", "tokenUsage": {"modelContextWindow": 258_400,
                                                     "last": {"totalTokens": 155_041}}}},
                    {"method": "turn/completed", "params": {"threadId": "thread",
                     "turn": {"id": "turn", "status": "interrupted"}}},
                ]

            def request(self, method, params=None):
                self.calls.append((method, params))
                if method == "thread/start":
                    return {"thread": {"id": "thread"}, "modelProvider": "openai"}
                if method == "turn/start":
                    return {"turn": {"id": "turn"}}
                return {}

            def notify(self, method):
                self.calls.append((method, None))

            def send(self, method, params):
                self.calls.append((method, params))
                return 10

            def receive(self, timeout):
                # The exact 60% boundary must not send an interruption.
                if len(self.messages) == 2:
                    assert not any(call[0] == "turn/interrupt" for call in self.calls)
                return self.messages.pop(0)

            def close(self):
                pass

        fake = FakeServer()
        with tempfile.TemporaryDirectory() as root:
            notes = Path(root) / ".neural-lab" / "notes"
            notes.mkdir(parents=True)
            (notes / "latest-session.json").write_text(json.dumps({"reason": "previous finding"}))
            with patch("neural_lab.orchestrator.AppServer", return_value=fake):
                outcome = run_session("gpt-6.1-sol", Path(root), "test prompt")
            handoff = json.loads((notes / "latest-session.json").read_text())
        self.assertEqual(outcome["reason"], "context exceeded 60% of reported window")
        self.assertEqual(outcome["exit_code"], 0)
        self.assertEqual(handoff["last_tokens"], 155041)
        self.assertEqual(handoff["model_context_window"], 258400)
        self.assertEqual(handoff["context_handoff_tokens"], 155040)
        self.assertEqual(handoff["context_handoff_fraction"], 0.6)
        self.assertEqual(handoff["recent_agent_messages"], ["Found a useful ablation."])
        self.assertIn("turn/interrupt", [call[0] for call in fake.calls])
        self.assertEqual(fake.calls[2][1]["model"], "gpt-6.1-sol")
        self.assertTrue(fake.calls[2][1]["config"]["features"]["multi_agent"])
        self.assertIn("previous finding", fake.calls[3][1]["input"][0]["text"])
        self.assertEqual(fake.calls[3][1]["input"][0]["text_elements"], [])

    def test_rejected_window_preserves_observation_and_catalog_failure(self):
        class FakeServer:
            model_catalog_error = "model catalog refresh timed out"

            def __init__(self):
                self.calls = []
                self.messages = [
                    {"method": "thread/tokenUsage/updated", "params": {"threadId": "thread",
                     "turnId": "turn", "tokenUsage": {"modelContextWindow": 0,
                                                         "last": {"totalTokens": 16644}}}},
                    {"method": "turn/completed", "params": {"threadId": "thread",
                     "turn": {"id": "turn", "status": "interrupted"}}},
                ]

            def request(self, method, params=None):
                if method == "thread/start":
                    return {"thread": {"id": "thread"}, "modelProvider": "openai"}
                if method == "turn/start":
                    return {"turn": {"id": "turn"}}
                return {}

            def notify(self, method):
                pass

            def send(self, method, params):
                self.calls.append(method)

            def receive(self, timeout):
                return self.messages.pop(0)

            def close(self):
                pass

        fake = FakeServer()
        with tempfile.TemporaryDirectory() as root:
            with patch("neural_lab.orchestrator.AppServer", return_value=fake):
                result = run_session("gpt-6.1-sol", Path(root), "test")
            handoff = json.loads((Path(root) / ".neural-lab/notes/latest-session.json").read_text())
        self.assertEqual(result["exit_code"], 1)
        self.assertIn("reported 0", result["reason"])
        self.assertIn("positive context window", result["reason"])
        self.assertIn("catalog refresh timed out", result["reason"])
        self.assertEqual(result["last_tokens"], 16644)
        self.assertEqual(result["model_context_window"], 0)
        self.assertFalse(result["context_window_verified"])
        self.assertEqual(handoff["model_catalog_error"], fake.model_catalog_error)
        self.assertEqual(fake.calls, ["turn/interrupt"])

    def test_live_subscription_gate_interrupts_turn(self):
        class FakeServer:
            def __init__(self):
                self.calls = []
                self.messages = [
                    {"id": 10, "result": {"ordinaryUsageAllowed": True,
                                          "rateLimits": {"primary": {"usedPercent": 75}}}},
                    {"method": "turn/completed", "params": {"threadId": "thread",
                     "turn": {"id": "turn", "status": "interrupted"}}},
                ]

            def request(self, method, params=None):
                if method == "thread/start":
                    return {"thread": {"id": "thread"}, "modelProvider": "openai"}
                if method == "turn/start":
                    return {"turn": {"id": "turn"}}
                return {}

            def notify(self, method):
                pass

            def send(self, method, params):
                self.calls.append(method)
                return 10

            def receive(self, timeout):
                return self.messages.pop(0)

            def close(self):
                pass

        fake = FakeServer()
        with tempfile.TemporaryDirectory() as root:
            with patch("neural_lab.orchestrator.AppServer", return_value=fake):
                outcome = run_session("gpt-6.1-sol", Path(root), "test", usage_poll_seconds=0)
        self.assertIn("75%", outcome["reason"])
        self.assertEqual(fake.calls, ["account/rateLimits/read", "turn/interrupt"])

    def test_poll_timeout_interrupts_even_without_new_message(self):
        class FakeServer:
            def __init__(self):
                self.calls = []
                self.waited = False

            def request(self, method, params=None):
                if method == "thread/start":
                    return {"thread": {"id": "thread"}, "modelProvider": "openai"}
                if method == "turn/start":
                    return {"turn": {"id": "turn"}}
                return {}

            def notify(self, method):
                pass

            def send(self, method, params):
                self.calls.append(method)
                return 10 if method == "account/rateLimits/read" else 11

            def receive(self, timeout):
                if not self.waited:
                    self.waited = True
                    time.sleep(0.02)
                    return None
                return {"method": "turn/completed", "params": {"threadId": "thread",
                        "turn": {"id": "turn", "status": "interrupted"}}}

            def close(self):
                pass

        fake = FakeServer()
        with tempfile.TemporaryDirectory() as root:
            with patch("neural_lab.orchestrator.AppServer", return_value=fake):
                outcome = run_session("gpt-6.1-sol", Path(root), "test", usage_poll_seconds=0,
                                      usage_timeout_seconds=0.01)
            pause = json.loads((Path(root) / ".neural-lab/notes/latest-session.json").read_text())
        self.assertEqual(fake.calls, ["account/rateLimits/read", "turn/interrupt"])
        self.assertEqual(outcome["reason"], "subscription usage check timed out")
        self.assertEqual(pause["reason"], "subscription usage check timed out")

    def test_sparse_usage_update_refetches_and_interrupts(self):
        class FakeServer:
            def __init__(self):
                self.calls = []
                self.messages = [
                    {"method": "account/rateLimits/updated", "params": {
                        "rateLimits": {"credits": {"hasCredits": False, "unlimited": False}}}},
                    {"id": 10, "result": {"rateLimits": {"primary": {"usedPercent": 75}}}},
                    {"method": "turn/completed", "params": {"threadId": "thread",
                        "turn": {"id": "turn", "status": "interrupted"}}},
                ]

            def request(self, method, params=None):
                if method == "thread/start":
                    return {"thread": {"id": "thread"}, "modelProvider": "openai"}
                if method == "turn/start":
                    return {"turn": {"id": "turn"}}
                return {}

            def notify(self, method):
                pass

            def send(self, method, params):
                self.calls.append(method)
                return 10

            def receive(self, timeout):
                return self.messages.pop(0)

            def close(self):
                pass

        fake = FakeServer()
        with tempfile.TemporaryDirectory() as root:
            with patch("neural_lab.orchestrator.AppServer", return_value=fake):
                result = run_session("gpt-6.1-sol", Path(root), "test")
        self.assertIn("75%", result["reason"])
        self.assertEqual(fake.calls, ["account/rateLimits/read", "turn/interrupt"])

    def test_paused_once_does_not_start_codex_session(self):
        with tempfile.TemporaryDirectory() as root:
            with patch("neural_lab.orchestrator.rpc_snapshot", return_value={
                "ordinaryUsageAllowed": True,
                "rateLimits": {"primary": {"usedPercent": 75}},
            }):
                with patch("neural_lab.orchestrator.run_session") as run:
                    result = main(["--model", "verified-model", "--repo", root, "--once"])
            self.assertEqual(result, 1)
            run.assert_not_called()
            self.assertTrue((Path(root) / ".neural-lab" / "notes").is_dir())

    def test_paused_log_line_names_backend_and_model(self):
        with tempfile.TemporaryDirectory() as root:
            with patch("neural_lab.orchestrator.rpc_snapshot", return_value={
                "ordinaryUsageAllowed": True,
                "rateLimits": {"primary": {"usedPercent": 75}},
            }):
                with patch("sys.stdout", new=io.StringIO()) as out:
                    main(["--model", "verified-model", "--repo", root, "--once"])
            record = json.loads(out.getvalue().strip().splitlines()[-1])
            self.assertEqual(record["orchestrator"], "paused")
            self.assertEqual(record["backend"], "codex")
            self.assertEqual(record["model"], "verified-model")

    def test_starting_and_finished_log_lines_name_backend_and_model(self):
        with tempfile.TemporaryDirectory() as root:
            with patch("neural_lab.orchestrator.rpc_snapshot", return_value={
                "ordinaryUsageAllowed": True,
                "rateLimitsByLimitId": {
                    "codex": {"primary": {"usedPercent": 10}, "secondary": {"usedPercent": 5}},
                },
            }):
                with patch("neural_lab.orchestrator.run_session",
                           return_value={"exit_code": 0, "reason": "ok"}):
                    with patch("sys.stdout", new=io.StringIO()) as out:
                        code = main(["--backend", "codex", "--model", "gpt-6.1-sol",
                                     "--repo", root, "--once"])
            self.assertEqual(code, 0)
            lines = [json.loads(line) for line in out.getvalue().strip().splitlines()]
            starting = next(r for r in lines if r["orchestrator"] == "starting")
            self.assertEqual(starting["backend"], "codex")
            self.assertEqual(starting["model"], "gpt-6.1-sol")
            finished = next(r for r in lines if r["orchestrator"] == "session_finished")
            self.assertEqual(finished["backend"], "codex")
            self.assertEqual(finished["model"], "gpt-6.1-sol")
            self.assertEqual(finished["exit_code"], 0)

    def test_session_error_outcome_log_line_names_backend_and_model(self):
        with tempfile.TemporaryDirectory() as root:
            with patch("neural_lab.orchestrator.rpc_snapshot", return_value={
                "ordinaryUsageAllowed": True,
                "rateLimitsByLimitId": {
                    "codex": {"primary": {"usedPercent": 10}, "secondary": {"usedPercent": 5}},
                },
            }):
                with patch("neural_lab.orchestrator.run_session",
                           side_effect=RuntimeError("boom")):
                    with patch("sys.stdout", new=io.StringIO()) as out:
                        code = main(["--backend", "codex", "--model", "gpt-6.1-sol",
                                     "--repo", root, "--once"])
            self.assertEqual(code, 1)
            lines = [json.loads(line) for line in out.getvalue().strip().splitlines()]
            finished = next(r for r in lines if r["orchestrator"] == "session_finished")
            self.assertEqual(finished["backend"], "codex")
            self.assertEqual(finished["model"], "gpt-6.1-sol")
            self.assertEqual(finished["exit_code"], 1)
            self.assertIn("session error", finished["reason"])


@unittest.skipUnless(os.environ.get("NEURAL_LAB_CODEX_PROTOCOL_TESTS") == "1", "Real Codex CLI protocol opt-in")
class RealProtocolTests(unittest.TestCase):
    def test_authless_handshake_and_thread_creation_without_inference(self):
        with tempfile.TemporaryDirectory(prefix="neural-lab-protocol-") as home:
            with patch.dict(os.environ, {"CODEX_HOME": home}):
                server = AppServer()
                try:
                    server.request("initialize", {
                        "clientInfo": {"name": "neural-lab-protocol-test", "version": "1"},
                        "capabilities": {"experimentalApi": True},
                    })
                    server.notify("initialized")
                    self.assertIsNone(server.request("account/read", {"refreshToken": False}).get("account"))
                    thread = server.request("thread/start", {
                        "cwd": str(Path.cwd()), "model": "gpt-6.1-sol", "modelProvider": "openai",
                        "approvalPolicy": "never", "sandbox": "danger-full-access", "ephemeral": True,
                        "config": {"forced_login_method": "chatgpt", "model_provider": "openai",
                                   "features": {"multi_agent": True}, "web_search": "live"},
                    })
                    self.assertTrue(thread["thread"]["id"])
                    self.assertEqual(thread["modelProvider"], "openai")
                    # No turn/start: this check never invokes a model or uses account capacity.
                finally:
                    server.close()


class PiTransportTests(unittest.TestCase):
    def test_budget_is_conservative_when_state_is_missing_or_unreadable(self):
        with tempfile.TemporaryDirectory() as root:
            permitted, reason = budget_ok(Path(root) / "missing.json", max_seconds=100, max_sessions=2)
            self.assertFalse(permitted)
            self.assertEqual(reason, "pi usage budget state unavailable")
            path = Path(root) / "pi-budget.json"
            path.write_text("{not json")
            self.assertEqual(budget_ok(path, max_seconds=100, max_sessions=2),
                             (False, "pi usage budget state unavailable"))
            path.write_text(json.dumps({"sessions": "2", "active_seconds": 0}))
            self.assertFalse(budget_ok(path, max_seconds=100, max_sessions=2)[0])

    def test_budget_blocks_at_session_and_second_limits(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "pi-budget.json"
            fresh = time.time()
            path.write_text(json.dumps({"sessions": 1, "active_seconds": 10, "window_start": fresh}))
            permitted, reason = budget_ok(path, max_seconds=100, max_sessions=2)
            self.assertTrue(permitted)
            self.assertIn("1/2", reason)
            path.write_text(json.dumps({"sessions": 2, "active_seconds": 10, "window_start": fresh}))
            permitted, reason = budget_ok(path, max_seconds=100, max_sessions=2)
            self.assertFalse(permitted)
            self.assertIn("session budget", reason)
            path.write_text(json.dumps({"sessions": 1, "active_seconds": 100, "window_start": fresh}))
            permitted, reason = budget_ok(path, max_seconds=100, max_sessions=2)
            self.assertFalse(permitted)
            self.assertIn("active-second budget", reason)

    def test_budget_gate_reopens_after_a_stale_or_invalid_window(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "pi-budget.json"
            path.write_text(json.dumps({
                "sessions": 9, "active_seconds": 99999,
                "window_start": time.time() - 3 * 86400,
            }))
            permitted, reason = budget_ok(path, max_seconds=100, max_sessions=2)
            self.assertTrue(permitted)
            self.assertIn("window reset", reason)
            self.assertIn("0/2 sessions", reason)
            self.assertIn("0/100 active seconds", reason)
            # A missing or invalid window_start is treated the same way.
            for state in ({"sessions": 9, "active_seconds": 99999},
                          {"sessions": 9, "active_seconds": 99999, "window_start": "oops"}):
                path.write_text(json.dumps(state))
                permitted, reason = budget_ok(path, max_seconds=100, max_sessions=2)
                self.assertTrue(permitted)
                self.assertIn("window reset", reason)
            # record_pi_budget persists the reset when the next session ends.
            record_pi_budget(Path(root), 7.0)
            persisted = json.loads(
                (Path(root) / ".neural-lab" / "notes" / "pi-budget.json").read_text())
            self.assertEqual(persisted["sessions"], 1)
            self.assertEqual(persisted["active_seconds"], 7.0)
            self.assertGreater(persisted["window_start"], time.time() - 60)

    def test_record_pi_budget_degrades_gracefully_on_hand_typed_state(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / ".neural-lab" / "notes" / "pi-budget.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({"sessions": "2", "active_seconds": "lots"}))
            record_pi_budget(Path(root), 5.0)
            state = json.loads(path.read_text())
            self.assertEqual(state["sessions"], 1)
            self.assertEqual(state["active_seconds"], 5.0)

    # --- pi context-window resolution helpers -------------------------------
    # pi 1.5.0 RPC events never carry usage.contextWindow, so every fake session
    # below emits realistic events WITHOUT that field and the window comes from
    # a temp models-store.json catalog or the env override instead.

    @staticmethod
    def write_pi_catalog(root, model="test-provider/pi-model", window=1000):
        provider, _, model_id = model.partition("/")
        (Path(root) / "models-store.json").write_text(json.dumps(
            {provider: {"models": [{"id": model_id, "contextWindow": window,
                                    "maxTokens": window}]}}))

    @contextmanager
    def pi_window_env(self, catalog_root=None, override=None,
                      model="test-provider/pi-model", window=1000):
        """Pin the window-resolution env vars for one run_pi_session call."""
        saved = {key: os.environ.pop(key, None)
                 for key in ("PI_CODING_AGENT_DIR", "NEURAL_LAB_PI_CONTEXT_WINDOW")}
        try:
            if catalog_root is not None:
                self.write_pi_catalog(catalog_root, model=model, window=window)
                os.environ["PI_CODING_AGENT_DIR"] = str(catalog_root)
            if override is not None:
                os.environ["NEURAL_LAB_PI_CONTEXT_WINDOW"] = override
            yield
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_pi_usage_reports_used_tokens_and_window_independently(self):
        # Realistic pi 1.5.0 shape: usage without contextWindow.
        self.assertEqual(pi_usage({"usage": {"totalTokens": 42}}), (42, None))
        # Future-proof shape: the event carries its own window.
        self.assertEqual(pi_usage({"usage": {"totalTokens": 42, "contextWindow": 1000}}),
                         (42, 1000))
        # Invalid fields degrade independently instead of rejecting both.
        self.assertEqual(pi_usage({"usage": {"totalTokens": -1, "contextWindow": 1000}}),
                         (None, 1000))
        self.assertEqual(pi_usage({"usage": {"totalTokens": 42, "contextWindow": 0}}),
                         (42, None))
        self.assertEqual(pi_usage({"usage": {}}), (None, None))
        self.assertEqual(pi_usage({}), (None, None))

    def test_resolve_pi_context_window_reads_catalog_and_env_override(self):
        with tempfile.TemporaryDirectory() as root:
            model = "fireworks/accounts/fireworks/models/glm-5p3-flash"
            with self.pi_window_env(root, model=model, window=1048573):
                self.assertEqual(resolve_pi_context_window(model), 1048573)
            # Missing catalog file, provider, entry, or field all resolve to None.
            with self.pi_window_env(None):
                self.assertIsNone(resolve_pi_context_window(model))
            with self.pi_window_env(root):
                self.assertIsNone(resolve_pi_context_window("fireworks/accounts/unknown"))
                self.assertIsNone(resolve_pi_context_window("no-slash-model"))

    def test_resolve_pi_context_window_env_override_wins_and_invalid_ignored(self):
        with tempfile.TemporaryDirectory() as root:
            model = "test-provider/pi-model"
            self.write_pi_catalog(root, model=model, window=1000)
            with self.pi_window_env(root, override="777"):
                self.assertEqual(resolve_pi_context_window(model), 777)
            # An invalid override is ignored (with a warning) and never crashes;
            # the catalog value is used as the fallback.
            stdout = io.StringIO()
            with self.pi_window_env(root, override="not-a-number"), redirect_stdout(stdout):
                self.assertEqual(resolve_pi_context_window(model), 1000)
            self.assertIn("invalid NEURAL_LAB_PI_CONTEXT_WINDOW ignored", stdout.getvalue())
            # A valid override needs no catalog at all.
            with self.pi_window_env(None, override="5000"):
                self.assertEqual(resolve_pi_context_window(model), 5000)

    def test_pi_budget_is_recorded_even_when_the_session_raises(self):
        with tempfile.TemporaryDirectory() as root:
            budget = Path(root) / ".neural-lab" / "notes" / "pi-budget.json"
            budget.parent.mkdir(parents=True)
            budget.write_text(json.dumps(
                {"sessions": 0, "active_seconds": 0, "window_start": time.time()}))
            with patch("neural_lab.orchestrator.run_pi_session",
                       side_effect=RuntimeError("boom")):
                code = main(["--backend", "pi", "--model", "pi-model",
                             "--repo", root, "--once"])
            self.assertEqual(code, 1)
            state = json.loads(budget.read_text())
            self.assertEqual(state["sessions"], 1)
            self.assertGreaterEqual(state["active_seconds"], 0)

    def test_pi_model_argument_is_not_rejected_as_a_codex_model(self):
        with tempfile.TemporaryDirectory() as root:
            budget = Path(root) / ".neural-lab" / "notes" / "pi-budget.json"
            budget.parent.mkdir(parents=True)
            budget.write_text(json.dumps(
                {"sessions": 0, "active_seconds": 0, "window_start": time.time()}))
            with patch("neural_lab.orchestrator.run_pi_session",
                       return_value={"exit_code": 0, "reason": "pi session settled"}):
                code = main(["--backend", "pi", "--model", "anthropic/claude-sonnet-4.5",
                             "--repo", root, "--once"])
            self.assertEqual(code, 0)

    def test_run_pi_session_hard_exits_on_stop_even_while_streaming(self):
        class StreamingSession:
            def __init__(self):
                self.calls = []

            def send(self, record_type, fields=None):
                self.calls.append((record_type, fields))
                return "neural-lab-1"

            def notify(self, record_type, fields=None):
                self.calls.append((record_type, fields))

            def receive(self, timeout):
                # Simulates a continuously streaming pi process that never starves.
                # Realistic pi 1.5.0 wire shape: no contextWindow on the event.
                return {"type": "message_update", "usage": {"totalTokens": 10}}

            def close(self):
                pass

        stop = threading.Event()
        stop.set()
        fake = StreamingSession()
        with tempfile.TemporaryDirectory() as root:
            with self.pi_window_env(root), \
                    patch("neural_lab.orchestrator.PiSession", return_value=fake):
                outcome = run_pi_session("test-provider/pi-model", Path(root), "test",
                                         stop=stop)
        self.assertEqual(outcome["exit_code"], 1)
        self.assertEqual(outcome["reason"], "shutdown requested")
        self.assertIn("abort", [call[0] for call in fake.calls])

    def test_run_pi_session_surfaces_sanitized_auth_error(self):
        class AuthFailSession:
            def __init__(self):
                self.calls = []
                self.auth_error = "pi could not authenticate with the configured provider"
                self.messages = [
                    {"type": "response", "id": "neural-lab-1", "success": False},
                    {"type": "agent_settled", "aborted": True},
                ]

            def send(self, record_type, fields=None):
                self.calls.append((record_type, fields))
                return "neural-lab-1"

            def notify(self, record_type, fields=None):
                self.calls.append((record_type, fields))

            def receive(self, timeout):
                return self.messages.pop(0) if self.messages else None

            def close(self):
                pass

        fake = AuthFailSession()
        with tempfile.TemporaryDirectory() as root:
            with self.pi_window_env(root), \
                    patch("neural_lab.orchestrator.PiSession", return_value=fake):
                outcome = run_pi_session("test-provider/pi-model", Path(root), "test")
        self.assertEqual(outcome["exit_code"], 1)
        self.assertEqual(outcome["reason"],
                         "pi could not authenticate with the configured provider")
        self.assertIn("abort", [call[0] for call in fake.calls])

    def test_pi_session_spawns_rpc_process_group_and_closes_with_signals(self):
        process = Mock()
        process.pid = 4321
        process.stderr = io.BytesIO(b"")
        with patch("neural_lab.orchestrator.subprocess.Popen", return_value=process) as popen:
            session = PiSession("pi-model")
        args, kwargs = popen.call_args
        self.assertEqual(args[0][:4], ["pi", "--mode", "rpc", "--no-session"])
        self.assertEqual(args[0][-2:], ["--model", "pi-model"])
        self.assertTrue(kwargs.get("start_new_session"))
        self.assertIsNone(kwargs.get("env"))
        with patch("neural_lab.orchestrator.os.killpg") as killpg:
            session.close()
        process.stdin.close.assert_called_once()
        self.assertEqual(killpg.call_args.args[1], signal.SIGTERM)

    def test_pi_session_diagnostics_keep_only_fixed_safe_categories(self):
        session = PiSession.__new__(PiSession)
        session.auth_error = None
        session.process = SimpleNamespace(stderr=io.BytesIO(
            b'unrelated private-sentinel\n'
            b'error: unauthorized private-sentinel\n'
        ))
        session._read_diagnostics()
        self.assertEqual(session.auth_error, "pi could not authenticate with the configured provider")

    def test_pi_session_receive_tolerates_unparseable_records(self):
        read_fd, write_fd = os.pipe()
        os.write(write_fd, b'not-json\n{"type":"agent_start"}\n')
        session = PiSession.__new__(PiSession)
        session.buffer = bytearray()
        session.pending = deque()
        session.process = SimpleNamespace(stdout=os.fdopen(read_fd, "rb"))
        try:
            message = session.receive(1)
            self.assertEqual(message, {"type": "agent_start"})
            self.assertIsNone(session.receive(0.01))
        finally:
            session.process.stdout.close()
            os.close(write_fd)

    def test_run_pi_session_aborts_when_context_telemetry_is_missing(self):
        class FakeSession:
            def __init__(self):
                self.calls = []
                self.messages = [
                    {"type": "message_update", "usage": {}},
                    {"type": "agent_settled", "aborted": True},
                ]

            def send(self, record_type, fields=None):
                self.calls.append((record_type, fields))
                return "neural-lab-1"

            def notify(self, record_type, fields=None):
                self.calls.append((record_type, fields))

            def receive(self, timeout):
                return self.messages.pop(0) if self.messages else None

            def close(self):
                pass

        fake = FakeSession()
        with tempfile.TemporaryDirectory() as root:
            notes = Path(root) / ".neural-lab" / "notes"
            notes.mkdir(parents=True)
            (notes / "latest-session.json").write_text(json.dumps({"reason": "previous finding"}))
            with self.pi_window_env(None), \
                    patch("neural_lab.orchestrator.PiSession", return_value=fake):
                outcome = run_pi_session("test-provider/pi-model", Path(root), "test prompt")
            handoff = json.loads((Path(root) / ".neural-lab" / "notes" / "latest-session.json").read_text())
        self.assertEqual(outcome["exit_code"], 1)
        self.assertEqual(outcome["reason"], "pi context telemetry unavailable")
        self.assertFalse(outcome["context_window_verified"])
        self.assertIn("abort", [call[0] for call in fake.calls])
        self.assertIn("Previous session handoff", fake.calls[0][1]["message"])
        self.assertEqual(handoff["backend"], "pi")
        self.assertEqual(handoff["reason"], "pi context telemetry unavailable")

    def test_run_pi_session_skips_usage_less_transcript_events(self):
        """pi 1.5.0 emits message_end/message_update events for system and
        user transcript messages with no usage field before the assistant
        turn starts. Those must be skipped, not treated as telemetry loss;
        only assistant events (which always carry usage) are validated."""
        class FakeSession:
            def __init__(self):
                self.calls = []
                self.messages = [
                    # Real probe shape: system and user transcript events
                    # carry no usage field at all.
                    {"type": "message_end"},
                    {"type": "message_end"},
                    # The assistant turn always carries usage (streaming
                    # deltas report totalTokens 0..., the final real counts).
                    {"type": "message_update", "usage": {"totalTokens": 0}},
                    {"type": "message_update", "usage": {"totalTokens": 400}},
                    {"type": "message_end", "usage": {"totalTokens": 400}},
                    {"type": "turn_end", "usage": {"totalTokens": 400}},
                    {"type": "agent_settled", "aborted": False},
                ]

            def send(self, record_type, fields=None):
                self.calls.append((record_type, fields))
                return "neural-lab-1"

            def notify(self, record_type, fields=None):
                self.calls.append((record_type, fields))

            def receive(self, timeout):
                return self.messages.pop(0) if self.messages else None

            def close(self):
                pass

        fake = FakeSession()
        with tempfile.TemporaryDirectory() as root:
            # Window comes from the temp models-store.json catalog, as in
            # real pi 1.5.0 sessions (events never carry contextWindow).
            with self.pi_window_env(root), \
                    patch("neural_lab.orchestrator.PiSession", return_value=fake):
                outcome = run_pi_session("test-provider/pi-model", Path(root), "test")
        self.assertEqual(outcome["exit_code"], 0)
        self.assertEqual(outcome["reason"], "pi session settled")
        self.assertTrue(outcome["context_window_verified"])
        self.assertEqual(outcome["last_tokens"], 400)
        self.assertEqual(outcome["model_context_window"], 1000)
        self.assertNotIn("abort", [call[0] for call in fake.calls])

    def test_run_pi_session_aborts_when_window_is_unresolvable(self):
        """Used tokens arrive but neither the event nor the catalog/override
        supplies a window: the conservative abort must still fire."""
        class FakeSession:
            def __init__(self):
                self.calls = []
                self.messages = [
                    # Realistic pi 1.5.0 shape: totalTokens but no contextWindow.
                    {"type": "message_update", "usage": {"totalTokens": 123}},
                    {"type": "agent_settled", "aborted": True},
                ]

            def send(self, record_type, fields=None):
                self.calls.append((record_type, fields))
                return "neural-lab-1"

            def notify(self, record_type, fields=None):
                self.calls.append((record_type, fields))

            def receive(self, timeout):
                return self.messages.pop(0) if self.messages else None

            def close(self):
                pass

        fake = FakeSession()
        with tempfile.TemporaryDirectory() as root:
            with self.pi_window_env(None), \
                    patch("neural_lab.orchestrator.PiSession", return_value=fake):
                outcome = run_pi_session("test-provider/missing-model", Path(root), "test")
        self.assertEqual(outcome["exit_code"], 1)
        self.assertEqual(outcome["reason"], "pi context telemetry unavailable")
        self.assertFalse(outcome["context_window_verified"])
        self.assertIsNone(outcome["last_tokens"])
        self.assertIsNone(outcome["model_context_window"])
        self.assertIn("abort", [call[0] for call in fake.calls])

    def test_run_pi_session_resolves_window_from_catalog_when_events_lack_it(self):
        class FakeSession:
            def __init__(self):
                self.messages = [
                    {"type": "message_update", "usage": {"totalTokens": 400}},
                    {"type": "agent_settled", "aborted": False},
                ]

            def send(self, record_type, fields=None):
                return "neural-lab-1"

            def notify(self, record_type, fields=None):
                pass

            def receive(self, timeout):
                return self.messages.pop(0) if self.messages else None

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as root:
            with self.pi_window_env(root), \
                    patch("neural_lab.orchestrator.PiSession", return_value=FakeSession()):
                outcome = run_pi_session("test-provider/pi-model", Path(root), "test")
        self.assertEqual(outcome["exit_code"], 0)
        self.assertTrue(outcome["context_window_verified"])
        self.assertEqual(outcome["last_tokens"], 400)
        self.assertEqual(outcome["model_context_window"], 1000)
        self.assertEqual(outcome["context_fraction"], 0.4)
        self.assertAlmostEqual(outcome["context_handoff_tokens"], 600.0)

    def test_run_pi_session_env_override_wins_over_catalog(self):
        class FakeSession:
            def __init__(self):
                self.messages = [
                    {"type": "message_update", "usage": {"totalTokens": 400}},
                    {"type": "agent_settled", "aborted": False},
                ]

            def send(self, record_type, fields=None):
                return "neural-lab-1"

            def notify(self, record_type, fields=None):
                pass

            def receive(self, timeout):
                return self.messages.pop(0) if self.messages else None

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as root:
            with self.pi_window_env(root, override="777"), \
                    patch("neural_lab.orchestrator.PiSession", return_value=FakeSession()):
                outcome = run_pi_session("test-provider/pi-model", Path(root), "test")
        self.assertEqual(outcome["exit_code"], 0)
        self.assertEqual(outcome["model_context_window"], 777)
        self.assertAlmostEqual(outcome["context_handoff_tokens"], 777 * 0.6)

    def test_run_pi_session_ignores_invalid_env_override(self):
        """An invalid NEURAL_LAB_PI_CONTEXT_WINDOW must not crash the session;
        the catalog value is used as the fallback window."""
        class FakeSession:
            def __init__(self):
                self.messages = [
                    {"type": "message_update", "usage": {"totalTokens": 400}},
                    {"type": "agent_settled", "aborted": False},
                ]

            def send(self, record_type, fields=None):
                return "neural-lab-1"

            def notify(self, record_type, fields=None):
                pass

            def receive(self, timeout):
                return self.messages.pop(0) if self.messages else None

            def close(self):
                pass

        stdout = io.StringIO()
        with tempfile.TemporaryDirectory() as root:
            with self.pi_window_env(root, override="not-a-number"), \
                    redirect_stdout(stdout), \
                    patch("neural_lab.orchestrator.PiSession", return_value=FakeSession()):
                outcome = run_pi_session("test-provider/pi-model", Path(root), "test")
        self.assertIn("invalid NEURAL_LAB_PI_CONTEXT_WINDOW ignored", stdout.getvalue())
        self.assertEqual(outcome["exit_code"], 0)
        self.assertEqual(outcome["model_context_window"], 1000)

    def test_run_pi_session_prefers_event_window_when_present(self):
        """Future-proofing: an event-provided window beats the resolved one."""
        class FakeSession:
            def __init__(self):
                self.messages = [
                    {"type": "message_update",
                     "usage": {"totalTokens": 400, "contextWindow": 2000}},
                    {"type": "agent_settled", "aborted": False},
                ]

            def send(self, record_type, fields=None):
                return "neural-lab-1"

            def notify(self, record_type, fields=None):
                pass

            def receive(self, timeout):
                return self.messages.pop(0) if self.messages else None

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as root:
            with self.pi_window_env(root, override="777"), \
                    patch("neural_lab.orchestrator.PiSession", return_value=FakeSession()):
                outcome = run_pi_session("test-provider/pi-model", Path(root), "test")
        self.assertEqual(outcome["exit_code"], 0)
        self.assertEqual(outcome["model_context_window"], 2000)

    def test_run_pi_session_success_requires_settled_and_verified_usage(self):
        class FakeSession:
            def __init__(self, messages):
                self.calls = []
                self.messages = messages

            def send(self, record_type, fields=None):
                self.calls.append((record_type, fields))
                return "neural-lab-1"

            def notify(self, record_type, fields=None):
                self.calls.append((record_type, fields))

            def receive(self, timeout):
                return self.messages.pop(0) if self.messages else None

            def close(self):
                pass

        def event_from_catalog(tokens):
            # Realistic pi 1.5.0 wire shape: no contextWindow on the event.
            return {"type": "message_update", "usage": {"totalTokens": tokens}}

        def event_with_window(tokens, window=1000):
            # Future-proof shape: the event carries its own window.
            return {"type": "message_update",
                    "usage": {"totalTokens": tokens, "contextWindow": window}}

        with tempfile.TemporaryDirectory() as root:
            success_session = FakeSession([event_from_catalog(400),
                                           {"type": "agent_settled", "aborted": False}])
            with self.pi_window_env(root), \
                    patch("neural_lab.orchestrator.PiSession", return_value=success_session):
                outcome = run_pi_session("test-provider/pi-model", Path(root), "test")
            self.assertEqual(outcome["exit_code"], 0)
            self.assertEqual(outcome["reason"], "pi session settled")
            self.assertTrue(outcome["context_window_verified"])
            self.assertEqual(outcome["last_tokens"], 400)
            self.assertEqual(outcome["model_context_window"], 1000)
            self.assertAlmostEqual(outcome["context_handoff_tokens"], 600.0)
            self.assertEqual(success_session.calls[0][0], "prompt")
            self.assertNotIn("abort", [call[0] for call in success_session.calls])

            aborted_session = FakeSession([event_from_catalog(400),
                                           {"type": "agent_settled", "aborted": True}])
            with self.pi_window_env(root), \
                    patch("neural_lab.orchestrator.PiSession", return_value=aborted_session):
                outcome = run_pi_session("test-provider/pi-model", Path(root), "test")
            self.assertEqual(outcome["exit_code"], 1)
            # The agent aborted on its own; the controller only aborts on limits.
            self.assertNotIn("abort", [call[0] for call in aborted_session.calls])

            unverified_session = FakeSession([{"type": "agent_settled", "aborted": False}])
            with self.pi_window_env(root), \
                    patch("neural_lab.orchestrator.PiSession", return_value=unverified_session):
                outcome = run_pi_session("test-provider/pi-model", Path(root), "test")

            # Same scenario over the future-proof wire shape (event-provided
            # window, no catalog): verification must work identically.
            windowed_session = FakeSession([event_with_window(400),
                                            {"type": "agent_settled", "aborted": False}])
            with self.pi_window_env(None), \
                    patch("neural_lab.orchestrator.PiSession", return_value=windowed_session):
                windowed_outcome = run_pi_session("test-provider/pi-model", Path(root), "test")
            self.assertEqual(windowed_outcome["exit_code"], 0)
            self.assertTrue(windowed_outcome["context_window_verified"])
            self.assertEqual(windowed_outcome["model_context_window"], 1000)

        # A settled session without any usage event is never a success, even when
        # the agent reports a normal completion; there is nothing left to abort.
        self.assertEqual(unverified_session.messages, [])
        self.assertEqual(outcome["exit_code"], 1)
        self.assertEqual(outcome["reason"], "pi session settled")
        self.assertFalse(outcome["context_window_verified"])

    def test_run_pi_session_context_handoff_aborts_above_60_percent(self):
        class FakeSession:
            def __init__(self):
                self.calls = []
                self.messages = [
                    # Realistic pi 1.5.0 shape: the window comes from the catalog.
                    {"type": "message_update", "usage": {"totalTokens": 601}},
                    {"type": "agent_settled", "aborted": True},
                ]

            def send(self, record_type, fields=None):
                self.calls.append((record_type, fields))
                return "neural-lab-1"

            def notify(self, record_type, fields=None):
                self.calls.append((record_type, fields))

            def receive(self, timeout):
                return self.messages.pop(0) if self.messages else None

            def close(self):
                pass

        fake = FakeSession()
        with tempfile.TemporaryDirectory() as root:
            with self.pi_window_env(root), \
                    patch("neural_lab.orchestrator.PiSession", return_value=fake):
                outcome = run_pi_session("test-provider/pi-model", Path(root), "test")
            budget_state = Path(root) / ".neural-lab" / "notes" / "pi-budget.json"
            self.assertFalse(budget_state.exists())  # budget is recorded by main(), not the session
        self.assertEqual(outcome["exit_code"], 0)
        self.assertEqual(outcome["reason"], "context exceeded 60% of reported window")
        self.assertIn("abort", [call[0] for call in fake.calls])
        self.assertEqual(outcome["context_fraction"], 0.601)


class PiResilienceTests(unittest.TestCase):
    """Regression coverage for pi provider-error visibility, failed-session
    backoff, and the cold-catalog warning (pi 1.5.0 resilience fixes)."""

    @contextmanager
    def pi_window_env(self, catalog_root=None, override=None,
                      model="test-provider/pi-model", window=1000):
        saved = {key: os.environ.pop(key, None)
                 for key in ("PI_CODING_AGENT_DIR", "NEURAL_LAB_PI_CONTEXT_WINDOW")}
        try:
            if catalog_root is not None:
                PiTransportTests.write_pi_catalog(catalog_root, model=model, window=window)
                os.environ["PI_CODING_AGENT_DIR"] = str(catalog_root)
            if override is not None:
                os.environ["NEURAL_LAB_PI_CONTEXT_WINDOW"] = override
            yield
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    @staticmethod
    def provider_error_transcript(error_message):
        """Byte-faithful event shapes captured from a real pi 1.5.0 session
        whose provider returned HTTP 429 (/tmp/pi-repro/err429.stdout):
        only message_start/message_end/turn_end + agent_settled, with the
        synthesized assistant message carrying stopReason "error" and NO
        top-level usage on any event."""
        assistant = {"role": "assistant", "content": [], "api": "openai-completions",
                     "provider": "fireworks", "model": "accounts/fireworks/models/glm-5p3-flash",
                     "usage": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0,
                               "totalTokens": 0,
                               "cost": {"input": 0, "output": 0, "cacheRead": 0,
                                        "cacheWrite": 0, "total": 0}},
                     "stopReason": "error", "timestamp": 1794027211200,
                     "errorMessage": error_message, "durationMs": 32,
                     "thinkingLevel": "high"}
        return [
            {"type": "message_start", "message": assistant},
            {"type": "message_end", "message": assistant},
            {"type": "turn_end", "message": assistant, "toolResults": []},
            {"type": "agent_settled", "aborted": False},
        ]

    class FakeErrorSession:
        def __init__(self, messages):
            self.calls = []
            self.messages = messages

        def send(self, record_type, fields=None):
            self.calls.append((record_type, fields))
            return "neural-lab-1"

        def notify(self, record_type, fields=None):
            self.calls.append((record_type, fields))

        def receive(self, timeout):
            return self.messages.pop(0) if self.messages else None

        def close(self):
            pass

    def test_run_pi_session_surfaces_sanitized_provider_error(self):
        error_message = '429: {"code":429,"message":"mock 429: quota exceeded for model"}'
        fake = self.FakeErrorSession(self.provider_error_transcript(error_message))
        fingerprint = hashlib.sha256(error_message.encode("utf-8")).hexdigest()[:12]
        with tempfile.TemporaryDirectory() as root:
            with self.pi_window_env(root), \
                    patch("neural_lab.orchestrator.PiSession", return_value=fake):
                outcome = run_pi_session("test-provider/pi-model", Path(root), "test")
        self.assertEqual(outcome["exit_code"], 1)
        self.assertEqual(outcome["reason"], f"provider error (fp={fingerprint})")
        # The raw provider error text must never leak into the outcome.
        self.assertNotIn("429", outcome["reason"])
        self.assertNotIn("quota", outcome["reason"])
        self.assertIsNone(outcome["last_tokens"])
        self.assertFalse(outcome["context_window_verified"])
        # Abort semantics unchanged: the controller still aborts with the
        # bounded grace instead of letting the session linger.
        self.assertIn("abort", [call[0] for call in fake.calls])

    def test_pi_provider_error_detection_handles_event_level_fields(self):
        raw = "500: internal provider detail"
        fingerprint = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
        # Fields directly on the event (defensive shape) are detected too.
        event = {"type": "message_update", "role": "assistant",
                 "stopReason": "error", "errorMessage": raw}
        self.assertEqual(pi_provider_error(event), f"provider error (fp={fingerprint})")
        # An error with no message text degrades to the bare category.
        self.assertEqual(pi_provider_error({"role": "assistant", "stopReason": "error"}),
                         "provider error")
        self.assertEqual(pi_provider_error({"role": "assistant", "errorMessage": raw}),
                         f"provider error (fp={fingerprint})")
        # Normal assistant messages never match.
        self.assertIsNone(pi_provider_error({"role": "assistant", "stopReason": "stop"}))
        self.assertIsNone(pi_provider_error({"role": "user", "stopReason": "error"}))
        self.assertIsNone(pi_provider_error({"type": "agent_settled", "aborted": False}))

    def test_run_pi_session_warns_when_context_window_unresolved(self):
        fake = self.FakeErrorSession([{"type": "agent_settled", "aborted": False}])
        stdout = io.StringIO()
        with tempfile.TemporaryDirectory() as root:
            with self.pi_window_env(None), redirect_stdout(stdout), \
                    patch("neural_lab.orchestrator.PiSession", return_value=fake):
                run_pi_session("test-provider/pi-model", Path(root), "test")
        warnings = [json.loads(line) for line in stdout.getvalue().splitlines()
                    if line.strip()
                    and json.loads(line).get("reason") == "pi context window unresolved at session start"]
        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0]["orchestrator"], "warning")
        self.assertEqual(warnings[0]["model"], "test-provider/pi-model")

        # A resolved window emits no warning.
        fake = self.FakeErrorSession([{"type": "agent_settled", "aborted": False}])
        stdout = io.StringIO()
        with tempfile.TemporaryDirectory() as root:
            with self.pi_window_env(root), redirect_stdout(stdout), \
                    patch("neural_lab.orchestrator.PiSession", return_value=fake):
                run_pi_session("test-provider/pi-model", Path(root), "test")
        self.assertEqual(stdout.getvalue(), "")


class FakeOrchestratorClock:
    def __init__(self, start=1_000_000.0):
        self.now = start

    def monotonic(self):
        return self.now


class FakeOrchestratorStopEvent:
    """threading.Event stand-in for main(): wait() advances a fake clock
    instead of sleeping, so backoff schedules are testable with no delays,
    and it can simulate SIGTERM arriving mid-wait."""

    def __init__(self, clock, stop_at=None, set_on_wait=None):
        self.clock = clock
        self.waits = []
        self.stop_at = stop_at
        self.set_on_wait = set_on_wait
        self._set = False

    def is_set(self):
        return self._set

    def set(self):
        self._set = True

    def wait(self, timeout=None):
        self.waits.append(timeout)
        self.clock.now += timeout or 0.0
        if self.set_on_wait is not None and len(self.waits) >= self.set_on_wait:
            self._set = True
        if self.stop_at is not None and self.clock.now >= self.stop_at:
            self._set = True
        return self._set


class MainBackoffTests(unittest.TestCase):
    """Failed-session exponential backoff in main() for the pi backend."""

    def run_main(self, clock, stop_event, outcomes):
        calls = []

        def fake_run(model, repo, prompt, **kwargs):
            calls.append(len(stop_event.waits))
            return outcomes[len(calls) - 1]

        fake_time = SimpleNamespace(monotonic=clock.monotonic, time=time.time)
        with tempfile.TemporaryDirectory() as root:
            budget = Path(root) / ".neural-lab" / "notes" / "pi-budget.json"
            budget.parent.mkdir(parents=True)
            budget.write_text(json.dumps(
                {"sessions": 0, "active_seconds": 0, "window_start": time.time()}))
            with patch("neural_lab.orchestrator.threading.Event", return_value=stop_event), \
                    patch("neural_lab.orchestrator.time", fake_time), \
                    patch("neural_lab.orchestrator.run_pi_session", side_effect=fake_run):
                code = main(["--backend", "pi", "--model", "test-provider/pi-model",
                             "--repo", root, "--poll-seconds", "300"])
        calls.append(len(stop_event.waits))
        return code, calls

    def iteration_sleeps(self, stop_event, calls):
        sleeps = []
        for i in range(len(calls) - 1):
            sleeps.append(sum(stop_event.waits[calls[i]:calls[i + 1]]))
        return sleeps

    def test_consecutive_failures_back_off_exponentially_with_cap(self):
        clock = FakeOrchestratorClock()
        # 60+120+240+480+960+1800+1800 = 5460 fake seconds of backoff, then stop.
        stop_event = FakeOrchestratorStopEvent(clock, stop_at=clock.now + 5460)
        outcome = {"exit_code": 1, "reason": "provider error"}
        code, calls = self.run_main(clock, stop_event, [outcome] * 7)
        self.assertEqual(code, 0)  # clean shutdown, never hung
        self.assertEqual([round(total) for total in self.iteration_sleeps(stop_event, calls)],
                         [60, 120, 240, 480, 960, 1800, 1800])
        # The backoff sleep is sliced so the stop event is honored promptly.
        self.assertLessEqual(max(stop_event.waits), 1.0)

    def test_success_resets_backoff_and_normal_poll_continues(self):
        clock = FakeOrchestratorClock()
        # fail (60s backoff) -> success (normal 300s poll) -> fail (60s again).
        stop_event = FakeOrchestratorStopEvent(clock, stop_at=clock.now + 60 + 300 + 60)
        outcomes = [{"exit_code": 1, "reason": "provider error"},
                    {"exit_code": 0, "reason": "pi session settled"},
                    {"exit_code": 1, "reason": "provider error"}]
        code, calls = self.run_main(clock, stop_event, outcomes)
        self.assertEqual(code, 0)
        sleeps = self.iteration_sleeps(stop_event, calls)
        self.assertEqual([round(total) for total in sleeps], [60, 300, 60])
        # Successes use the plain --poll-seconds wait; failures use slices.
        self.assertEqual(stop_event.waits[calls[1]:calls[2]], [300])
        self.assertGreater(len(stop_event.waits[calls[0]:calls[1]]), 1)
        self.assertGreater(len(stop_event.waits[calls[2]:calls[3]]), 1)

    def test_stop_event_interrupts_the_backoff_sleep(self):
        clock = FakeOrchestratorClock()
        # SIGTERM arrives during the first backoff slice: the sleep must end
        # after that slice instead of running the full 60 seconds.
        stop_event = FakeOrchestratorStopEvent(clock, set_on_wait=1)
        outcome = {"exit_code": 1, "reason": "provider error"}
        code, calls = self.run_main(clock, stop_event, [outcome])
        self.assertEqual(code, 0)
        self.assertEqual(len(stop_event.waits), 1)
        self.assertLessEqual(stop_event.waits[0], 1.0)
        self.assertLess(clock.now - 1_000_000.0, 60)


if __name__ == "__main__":
    unittest.main()
