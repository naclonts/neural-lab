import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from neural_lab.orchestrator import AppServer, capacity_ok, context_fraction, main, rpc_snapshot, run_session


class OrchestratorTests(unittest.TestCase):
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

    def test_context_uses_reported_window_and_250k_handoff(self):
        def event(tokens, window=500_000):
            return {"params": {"tokenUsage": {"modelContextWindow": window,
                                                "last": {"totalTokens": tokens}}}}
        self.assertEqual(context_fraction(event(250_000)), 0.5)
        self.assertGreater(context_fraction(event(250_001)), 0.5)
        self.assertGreater(context_fraction(event(250_001, 475_000)), 0.5)
        with self.assertRaises(RuntimeError):
            context_fraction(event(100, 474_999))
        with self.assertRaises(RuntimeError):
            context_fraction(event(100, None))

    def test_live_context_rollover_interrupts_turn(self):
        class FakeServer:
            def __init__(self):
                self.calls = []
                self.messages = [
                    {"method": "item/completed", "params": {"threadId": "thread",
                     "turnId": "turn", "item": {"type": "agentMessage", "text": "Found a useful ablation."}}},
                    {"method": "thread/tokenUsage/updated", "params": {"threadId": "thread",
                     "turnId": "turn", "tokenUsage": {"modelContextWindow": 500_000,
                                                     "last": {"totalTokens": 250_001}}}},
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
        self.assertEqual(outcome["reason"], "context exceeded 250000 tokens")
        self.assertEqual(outcome["exit_code"], 0)
        self.assertEqual(handoff["last_tokens"], 250001)
        self.assertEqual(handoff["recent_agent_messages"], ["Found a useful ablation."])
        self.assertIn("turn/interrupt", [call[0] for call in fake.calls])
        self.assertEqual(fake.calls[2][1]["model"], "gpt-6.1-sol")
        self.assertTrue(fake.calls[2][1]["config"]["features"]["multi_agent"])
        self.assertIn("previous finding", fake.calls[3][1]["input"][0]["text"])
        self.assertEqual(fake.calls[3][1]["input"][0]["text_elements"], [])

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


if __name__ == "__main__":
    unittest.main()
