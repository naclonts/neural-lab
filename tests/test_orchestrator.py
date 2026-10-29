import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from neural_lab.orchestrator import capacity_ok, main, rpc_snapshot, run_session


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
        self.assertFalse(capacity_ok(snapshot)[0])
        del snapshot["ordinaryUsageAllowed"]
        self.assertFalse(capacity_ok(snapshot)[0])
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

    def test_exec_enables_subagents_and_forces_chatgpt_login(self):
        class FakeProcess:
            def __init__(self):
                self.stdin = self
                self.stdout = iter([json.dumps({"type": "turn.completed", "usage": {"input_tokens": 100}})])
                self.prompt = ""

            def write(self, value):
                self.prompt += value

            def close(self):
                pass

            def wait(self):
                return 0

        fake = FakeProcess()
        with patch("neural_lab.orchestrator.subprocess.Popen", return_value=fake) as launch:
            code, tokens = run_session("verified-model", 1_000_000, "/repo", "test prompt")
        command = launch.call_args.args[0]
        self.assertEqual((code, tokens), (0, 100))
        self.assertIn("multi_agent", command)
        self.assertIn("--ignore-user-config", command)
        self.assertIn('forced_login_method="chatgpt"', command)
        self.assertNotIn("--oss", command)
        self.assertEqual(fake.prompt, "test prompt")

    def test_paused_once_does_not_start_codex_session(self):
        with tempfile.TemporaryDirectory() as root:
            with patch("neural_lab.orchestrator.rpc_snapshot", return_value={
                "ordinaryUsageAllowed": True,
                "rateLimits": {"primary": {"usedPercent": 75}},
            }):
                with patch("neural_lab.orchestrator.run_session") as run:
                    result = main(["--model", "verified-model", "--context-tokens", "1000000",
                                   "--repo", root, "--once"])
            self.assertEqual(result, 1)
            run.assert_not_called()
            self.assertTrue((Path(root) / ".neural-lab" / "notes").is_dir())


if __name__ == "__main__":
    unittest.main()
