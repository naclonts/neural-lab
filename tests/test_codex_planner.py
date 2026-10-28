import json
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

from neural_lab.codex_planner import CODEX_IMAGE, CodexPlanner, DISABLED_FEATURES, PLAN_SCHEMA
from neural_lab.planner import PlannerError
from neural_lab.runner import CleanupError


PLAN = {"hypothesis": "Compare residual blocks", "code": "print('sandbox only')\n"}


class CodexPlannerTests(unittest.TestCase):
    def test_command_uses_subscription_and_keeps_personal_files_and_api_keys_out(self):
        with tempfile.TemporaryDirectory() as auth:
            planner = CodexPlanner("owner", auth_dir=auth)
            command = planner.command("plan-test", "/tmp/request", "/tmp/answer")
        joined = " ".join(command)
        for expected in (
            'forced_login_method="chatgpt"', "--sandbox read-only",
            "--ignore-user-config", "--ephemeral", "--skip-git-repo-check",
            "dst=/request,readonly", "dst=/codex", "--output-schema /request/schema.json",
        ):
            self.assertIn(expected, joined)
        for feature in DISABLED_FEATURES:
            self.assertIn("--disable " + feature, joined)
        for forbidden in ("--gpus", "/var/run/docker.sock", "OPENAI_API_KEY", "CODEX_API_KEY", "NEURAL_LAB_API_KEY"):
            self.assertNotIn(forbidden, joined)
        self.assertEqual(command[-1], "-")
        self.assertEqual(PLAN_SCHEMA["additionalProperties"], False)

    def test_schema_response_is_validated_and_planner_container_is_removed(self):
        prompt = []
        with tempfile.TemporaryDirectory() as auth:
            planner = CodexPlanner("owner", auth_dir=auth, runtime_seconds=17)
            def control(*args):
                if args[0] == "create":
                    mount = next(arg for arg in args if "dst=/answer" in arg)
                    answer = Path(mount.split("src=", 1)[1].split(",dst=", 1)[0])
                    (answer / "plan.json").write_text(json.dumps(PLAN))
                return "container"
            process = Mock()
            process.returncode = 0
            process.poll.return_value = 0
            process.communicate.side_effect = lambda **kwargs: prompt.append(kwargs["input"])
            with patch("neural_lab.codex_planner.DockerRunner._docker", side_effect=control) as docker:
                with patch("neural_lab.codex_planner.subprocess.Popen", return_value=process):
                    result = planner.plan([{"summary": "previous experiment"}], 2)
            self.assertEqual(result.code, PLAN["code"])
            self.assertIn('"runtime_seconds": 17', prompt[0].decode())
            self.assertIn("previous experiment", prompt[0].decode())
            self.assertEqual(docker.call_args.args[:2], ("rm", "-f"))

    def test_codex_error_never_invokes_api_fallback_or_exposes_raw_output(self):
        with tempfile.TemporaryDirectory() as auth:
            planner = CodexPlanner("owner", auth_dir=auth)
            process = Mock(returncode=1)
            process.poll.return_value = 1
            with patch("neural_lab.codex_planner.DockerRunner._docker") as docker:
                with patch("neural_lab.codex_planner.subprocess.Popen", return_value=process):
                    with self.assertRaisesRegex(PlannerError, "No API fallback"):
                        planner.plan([], 1)
            self.assertEqual(docker.call_args.args[:2], ("rm", "-f"))

    def test_shutdown_kills_cli_process_and_removes_planner_container(self):
        stop = threading.Event()
        stop.set()
        with tempfile.TemporaryDirectory() as auth:
            planner = CodexPlanner("owner", auth_dir=auth, stop=stop)
            process = Mock()
            process.poll.return_value = None
            with patch("neural_lab.codex_planner.DockerRunner._docker") as docker:
                with patch("neural_lab.codex_planner.subprocess.Popen", return_value=process):
                    with self.assertRaisesRegex(PlannerError, "shutdown"):
                        planner.plan([], 1)
            process.kill.assert_called_once()
            self.assertEqual(docker.call_args.args[:2], ("rm", "-f"))

    def test_timeout_cleans_up_and_cleanup_failure_halts(self):
        with tempfile.TemporaryDirectory() as auth:
            planner = CodexPlanner("owner", auth_dir=auth, timeout=1)
            process = Mock()
            process.poll.return_value = None
            with patch("neural_lab.codex_planner.DockerRunner._docker"):
                with patch("neural_lab.codex_planner.subprocess.Popen", return_value=process):
                    with patch("neural_lab.codex_planner.time.monotonic", side_effect=[0, 2]):
                        with self.assertRaisesRegex(PlannerError, "timeout"):
                            planner.plan([], 1)
            process.kill.assert_called_once()
            def bad_cleanup(*args):
                if args[0] == "rm":
                    raise RuntimeError("daemon unavailable")
                return "container"
            with patch("neural_lab.codex_planner.DockerRunner._docker", side_effect=bad_cleanup):
                with patch("neural_lab.codex_planner.subprocess.Popen", return_value=process):
                    with self.assertRaises(CleanupError):
                        planner.plan([], 1)

    def test_missing_dedicated_auth_directory_is_actionable(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(PlannerError, "codex-login"):
                CodexPlanner("owner", auth_dir=Path(directory) / "missing")


if __name__ == "__main__":
    unittest.main()
