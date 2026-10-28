import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from neural_lab.manage import LowDiskSpace, check_readiness, main, read_status, require_disk_space
from neural_lab.supervisor import Journal, main as supervisor_main


class ManageTests(unittest.TestCase):
    def test_status_on_uninitialized_path_does_not_create_state(self):
        with tempfile.TemporaryDirectory() as root:
            state = Path(root) / "missing"
            result = read_status(state)
            self.assertFalse(result["initialized"])
            self.assertFalse(state.exists())

    def test_status_reads_live_journal_without_recovery_or_budget_changes(self):
        with tempfile.TemporaryDirectory() as root:
            journal = Journal(root)
            run_id, _ = journal.reserve(max_attempts=2, budget_seconds=20, timeout=10)
            journal.update(run_id, "running", hypothesis="example")
            result = read_status(root)
            self.assertEqual(result["attempts"], 1)
            self.assertEqual(result["reserved_training_seconds"], 10)
            self.assertEqual(result["status_counts"], {"running": 1})
            self.assertEqual(journal.history()[0]["status"], "running")
            journal.update(run_id, "succeeded", result={"metrics": {"accuracy": 0.75}, "summary": "held-out result"})
            result = read_status(root, limit=1)
            self.assertEqual(result["latest_runs"][0]["result"]["metrics"]["accuracy"], 0.75)
            self.assertEqual(result["attempts"], 1)
            journal.close()

    def test_disk_admission_uses_existing_ancestor_without_creating_directories(self):
        usage = SimpleNamespace(total=20 * 1024**3, used=19 * 1024**3, free=1024**3)
        with tempfile.TemporaryDirectory() as root:
            state = Path(root) / "not-created"
            with patch("neural_lab.manage.shutil.disk_usage", return_value=usage):
                with self.assertRaises(LowDiskSpace):
                    require_disk_space(state, 10)
            self.assertFalse(state.exists())

    def test_doctor_reports_safe_diagnostics_without_raw_stderr_or_environment(self):
        def failure(*args):
            raise subprocess.CalledProcessError(1, args, stderr="sensitive-sentinel")
        with patch("neural_lab.manage.os.getuid", return_value=1000):
            with patch("neural_lab.manage.platform.system", return_value="Linux"):
                with patch("neural_lab.manage.available_disk", return_value=100):
                    with patch("neural_lab.manage._command", side_effect=failure):
                        result = check_readiness(planner="demo", gpu=None)
        self.assertFalse(all(item["ok"] for item in result))
        self.assertNotIn("sensitive-sentinel", json.dumps(result))
        self.assertIn("docker", [item["check"] for item in result])

    def test_remote_docker_endpoint_is_rejected_before_container_commands(self):
        with patch("neural_lab.manage.os.getuid", return_value=1000):
            with patch("neural_lab.manage.platform.system", return_value="Linux"):
                with patch("neural_lab.manage.available_disk", return_value=100):
                    with patch.dict("os.environ", {"DOCKER_HOST": "ssh://remote-host", "DOCKER_CONTEXT": ""}):
                        with patch("neural_lab.manage._command") as command:
                            result = check_readiness(planner="demo", gpu=None)
        command.assert_not_called()
        self.assertFalse(next(item for item in result if item["check"] == "docker")["ok"])

    def test_failed_preflight_does_not_call_planner_or_reserve_attempt(self):
        with tempfile.TemporaryDirectory() as root:
            with contextlib.redirect_stdout(io.StringIO()):
                with patch("neural_lab.supervisor.os.getuid", return_value=1000):
                    with patch("neural_lab.supervisor.signal.signal"):
                        with patch("neural_lab.supervisor.os.umask"):
                            with patch("neural_lab.supervisor.DockerRunner") as runner:
                                with patch("neural_lab.supervisor.check_readiness", return_value=[
                                    {"check": "worker_image", "ok": False, "fix": "Pull image"}
                                ]):
                                    with patch("neural_lab.supervisor.DemoPlanner") as planner:
                                        code = supervisor_main(["--state-dir", root, "--planner", "demo", "--once"])
            self.assertEqual(code, 1)
            planner.assert_not_called()
            runner.return_value.recover.assert_called_once()
            self.assertEqual(read_status(root)["attempts"], 0)

    def test_low_disk_after_preflight_pauses_before_planner_call(self):
        with tempfile.TemporaryDirectory() as root:
            with contextlib.redirect_stdout(io.StringIO()):
                with patch("neural_lab.supervisor.os.getuid", return_value=1000):
                    with patch("neural_lab.supervisor.signal.signal"):
                        with patch("neural_lab.supervisor.os.umask"):
                            with patch("neural_lab.supervisor.DockerRunner"):
                                with patch("neural_lab.supervisor.check_readiness", return_value=[{"ok": True}]):
                                    with patch("neural_lab.supervisor.require_disk_space", side_effect=LowDiskSpace("low disk")):
                                        with patch("neural_lab.supervisor.DemoPlanner") as planner:
                                            code = supervisor_main(["--state-dir", root, "--planner", "demo"])
            self.assertEqual(code, 0)
            planner.return_value.plan.assert_not_called()
            self.assertEqual(read_status(root)["attempts"], 0)

    def test_status_cli_outputs_json_and_checks_argument_limits(self):
        with tempfile.TemporaryDirectory() as root:
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(main(["status", "--state-dir", root]), 0)
            self.assertFalse(json.loads(output.getvalue())["initialized"])
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                main(["status", "--limit", "0"])


if __name__ == "__main__":
    unittest.main()
