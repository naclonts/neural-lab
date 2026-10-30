import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import patch

from neural_lab.planner import APIPlanner, DemoPlanner, Plan, validate_plan
from neural_lab.runner import CleanupError, DockerRunner, read_result
from neural_lab.supervisor import BudgetExhausted, Journal, experiment_step, instance_lock


RESULT = {"metrics": {"accuracy": 0.75}, "summary": "Synthetic held-out result."}


class FakeRunner:
    def run(self, run_id, inputs, outputs, stop):
        assert (inputs / "train.py").is_file()
        return {"status": "succeeded", "result": RESULT}


class SupervisorTests(unittest.TestCase):
    def test_budget_reservation_survives_reopen_and_failed_plans(self):
        class BrokenPlanner:
            def plan(self, history, attempt):
                raise RuntimeError("failed API request")
        with tempfile.TemporaryDirectory() as root:
            journal = Journal(root)
            owner = journal.owner
            status = experiment_step(journal, BrokenPlanner(), FakeRunner(), threading.Event(),
                                     max_attempts=2, budget_seconds=10, timeout=10)
            self.assertEqual(status, "failed")
            journal.close()
            journal = Journal(root)
            self.assertEqual(journal.owner, owner)
            self.assertEqual(journal.failures(), 1)
            with self.assertRaises(BudgetExhausted):
                journal.reserve(max_attempts=2, budget_seconds=10, timeout=10)
            journal.close()

    def test_successful_step_journals_code_hypothesis_and_history(self):
        with tempfile.TemporaryDirectory() as root:
            journal = Journal(root)
            status = experiment_step(journal, DemoPlanner(), FakeRunner(), threading.Event(),
                                     max_attempts=1, budget_seconds=10, timeout=10)
            self.assertEqual(status, "succeeded")
            row = journal.history()[0]
            self.assertEqual(row["result"], RESULT)
            self.assertIn("MLP", row["hypothesis"])
            self.assertEqual(len(list((Path(root) / "inputs").glob("*/train.py"))), 1)
            with self.assertRaises(BudgetExhausted):
                journal.reserve(max_attempts=1, budget_seconds=20, timeout=10)
            journal.close()

    def test_recovery_marks_unfinished_runs_and_preserves_reservation(self):
        with tempfile.TemporaryDirectory() as root:
            journal = Journal(root)
            run_id, _ = journal.reserve(max_attempts=2, budget_seconds=20, timeout=10)
            journal.update(run_id, "running")
            journal.recover()
            self.assertEqual(journal.history()[0]["status"], "interrupted")
            self.assertEqual(journal.failures(), 1)
            journal.reserve(max_attempts=2, budget_seconds=20, timeout=10)
            with self.assertRaises(BudgetExhausted):
                journal.reserve(max_attempts=3, budget_seconds=20, timeout=10)
            journal.close()

    def test_instance_lock_rejects_concurrent_supervisor(self):
        with tempfile.TemporaryDirectory() as root:
            with instance_lock(root):
                with self.assertRaises(RuntimeError):
                    with instance_lock(root):
                        pass

    def test_cleanup_failure_halts_instead_of_becoming_retry(self):
        class BrokenRunner:
            def run(self, *args):
                raise CleanupError("container might still be alive")
        with tempfile.TemporaryDirectory() as root:
            journal = Journal(root)
            with self.assertRaises(CleanupError):
                experiment_step(journal, DemoPlanner(), BrokenRunner(), threading.Event(),
                                max_attempts=2, budget_seconds=20, timeout=10)
            self.assertEqual(journal.history()[0]["status"], "failed")
            journal.close()

    def test_shutdown_after_planning_prevents_container_launch(self):
        stop = threading.Event()
        class StoppingPlanner:
            def plan(self, history, attempt):
                stop.set()
                return Plan("stop", "raise AssertionError('must not run')")
        with tempfile.TemporaryDirectory() as root:
            journal = Journal(root)
            runner = FakeRunner()
            with patch.object(runner, "run", side_effect=AssertionError("launched")):
                status = experiment_step(journal, StoppingPlanner(), runner, stop,
                                         max_attempts=1, budget_seconds=10, timeout=10)
            self.assertEqual(status, "interrupted")
            journal.close()

    def test_docker_command_has_fixed_boundaries_and_no_host_credentials(self):
        runner = DockerRunner("testowner")
        with patch.dict(os.environ, {"NEURAL_LAB_API_KEY": "secret-sentinel"}):
            command = runner.command("test", "/tmp/input", "/tmp/output")
        joined = " ".join(command)
        for required in ("--read-only", "--cap-drop ALL", "no-new-privileges:true",
                         "--pids-limit 256", "--memory 8g", "--gpus device=0",
                         "dst=/input,readonly", "dst=/opt/tooling/neural_lab,readonly"):
            self.assertIn(required, joined)
        for forbidden in ("secret-sentinel", "/var/run/docker.sock", "--privileged", "--network host"):
            self.assertNotIn(forbidden, joined)
        self.assertNotIn("--gpus", DockerRunner("testowner", gpu=None).command("test", "/tmp/in", "/tmp/out"))

    def test_nvidia_backend_selects_only_requested_gpu_and_preserves_isolation(self):
        with patch.dict(os.environ, {"NEURAL_LAB_GPU_BACKEND": "nvidia"}):
            command = DockerRunner("testowner", gpu="2").command("test", "/tmp/in", "/tmp/out")
            cpu = DockerRunner("testowner", gpu=None).command("cpu", "/tmp/in", "/tmp/out")
        joined = " ".join(command)
        for required in ("--runtime nvidia", "NVIDIA_VISIBLE_DEVICES=2",
                         "NVIDIA_DRIVER_CAPABILITIES=compute,utility", "--read-only",
                         "--cap-drop ALL", "no-new-privileges:true"):
            self.assertIn(required, joined)
        self.assertNotIn("--gpus", command)
        self.assertNotIn("--runtime", cpu)
        self.assertNotIn("NVIDIA_VISIBLE_DEVICES", " ".join(cpu))
        with self.assertRaises(ValueError):
            DockerRunner("testowner", gpu_backend="unsupported")

    def test_timeout_removes_container(self):
        runner = DockerRunner("testowner", gpu=None, timeout=0)
        with patch.object(runner, "_docker", return_value="container") as docker:
            outcome = runner.run("run", "/tmp/in", "/tmp/out", threading.Event())
        self.assertEqual(outcome["status"], "failed")
        self.assertIn("runtime", outcome["error"])
        self.assertEqual(docker.call_args.args, ("rm", "-f", "neural-lab-run"))

    def test_result_reader_rejects_symlink_fifo_oversize_and_nonfinite_metrics(self):
        with tempfile.TemporaryDirectory() as root:
            target = Path(root) / "result.json"
            target.write_text(json.dumps(RESULT))
            self.assertEqual(read_result(root), RESULT)
            target.unlink()
            target.symlink_to("/etc/passwd")
            with self.assertRaises(OSError):
                read_result(root)
            target.unlink()
            os.mkfifo(target)
            with self.assertRaises(ValueError):
                read_result(root)
            target.unlink()
            target.write_text("x" * 65537)
            with self.assertRaises(ValueError):
                read_result(root)
            target.write_text('{"metrics":{"x":NaN},"summary":"invalid"}')
            with self.assertRaises(ValueError):
                read_result(root)
            target.write_text('{"metrics":{"x":true},"summary":"invalid"}')
            with self.assertRaises(ValueError):
                read_result(root)

    def test_plan_schema_rejects_container_controls_and_oversized_code(self):
        self.assertEqual(validate_plan({"hypothesis": "test", "code": "pass"}).code, "pass")
        for invalid in (
            {"hypothesis": "test", "code": "pass", "mount": "/home"},
            {"hypothesis": "", "code": "pass"},
            {"hypothesis": "test", "code": "x" * 65537},
        ):
            with self.assertRaises(ValueError):
                validate_plan(invalid)

    def test_api_planner_sends_bounded_contract_and_validates_response(self):
        import io
        response = {"choices": [{"message": {"content": json.dumps(
            {"hypothesis": "test architecture", "code": "pass\n"}
        )}}]}
        with patch.dict(os.environ, {"NEURAL_LAB_API_KEY": "host-only-sentinel"}):
            with patch("urllib.request.build_opener") as factory:
                factory.return_value.open.return_value = io.BytesIO(json.dumps(response).encode())
                plan = APIPlanner("test-model", runtime_seconds=17).plan([], 1)
                request = factory.return_value.open.call_args.args[0]
                payload = json.loads(request.data)
                self.assertEqual(payload["max_tokens"], 6000)
                self.assertEqual(json.loads(payload["messages"][1]["content"])["runtime_seconds"], 17)
                self.assertNotIn("host-only-sentinel", request.data.decode())
                self.assertEqual(plan.code, "pass\n")

    def test_demo_trains_and_produces_visualizations_and_plain_json_weights(self):
        import torch
        from neural_lab.demo_experiment import run
        with tempfile.TemporaryDirectory() as root:
            with patch("neural_lab.demo_experiment.Path", return_value=Path(root)):
                with patch.object(torch.cuda, "is_available", return_value=False):
                    run({"width": 16, "depth": 1, "residual": True, "seed": 7, "epochs": 2})
            result = read_result(root)
            self.assertIn("baseline_accuracy", result["metrics"])
            self.assertTrue((Path(root) / "visualizations" / "architecture.svg").is_file())
            self.assertIn("input.weight", json.loads((Path(root) / "model-state.json").read_text()))

    def test_demo_plans_change_architecture_and_api_endpoint_requires_transport_security(self):
        planner = DemoPlanner()
        self.assertNotEqual(planner.plan([], 1).code, planner.plan([], 2).code)
        for url in ("http://example.com/v1/chat/completions", "https://user:pass@example.com/api"):
            with self.assertRaises(ValueError):
                APIPlanner("model", url=url)
        APIPlanner("model", url="http://127.0.0.1:8000/v1/chat/completions")


@unittest.skipUnless(os.environ.get("NEURAL_LAB_DOCKER_TESTS") == "1", "Docker integration opt-in")
class DockerIntegrationTests(unittest.TestCase):
    def run_script(self, source, timeout=15):
        with tempfile.TemporaryDirectory() as root:
            inputs, outputs = Path(root) / "input", Path(root) / "output"
            inputs.mkdir()
            outputs.mkdir()
            (inputs / "train.py").write_text(source)
            runner = DockerRunner("ci-" + Path(root).name, image="python:3.11-slim", gpu=None, timeout=timeout)
            result = runner.run(Path(root).name, inputs, outputs, threading.Event())
            self.assertEqual(runner._docker("ps", "-aq", "--filter", "label=neural-lab.owner=" + runner.owner), "")
            return result

    def test_real_container_has_readonly_mounts_and_does_not_receive_planner_key(self):
        script = """import json, os
from pathlib import Path
assert os.getuid() != 0
assert 'NEURAL_LAB_API_KEY' not in os.environ
for path in ('/input/train.py', '/etc/forbidden', '/opt/tooling/neural_lab/forbidden'):
    try:
        Path(path).write_text('should fail')
    except OSError:
        pass
    else:
        raise AssertionError('writable restricted path: ' + path)
Path('/output/result.json').write_text(json.dumps(
    {'metrics': {'isolation': 1.0}, 'summary': 'Docker boundary checks passed'}))
"""
        with patch.dict(os.environ, {"NEURAL_LAB_API_KEY": "host-only-secret"}):
            outcome = self.run_script(script)
        self.assertEqual(outcome["status"], "succeeded")

    def test_real_timeout_stops_and_removes_worker(self):
        outcome = self.run_script("import time\ntime.sleep(60)\n", timeout=2)
        self.assertEqual(outcome["status"], "failed")
        self.assertIn("runtime", outcome["error"])


if __name__ == "__main__":
    unittest.main()
