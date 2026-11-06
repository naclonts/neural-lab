import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

from neural_lab.codex_planner import PLAN_SCHEMA
from neural_lab.pi_planner import PI_IMAGE, PLAN_OUTPUT_LIMIT, PiPlanner, extract_plan
from neural_lab.planner import PlannerError
from neural_lab.runner import CleanupError


PLAN = {"hypothesis": "Compare gated convolutions", "code": "print('sandbox only')\n"}


def make_process(data=b"", returncode=0, exited=True):
    """A process-like mock whose stdout is a real pipe carrying data."""
    read_fd, write_fd = os.pipe()

    def feed():
        view = memoryview(data)
        while view:
            try:
                written = os.write(write_fd, view[:4096])
            except BrokenPipeError:
                break
            view = view[written:]
        os.close(write_fd)

    threading.Thread(target=feed, daemon=True).start()
    process = Mock()
    process.stdout = os.fdopen(read_fd, "rb")
    process.returncode = returncode
    process.poll.return_value = None if not exited else returncode
    return process


class PiPlannerTests(unittest.TestCase):
    def test_command_is_hardened_and_keeps_api_keys_out_of_the_environment(self):
        with tempfile.TemporaryDirectory() as auth:
            planner = PiPlanner("owner", auth_dir=auth, model="pi-model")
            command = planner.command("plan-test", "/tmp/request", "/tmp/answer")
        joined = " ".join(command)
        for expected in (
            "--read-only --cap-drop ALL", "--security-opt no-new-privileges:true",
            "--pids-limit 128", "--memory 1g", "--cpus 1", "--init",
            "dst=/pi", "dst=/request,readonly", "dst=/answer",
            "--env PI_CODING_AGENT_DIR=/pi", "--env HOME=/tmp",
            "--entrypoint pi", "--print --no-session --no-tools", "--offline",
            "--system-prompt /request/system-prompt.txt",
        ):
            self.assertIn(expected, joined)
        for forbidden in ("--gpus", "/var/run/docker.sock", "ANTHROPIC_API_KEY",
                          "OPENAI_API_KEY", "NEURAL_LAB_API_KEY"):
            self.assertNotIn(forbidden, joined)
        self.assertIn("--model pi-model", joined)
        self.assertEqual(PLAN_SCHEMA["additionalProperties"], False)

    def test_fenced_annotated_stdout_is_parsed_validated_and_container_removed(self):
        files = {}
        with tempfile.TemporaryDirectory() as auth:
            planner = PiPlanner("owner", auth_dir=auth, runtime_seconds=17)
            def control(*args):
                if args[0] == "create":
                    mount = next(arg for arg in args if "dst=/request,readonly" in arg)
                    request = Path(mount.split("src=", 1)[1].split(",dst=", 1)[0])
                    files["request"] = (request / "request.json").read_text()
                    files["system"] = (request / "system-prompt.txt").read_text()
                return "container"
            stdout = ("Scratch notes before the plan.\n```json\n"
                      + json.dumps(PLAN) + "\n```\nTrailing remarks.\n").encode()
            process = make_process(stdout)
            with patch("neural_lab.pi_planner.DockerRunner._docker", side_effect=control) as docker:
                with patch("neural_lab.pi_planner.subprocess.Popen", return_value=process):
                    result = planner.plan([{"summary": "previous experiment"}], 2)
            self.assertEqual(result.code, PLAN["code"])
            self.assertEqual(result.hypothesis, PLAN["hypothesis"])
            self.assertEqual(docker.call_args.args[:2], ("rm", "-f"))
        self.assertIn('"runtime_seconds": 17', files["request"])
        self.assertIn("previous experiment", files["request"])
        self.assertIn('"attempt": 2', files["request"])
        self.assertIn("hypothesis", files["system"])

    def test_unparseable_output_raises_sanitized_error_without_raw_stdout(self):
        with tempfile.TemporaryDirectory() as auth:
            planner = PiPlanner("owner", auth_dir=auth)
            process = make_process(b"The plan is SECRET-MARKER, trust me.")
            with patch("neural_lab.pi_planner.DockerRunner._docker"):
                with patch("neural_lab.pi_planner.subprocess.Popen", return_value=process):
                    with self.assertRaises(PlannerError) as caught:
                        planner.plan([], 1)
            self.assertNotIn("SECRET-MARKER", str(caught.exception))
            self.assertIn("unparseable", str(caught.exception))

    def test_nonzero_exit_reports_no_api_fallback_and_removes_container(self):
        with tempfile.TemporaryDirectory() as auth:
            planner = PiPlanner("owner", auth_dir=auth)
            process = make_process(b"", returncode=1)
            with patch("neural_lab.pi_planner.DockerRunner._docker") as docker:
                with patch("neural_lab.pi_planner.subprocess.Popen", return_value=process):
                    with self.assertRaisesRegex(PlannerError, "No API fallback"):
                        planner.plan([], 1)
            self.assertEqual(docker.call_args.args[:2], ("rm", "-f"))

    def test_shutdown_kills_cli_process_and_removes_planner_container(self):
        stop = threading.Event()
        stop.set()
        with tempfile.TemporaryDirectory() as auth:
            planner = PiPlanner("owner", auth_dir=auth, stop=stop)
            process = make_process(b"", exited=False)
            with patch("neural_lab.pi_planner.DockerRunner._docker") as docker:
                with patch("neural_lab.pi_planner.subprocess.Popen", return_value=process):
                    with self.assertRaisesRegex(PlannerError, "shutdown"):
                        planner.plan([], 1)
            process.kill.assert_called_once()
            self.assertEqual(docker.call_args.args[:2], ("rm", "-f"))

    def test_timeout_cleans_up_and_cleanup_failure_halts(self):
        with tempfile.TemporaryDirectory() as auth:
            planner = PiPlanner("owner", auth_dir=auth, timeout=1)
            process = make_process(b"", exited=False)
            with patch("neural_lab.pi_planner.DockerRunner._docker"):
                with patch("neural_lab.pi_planner.subprocess.Popen", return_value=process):
                    with patch("neural_lab.pi_planner.time.monotonic", side_effect=[0, 2]):
                        with self.assertRaisesRegex(PlannerError, "timeout"):
                            planner.plan([], 1)
            process.kill.assert_called_once()
            def bad_cleanup(*args):
                if args[0] == "rm":
                    raise RuntimeError("daemon unavailable")
                return "container"
            with patch("neural_lab.pi_planner.DockerRunner._docker", side_effect=bad_cleanup):
                with patch("neural_lab.pi_planner.subprocess.Popen", return_value=process):
                    with self.assertRaises(CleanupError):
                        planner.plan([], 1)

    def test_missing_dedicated_auth_directory_is_actionable(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(PlannerError, "pi-auth"):
                PiPlanner("owner", auth_dir=Path(directory) / "missing")

    def test_extract_plan_handles_nested_braces_and_braces_inside_strings(self):
        text = ('Scratchpad mentioning braces and fences.\n```json\n'
                + json.dumps({
                    "hypothesis": "nested {braces} and \"quotes\" \\ inside",
                    "code": "d = {1: {2: 3}}\nassert d[1] == {2: 3}\n",
                }) + "\n```\n")
        result = extract_plan(text)
        self.assertEqual(result["code"], "d = {1: {2: 3}}\nassert d[1] == {2: 3}\n")
        self.assertEqual(result["hypothesis"], 'nested {braces} and "quotes" \\ inside')

    def test_oversized_stdout_is_capped_killed_and_sanitized(self):
        with tempfile.TemporaryDirectory() as auth:
            planner = PiPlanner("owner", auth_dir=auth)
            data = b"SECRET-MARKER" * (PLAN_OUTPUT_LIMIT // 8 + 1)
            process = make_process(data, exited=False)
            with patch("neural_lab.pi_planner.DockerRunner._docker") as docker:
                with patch("neural_lab.pi_planner.subprocess.Popen", return_value=process):
                    with self.assertRaisesRegex(PlannerError, "size limit") as caught:
                        planner.plan([], 1)
            self.assertNotIn("SECRET-MARKER", str(caught.exception))
            process.kill.assert_called_once()
            self.assertEqual(docker.call_args.args[:2], ("rm", "-f"))

    def test_output_at_the_cap_is_still_accepted(self):
        plan = json.dumps(PLAN).encode()
        padding = PLAN_OUTPUT_LIMIT - len(plan) - 20
        stdout = b"x" * padding + b"\n```json\n" + plan + b"\n```\n"
        self.assertLessEqual(len(stdout), PLAN_OUTPUT_LIMIT)
        with tempfile.TemporaryDirectory() as auth:
            planner = PiPlanner("owner", auth_dir=auth)
            process = make_process(stdout)
            with patch("neural_lab.pi_planner.DockerRunner._docker"):
                with patch("neural_lab.pi_planner.subprocess.Popen", return_value=process):
                    result = planner.plan([], 1)
        self.assertEqual(result.hypothesis, PLAN["hypothesis"])


if __name__ == "__main__":
    unittest.main()
