import io
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from neural_lab.orchestrator import acknowledge_gpu_request, main, unload_ollama


class OllamaHandoffTests(unittest.TestCase):
    def test_unloads_each_model_and_verifies_empty(self):
        replies = [{"models": [{"name": "one"}, {"name": "two"}]}, {}, {}, {"models": []}]
        calls = []

        def urlopen(req, timeout):
            calls.append((req.full_url, json.loads(req.data) if req.data else None))
            return io.BytesIO(json.dumps(replies.pop(0)).encode())

        with patch("neural_lab.orchestrator.urllib.request.urlopen", side_effect=urlopen):
            unload_ollama("http://ollama:11434")
        self.assertEqual([c[0].rsplit("/", 1)[-1] for c in calls], ["ps", "generate", "generate", "ps"])
        self.assertEqual(calls[1][1]["keep_alive"], 0)

    def test_no_ack_when_unload_fails_or_request_changes(self):
        with tempfile.TemporaryDirectory() as root:
            state = Path(root)
            request = state / "ollama-gpu-request"
            request.write_text("a" * 32)
            with patch("neural_lab.orchestrator.unload_ollama", side_effect=RuntimeError):
                with self.assertRaises(RuntimeError):
                    acknowledge_gpu_request(state, "http://example")
            self.assertFalse((state / "ollama-gpu-ack").exists())
            with patch("neural_lab.orchestrator.unload_ollama", side_effect=lambda _: request.write_text("b" * 32)):
                self.assertFalse(acknowledge_gpu_request(state, "http://example"))
            self.assertFalse((state / "ollama-gpu-ack").exists())

    def test_running_pi_stops_before_unload_and_ack(self):
        with tempfile.TemporaryDirectory() as root:
            repo = Path(root)
            state = repo / ".neural-lab"
            state.mkdir()
            request = state / "ollama-gpu-request"
            ack = state / "ollama-gpu-ack"
            started = threading.Event()
            ended = threading.Event()
            errors = []

            def session(*args, stop, **kwargs):
                started.set()
                if not stop.wait(5):
                    raise AssertionError("session was not stopped")
                ended.set()
                return {"exit_code": 0, "reason": "stopped"}

            def worker():
                if not started.wait(5):
                    errors.append("session did not start")
                    return
                request.write_text("a" * 32)
                deadline = time.monotonic() + 5
                while not ack.exists() and time.monotonic() < deadline:
                    time.sleep(.01)
                if not ack.exists() or ack.read_text() != "a" * 32:
                    errors.append("ack missing")
                request.unlink(missing_ok=True)

            def unload(_):
                self.assertTrue(ended.is_set())
                self.assertFalse(ack.exists())

            thread = threading.Thread(target=worker)
            thread.start()
            with patch.dict(os.environ, {"PI_PROVIDER": "ollama"}), \
                    patch("neural_lab.orchestrator.budget_ok", return_value=(True, "ok")), \
                    patch("neural_lab.orchestrator.run_pi_session", side_effect=session), \
                    patch("neural_lab.orchestrator.unload_ollama", side_effect=unload):
                main(["--backend", "pi", "--repo", root])
            thread.join(5)
            self.assertFalse(thread.is_alive())
            self.assertEqual(errors, [])

    def test_back_to_back_requests_are_both_acknowledged(self):
        with tempfile.TemporaryDirectory() as root:
            state = Path(root) / ".neural-lab"
            state.mkdir()
            request = state / "ollama-gpu-request"
            ack = state / "ollama-gpu-ack"
            request.write_text("a" * 32)
            errors = []

            def worker():
                for current, following in (("a", "b"), ("b", None)):
                    deadline = time.monotonic() + 5
                    while time.monotonic() < deadline:
                        if ack.exists() and ack.read_text() == current * 32:
                            break
                        time.sleep(.01)
                    else:
                        errors.append("ack missing for " + current)
                        request.unlink(missing_ok=True)
                        return
                    if following:
                        # Replace immediately, without a visible idle interval.
                        replacement = state / "replacement"
                        replacement.write_text(following * 32)
                        os.replace(replacement, request)
                    else:
                        request.unlink()

            thread = threading.Thread(target=worker)
            thread.start()
            with patch.dict(os.environ, {"PI_PROVIDER": "ollama"}), \
                    patch("neural_lab.orchestrator.unload_ollama"):
                main(["--backend", "pi", "--repo", root])
            thread.join(5)
            self.assertFalse(thread.is_alive())
            self.assertEqual(errors, [])

