from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from neural_lab.queue import Queue, read_script, run_job, worker_lock, wait_for_ollama_gpu
from neural_lab.runner import CleanupError


class QueueTests(unittest.TestCase):
    def test_enqueue_is_immutable_snapshot_and_fifo(self):
        with tempfile.TemporaryDirectory() as root:
            script = Path(root) / "candidate.py"
            script.write_text("print('one')\n")
            queue = Queue(Path(root) / "state")
            first = queue.enqueue(script, "First")
            script.write_text("print('two')\n")
            second = queue.enqueue(script, "Second")
            self.assertEqual((queue.root / "queue-inputs" / first / "train.py").read_text(), "print('one')\n")
            self.assertEqual(queue.status()["counts"], {"queued": 2})
            row = queue.next_job()
            self.assertEqual(row["id"], first)
            self.assertEqual(row["timeout_seconds"], 1800)
            queue.finish(first, {"status": "succeeded", "result": {"metrics": {"x": 1}, "summary": "ok"}})
            self.assertEqual(queue.next_job()["id"], second)
            self.assertEqual(queue.status()["counts"], {"running": 1, "succeeded": 1})
            queue.close()

    def test_recovery_preserves_queued_and_marks_running_interrupted(self):
        with tempfile.TemporaryDirectory() as root:
            script = Path(root) / "candidate.py"
            script.write_text("pass\n")
            queue = Queue(Path(root) / "state")
            queue.enqueue(script, "First")
            queue.enqueue(script, "Second")
            running = queue.next_job()
            owner = queue.owner
            queue.close()
            queue = Queue(Path(root) / "state")
            self.assertEqual(queue.owner, owner)
            queue.recover()
            self.assertEqual(queue.status()["counts"], {"interrupted": 1, "queued": 1})
            self.assertNotEqual(queue.next_job()["id"], running["id"])
            queue.close()

    def test_script_rejects_symlink_and_tampering_before_launch(self):
        with tempfile.TemporaryDirectory() as root:
            script = Path(root) / "candidate.py"
            script.write_text("pass\n")
            link = Path(root) / "link.py"
            link.symlink_to(script)
            with self.assertRaises(OSError):
                read_script(link)
            queue = Queue(Path(root) / "state")
            job_id = queue.enqueue(script, "test")
            row = queue.next_job()
            (queue.root / "queue-inputs" / job_id / "train.py").write_text("print('changed')\n")
            with patch("neural_lab.queue.DockerRunner") as runner:
                outcome = run_job(queue, row, threading.Event(), image="test", gpu=None)
            self.assertEqual(outcome["status"], "failed")
            runner.assert_not_called()
            queue.close()

    def test_single_worker_lock(self):
        with tempfile.TemporaryDirectory() as root:
            with worker_lock(Path(root)):
                with self.assertRaises(BlockingIOError):
                    with worker_lock(Path(root)):
                        pass

    def test_ollama_handoff_requires_matching_fresh_ack(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "ollama-gpu-ack").write_text("stale")
            stop = threading.Event()
            results = []
            thread = threading.Thread(target=lambda: results.append(wait_for_ollama_gpu(root, stop)))
            thread.start()
            request = root / "ollama-gpu-request"
            for _ in range(100):
                if request.exists():
                    break
                threading.Event().wait(0.01)
            self.assertTrue(request.exists())
            self.assertTrue(thread.is_alive())
            token = request.read_text()
            self.assertEqual(len(token), 32)
            (root / "ollama-gpu-ack").write_text(token)
            thread.join(2)
            self.assertEqual(results, [True])
            self.assertTrue(request.exists())

    def test_shutdown_cancels_wait_without_claiming_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stop = threading.Event()
            stop.set()
            self.assertFalse(wait_for_ollama_gpu(root, stop))
            self.assertFalse((root / "ollama-gpu-request").exists())

    def test_cleanup_failure_preserves_request_for_recovery(self):
        from neural_lab.queue import main
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / "job.py"
            script.write_text("pass\n")
            queue = Queue(root / "state")
            queue.enqueue(script, "test")
            queue.close()
            request = root / "state" / "ollama-gpu-request"
            def handoff(path, stop):
                request.write_text("active")
                return True
            with patch.dict("os.environ", {"PI_PROVIDER": "ollama"}), \
                    patch("neural_lab.queue.os.getuid", return_value=1000), \
                    patch("neural_lab.queue.DockerRunner") as runner, \
                    patch("neural_lab.queue.check_readiness", return_value=[{"ok": True}]), \
                    patch("neural_lab.queue.wait_for_ollama_gpu", side_effect=handoff), \
                    patch("neural_lab.queue.run_job", side_effect=CleanupError):
                with self.assertRaises(CleanupError):
                    main(["--state-dir", str(root / "state"), "worker", "--once"])
            self.assertTrue(request.exists())
            runner.return_value.recover.assert_called_once()

    def test_failed_recovery_does_not_release_stale_request(self):
        from neural_lab.queue import main
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = root / "ollama-gpu-request"
            request.write_text("previous-worker")
            with patch("neural_lab.queue.os.getuid", return_value=1000), \
                    patch("neural_lab.queue.DockerRunner") as runner:
                runner.return_value.recover.side_effect = RuntimeError("Docker unavailable")
                with self.assertRaises(RuntimeError):
                    main(["--state-dir", str(root), "worker", "--once"])
            self.assertEqual(request.read_text(), "previous-worker")


if __name__ == "__main__":
    unittest.main()
