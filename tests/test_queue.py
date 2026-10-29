from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from neural_lab.queue import Queue, read_script, run_job, worker_lock


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
            self.assertEqual(queue.next_job()["id"], first)
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


if __name__ == "__main__":
    unittest.main()
