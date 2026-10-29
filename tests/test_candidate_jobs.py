"""Short CPU executions of the queued architecture comparison scripts."""
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from neural_lab.runner import read_result


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = (
    ROOT / "jobs/candidates/conditioned_low_rank_gate.py",
    ROOT / "jobs/candidates/energy_normalized_mixer.py",
)


class CandidateJobTests(unittest.TestCase):
    def test_candidate_scripts_train_and_emit_valid_results(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is installed in CI and the GPU worker, not this host shell")

        previous_deterministic = torch.are_deterministic_algorithms_enabled()
        previous_threads = torch.get_num_threads()
        try:
            for index, script in enumerate(SCRIPTS):
                with self.subTest(script=script.name), tempfile.TemporaryDirectory() as directory:
                    spec = importlib.util.spec_from_file_location(f"candidate_smoke_{index}", script)
                    self.assertIsNotNone(spec)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    module.SEEDS = (1101,)
                    module.EPOCHS = 3
                    output = Path(directory)
                    real_path = Path
                    with patch.object(module, "Path", side_effect=lambda value: output if value == "/output" else real_path(value)):
                        with patch.object(module.torch.cuda, "is_available", return_value=False):
                            module.main()

                    result = read_result(output)
                    saved = json.loads((output / "result.json").read_text())
                    details = json.loads((output / "details.json").read_text())
                    self.assertEqual(result, saved)
                    self.assertEqual(details["device"], "cpu")
                    self.assertEqual(len(details["records"]), 1)
                    self.assertEqual(details["records"][0]["seed"], 1101)
                    metrics = result["metrics"]
                    self.assertLess(
                        abs(metrics["candidate_parameters"] / metrics["baseline_parameters"] - 1),
                        0.05,
                    )
                    for key in ("baseline_accuracy", "candidate_accuracy"):
                        self.assertGreaterEqual(metrics[key], 0)
                        self.assertLessEqual(metrics[key], 1)
                    for key in ("baseline_heldout_loss", "candidate_heldout_loss",
                                "baseline_training_seconds", "candidate_training_seconds"):
                        self.assertTrue(math.isfinite(metrics[key]), key)
                        self.assertGreaterEqual(metrics[key], 0)
                    self.assertAlmostEqual(
                        metrics["accuracy_delta"],
                        metrics["candidate_accuracy"] - metrics["baseline_accuracy"],
                    )
        finally:
            torch.use_deterministic_algorithms(previous_deterministic)
            torch.set_num_threads(previous_threads)


if __name__ == "__main__":
    unittest.main()
