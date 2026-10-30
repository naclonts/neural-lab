"""Real numeric dataset and checkpoint round trips inside the PyTorch worker."""
import hashlib
import importlib.util
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None

from neural_lab.artifacts import ArtifactWriter, BestCheckpoint, load_json_state


@unittest.skipIf(torch is None, "PyTorch runs in the training container")
class ArtifactTests(unittest.TestCase):
    def test_dataset_round_trip_preserves_values_dtypes_shapes_and_provenance(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            writer = ArtifactWriter(directory)
            features = torch.randn(12, 4, generator=torch.Generator().manual_seed(5))
            labels = torch.arange(12) % 3
            entry = writer.save_dataset("synthetic-seed-5", {"train_x": features, "train_y": labels},
                provenance={"kind": "generated", "seed": 5, "generator": "test"})
            path = Path(directory) / entry["path"]
            with np.load(path, allow_pickle=False) as saved:
                self.assertTrue(torch.equal(torch.from_numpy(saved["train_x"]), features))
                self.assertTrue(torch.equal(torch.from_numpy(saved["train_y"]), labels))
            self.assertEqual(entry["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(entry["arrays"]["train_y"], {"shape": [12], "dtype": "int64"})
            manifest = json.loads((Path(directory) / "artifacts.json").read_text())
            self.assertEqual(manifest["datasets"][0]["source"]["seed"], 5)

    def test_best_weights_restore_selected_predictions_without_pickle(self):
        with tempfile.TemporaryDirectory() as directory:
            model = torch.nn.Linear(4, 3)
            sample = torch.randn(2, 4)
            best = BestCheckpoint()
            best.observe(model, 1.0, 0)
            expected = model(sample).detach().clone()
            with torch.no_grad():
                model.weight.add_(100)
            best.observe(model, 2.0, 1)
            writer = ArtifactWriter(directory)
            entry = best.save(writer, "candidate-seed-5-best", arm="candidate", seed=5,
                              datasets=["datasets/synthetic-seed-5.npz"])
            saved = torch.nn.Linear(4, 3)
            saved.load_state_dict(load_json_state(Path(directory) / entry["path"]))
            self.assertTrue(torch.equal(saved(sample), expected))
            self.assertEqual(entry["selection"]["step"], 0)
            self.assertEqual(entry["selection"]["metric"], "training_cross_entropy")
            self.assertEqual(entry["selection"]["observations"], 2)
            self.assertIn("held-out data not used", entry["selection"]["scope"])

    def test_non_numeric_datasets_and_non_finite_selection_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = ArtifactWriter(directory)
            with self.assertRaises(ValueError):
                writer.save_dataset("unsafe", {"objects": [{"pickle": True}]}, provenance={})
            best = BestCheckpoint()
            with self.assertRaises(ValueError):
                best.observe(torch.nn.Linear(1, 1), float("nan"), 0)

    def test_existing_manifest_is_extended_and_external_files_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            writer = ArtifactWriter(output)
            first = writer.save_dataset("first", {"x": torch.ones(2)}, provenance={"kind": "generated"})
            second = output / "datasets" / "selected.parquet"
            second.write_bytes(b"opaque-dataset-download")
            reopened = ArtifactWriter(output)
            entry = reopened.register_file("datasets", second, "selected", format="parquet", source={"kind": "huggingface"})
            manifest = json.loads((output / "artifacts.json").read_text())
            self.assertEqual([v["path"] for v in manifest["datasets"]], [first["path"], entry["path"]])
            outside = Path(directory) / "outside"
            outside.write_bytes(b"not-this-job")
            with self.assertRaises(ValueError):
                reopened.register_file("datasets", outside, "outside")

    def test_product_jobs_save_each_arm_and_actual_dataset(self):
        root = Path(__file__).resolve().parents[1]
        previous_threads = torch.get_num_threads()
        previous_deterministic = torch.are_deterministic_algorithms_enabled()
        try:
            for script_name in ("centered_product_mixer", "product_scale_diagnostic"):
                spec = importlib.util.spec_from_file_location(script_name + "_artifact_smoke", root / "jobs/candidates" / (script_name + ".py"))
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                if script_name == "centered_product_mixer":
                    module.SEEDS, module.EPOCHS = (1101,), 2
                else:
                    module.TEACHER_SEEDS, module.INIT_SEEDS = (5101,), (7101,)
                    module.STEPS, module.CHECKPOINTS = 2, (0, 1, 2)
                with self.subTest(script=script_name), tempfile.TemporaryDirectory() as directory:
                    output, real_path = Path(directory), Path
                    with patch.object(module, "Path", side_effect=lambda value: output if value == "/output" else real_path(value)):
                        with patch.object(module.torch.cuda, "is_available", return_value=False), contextlib.redirect_stdout(io.StringIO()):
                            module.main()
                    manifest = json.loads((output / "artifacts.json").read_text())
                    self.assertEqual(len(manifest["datasets"]), 1)
                    self.assertEqual({entry["arm"] for entry in manifest["checkpoints"]}, set(module.ARMS))
                    for entry in manifest["checkpoints"]:
                        model = module.Network(entry["arm"])
                        model.load_state_dict(load_json_state(output / entry["path"]))
                        self.assertEqual(entry["selection"]["observed_steps"], [0, 1, 2])
                        self.assertTrue(torch.isfinite(model(torch.zeros(2, 16))).all())
        finally:
            torch.set_num_threads(previous_threads)
            torch.use_deterministic_algorithms(previous_deterministic)


if __name__ == "__main__":
    unittest.main()
