import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from neural_lab.artifacts import ArtifactWriter
from neural_lab.hf_data import load_public_dataset


class FakeDataset:
    features = SimpleNamespace(to_dict=lambda: {'x': {'dtype': 'float32'}})

    def __init__(self, rows=4):
        self.rows = rows

    def __len__(self):
        return self.rows

    def select(self, rows):
        return FakeDataset(len(rows))

    def to_parquet(self, path):
        Path(path).write_bytes(b'PAR1test-fixture' + str(self.rows).encode())


class HuggingFaceDataTests(unittest.TestCase):
    def test_pins_revision_disables_auth_and_registers_actual_split_with_digest(self):
        sha = 'a' * 40
        info = SimpleNamespace(sha=sha, card_data={'license': 'cc0-1.0'})
        api = Mock()
        api.dataset_info.return_value = info
        api_constructor = Mock(return_value=api)
        loader = Mock(return_value=FakeDataset())
        fake_modules = {'datasets': SimpleNamespace(load_dataset=loader, __version__='4.8.5'),
                        'huggingface_hub': SimpleNamespace(HfApi=api_constructor)}
        with tempfile.TemporaryDirectory() as root, patch.dict(sys.modules, fake_modules):
            writer = ArtifactWriter(root)
            dataset, entry = load_public_dataset('public/example', 'train', writer, max_rows=2)
            api_constructor.assert_called_once_with(endpoint='https://huggingface.co', token=False)
            api.dataset_info.assert_called_once_with('public/example', revision='main', timeout=20)
            self.assertEqual(loader.call_args.kwargs['revision'], sha)
            self.assertIs(loader.call_args.kwargs['token'], False)
            self.assertEqual(len(dataset), 2)
            self.assertEqual(entry['source']['license'], 'cc0-1.0')
            self.assertEqual(entry['source']['revision'], sha)
            self.assertEqual(entry['source']['rows'], 2)
            saved = Path(root) / entry['path']
            self.assertTrue(saved.is_file())
            self.assertEqual(entry['sha256'], hashlib.sha256(saved.read_bytes()).hexdigest())
            manifest = json.loads((Path(root) / 'artifacts.json').read_text())
            self.assertEqual(manifest['datasets'][0]['path'], entry['path'])

    def test_rejects_local_paths_urls_missing_splits_and_invalid_limits_before_import(self):
        with tempfile.TemporaryDirectory() as root:
            writer = ArtifactWriter(root)
            for repo in ('/tmp/private', '../private', 'https://huggingface.co/datasets/example', 'org/../private'):
                with self.assertRaises(ValueError):
                    load_public_dataset(repo, 'train', writer)
            for rows in (0, -1, True, '100'):
                with self.assertRaises(ValueError):
                    load_public_dataset('public/example', 'train', writer, max_rows=rows)
            with self.assertRaises(ValueError):
                load_public_dataset('public/example', '', writer)

    def test_missing_commit_does_not_download_a_moving_revision(self):
        loader = Mock()
        api = Mock()
        api.dataset_info.return_value = SimpleNamespace(sha=None)
        with tempfile.TemporaryDirectory() as root, patch.dict(sys.modules, {
            'datasets': SimpleNamespace(load_dataset=loader),
            'huggingface_hub': SimpleNamespace(HfApi=Mock(return_value=api)),
        }):
            with self.assertRaises(ValueError):
                load_public_dataset('public/example', 'train', ArtifactWriter(root))
            loader.assert_not_called()


if __name__ == '__main__':
    unittest.main()
