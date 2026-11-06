import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from neural_lab.artifacts import ArtifactWriter
from neural_lab.nl001_corpus import make_sampled_corpus
from neural_lab.nl001_data import digest
from neural_lab.nl001_training import _batch, _flatten
from neural_lab.nl001_training_data import example

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('calibration_draft', ROOT / 'jobs/drafts/nl001_calibration_session11_guarded.py')
draft = importlib.util.module_from_spec(spec)
spec.loader.exec_module(draft)


class RootArtifacts(unittest.TestCase):
    def test_registers_selected_checkpoint_bytes_and_rejects_digest_mismatch(self):
        from neural_lab.nl001_models import CalibrationModel

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            work = root / 'calibration'
            reports = {}
            for arm in ('A', 'B'):
                entry = ArtifactWriter(work / 'selection' / arm).save_checkpoint(
                    'best-' + arm, CalibrationModel(arm), arm=arm, seed=1, selection={})
                reports[arm] = {'training': {'checkpoint_sha256': entry['sha256']}}
            writer = ArtifactWriter(root)
            draft.register_checkpoints(writer, work, {'reports': reports})
            entries = json.loads((root / 'artifacts.json').read_text())['checkpoints']
            self.assertEqual([e['sha256'] for e in entries],
                             [reports[a]['training']['checkpoint_sha256'] for a in ('A', 'B')])
            self.assertTrue(all(e['format'] == 'torch-pt' and e['path'].endswith('.pt') for e in entries))
            reports['A']['training']['checkpoint_sha256'] = 'bad'
            with self.assertRaisesRegex(RuntimeError, 'digest mismatch'):
                draft.register_checkpoints(writer, work, {'reports': reports})

    def test_all_split_tensors_and_training_order(self):
        corpus = make_sampled_corpus({'train': [11], 'validation': [12], 'test': [13]}, per_stratum=1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            draft.save_split_artifacts(ArtifactWriter(root), corpus, 'manifest-pin')
            entries = json.loads((root / 'artifacts.json').read_text())['datasets']
            self.assertEqual(len(entries), 3)
            for split, entry in zip(('train', 'validation', 'test'), entries):
                programs = corpus['splits'][split]
                records = [example(v, q) for p in programs for v in p['variants'] for q in v['questions']]
                inputs, targets = _batch(records, range(len(records)), 'cpu')
                if split == 'train':
                    self.assertEqual(records, _flatten(programs))
                expected = {**{'input_' + k: v for k, v in inputs.items()},
                            **{'target_' + k: v for k, v in targets.items()}}
                self.assertEqual(entry['source']['split_canonical_sha256'], digest(programs))
                with np.load(root / entry['path'], allow_pickle=False) as saved:
                    self.assertEqual(set(saved.files), set(expected))
                    for key, value in expected.items():
                        np.testing.assert_array_equal(saved[key], value.numpy())


if __name__ == '__main__':
    unittest.main()
