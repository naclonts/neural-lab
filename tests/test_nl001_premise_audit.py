import tempfile
import unittest
from pathlib import Path

from neural_lab.artifacts import ArtifactWriter
from neural_lab.nl001_corpus import make_sampled_corpus
from neural_lab.nl001_data import digest
from neural_lab.nl001_manifest import make_manifest
from neural_lab.nl001_models import CalibrationModel
from neural_lab.nl001_premise_audit import audit


class PremiseAuditTests(unittest.TestCase):
    def test_threshold_extremes_and_pins(self):
        corpus = make_sampled_corpus({'train': [11], 'validation': [12], 'test': [13]}, per_stratum=1)
        manifest = make_manifest(corpus)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            entry = ArtifactWriter(root).save_checkpoint('model', CalibrationModel('B'), arm='B', seed=1, selection={})
            args = dict(programs=corpus['splits']['validation'], split='validation', manifest=manifest,
                        manifest_sha256=digest(manifest), checkpoint=root / entry['path'], checkpoint_sha256=entry['sha256'])
            all_edges = audit(**args, edge_threshold=0)
            no_edges = audit(**args, edge_threshold=1)
            self.assertGreater(all_edges['witness_edges'], 0)
            self.assertEqual(all_edges['micro_witness_recall'], 1)
            self.assertEqual(all_edges['full_witnesses_selected'], all_edges['positive_claims'])
            self.assertEqual(no_edges['micro_witness_recall'], 0)
            self.assertEqual(no_edges['full_witnesses_selected'], 0)
            with self.assertRaises(ValueError):
                audit(**{**args, 'checkpoint_sha256': '0' * 64})
            with self.assertRaises(ValueError):
                audit(**args, edge_threshold=True)
            args['programs'][0]['variants'][0]['questions'][0]['label'] = 'Unknown'
            with self.assertRaises(ValueError):
                audit(**args)


if __name__ == '__main__':
    unittest.main()
