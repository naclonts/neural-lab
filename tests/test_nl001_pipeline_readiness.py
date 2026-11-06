import copy
import json
import tempfile
import unittest
from pathlib import Path

from neural_lab.nl001_corpus import make_sampled_corpus
from neural_lab.nl001_data import digest
from neural_lab.nl001_manifest import make_manifest, authenticate_split
from neural_lab.nl001_run_evaluation import evaluate_split


class ReadinessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = make_sampled_corpus({'train': [11], 'validation': [12], 'test': [13]}, per_stratum=1)
        cls.manifest = make_manifest(cls.corpus)
        cls.pin = digest(cls.manifest)

    def test_split_ownership_and_tampering(self):
        splits = self.corpus['splits']
        authenticate_split(splits['train'], 'train', self.manifest, self.pin)
        with self.assertRaises(ValueError):
            authenticate_split(splits['test'], 'train', self.manifest, self.pin)
        changed = copy.deepcopy(splits['train'])
        changed[0]['variants'][0]['questions'][0]['claim'] += '!'
        with self.assertRaises(ValueError):
            authenticate_split(changed, 'train', self.manifest, self.pin)
        with self.assertRaises(ValueError):
            authenticate_split(splits['train'], 'train', self.manifest, '0' * 64)

    def test_control_replay_and_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            report = evaluate_split(programs=self.corpus['splits']['validation'], split='validation',
                manifest=self.manifest, manifest_sha256=self.pin, output=directory, mode='oracle')
            self.assertGreater(report['counts']['replay_passed'], 0)
            self.assertEqual(report['counts']['replay_rejected'], 0)
            self.assertEqual(report['counts']['false_proven'], 0)
            artifacts = json.loads((Path(directory) / 'artifacts.json').read_text())
            self.assertEqual(len(artifacts['datasets']), 4)
            self.assertTrue((Path(directory) / 'predictions.json').exists())
