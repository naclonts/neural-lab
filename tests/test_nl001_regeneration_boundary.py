"""No-training integration test: real generator, manifest, audit and candidate.

The learning curve is replaced by a sentinel, so no model or replay is tested.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from neural_lab.nl001_corpus import make_sampled_corpus
from neural_lab.nl001_corpus_audit import audit_corpus
from neural_lab.nl001_data import digest
from neural_lab.queue import read_script
from scripts.build_nl001_worker import build
from scripts.validate_nl001_output import validate

ROOT = Path(__file__).resolve().parents[1]

def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode()

def sha(data):
    return hashlib.sha256(data).hexdigest()


class RegenerationBoundaryTest(unittest.TestCase):
    @unittest.skip('legacy builder and candidate disabled pending mandatory selection gates')
    def test_actual_candidate_no_training_and_relocated_inventory(self):
        parameters = {'split_seeds': {'train': [11], 'validation': [12], 'test': [13]},
                      'nodes': 10, 'edge_probability': 0.2, 'missing_fraction': 0.25,
                      'per_stratum': 2, 'sampling_seed': 42}
        corpus = make_sampled_corpus(**parameters)
        audit = audit_corpus(corpus, corpus['sha256'])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source'; source.mkdir()
            for name in ('__init__.py', 'nl001_corpus.py', 'nl001_data.py', 'synthetic_graph.py',
                         'nl001_manifest.py', 'artifacts.py'):
                shutil.copyfile(ROOT / 'neural_lab' / name, source / name)
            # No optimizer, weights, inference, or scientific metrics are produced.
            (source / 'nl001_learning_curve.py').write_text('''import json
from pathlib import Path
def run_curve(*, output, **kwargs):
 output = Path(output); output.mkdir()
 (output / 'summary.json').write_text('{"diagnostic_only":true}')
 (output / 'infer-request-fixture.json').write_text(json.dumps({'checkpoint': str((output / 'fixture.pt').resolve())}))
 (output / 'fixture.pt').write_bytes(b'diagnostic, not a checkpoint')
 return {'fits': {}}
''')
            inputs = {name: {'sha256': sha(encoded(corpus['splits'][name]))} for name in ('train', 'validation')}
            for name, value in [('manifest', audit['manifest']), ('audit', audit)]:
                path = root / (name + '.json'); path.write_bytes(encoded(value))
                inputs[name] = {'path': str(path), 'sha256': sha(path.read_bytes())}
            config = {'version': 'nl001-curve-candidate-v1', 'source_dir': str(source),
                      'source_sha256': {p.name: sha(p.read_bytes()) for p in source.glob('*.py')},
                      'inputs': inputs, 'manifest_sha256': digest(audit['manifest']),
                      'audit_sha256': digest(audit), 'curve': {},
                      'regeneration': {'parameters': parameters, 'corpus_sha256': corpus['sha256']}}
            config_path = root / 'config.json'; config_path.write_bytes(encoded(config))
            worker = root / 'worker.py'
            build(config_path, sha(config_path.read_bytes()), ROOT / 'scripts/nl001_curve_candidate.py', worker)
            self.assertEqual(read_script(worker), worker.read_bytes())
            output = root / 'output'; output.mkdir()
            result = subprocess.run([sys.executable, '-B', str(worker)],
                                    env=dict(os.environ, NL001_WORKER_OUTPUT=str(output)),
                                    capture_output=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr.decode())
            self.assertEqual(json.loads((output / 'result.json').read_text()),
                             json.loads((output / 'completed' / 'result.json').read_text()))
            relocated = root / 'relocated'; shutil.move(output / 'completed', relocated)
            self.assertGreater(validate(relocated)['files_verified'], 0)
            request = json.loads((relocated / 'curve/infer-request-fixture.json').read_text())
            self.assertEqual(request['checkpoint'], 'curve/fixture.pt')
            self.assertEqual((relocated / request['checkpoint']).read_bytes(), b'diagnostic, not a checkpoint')
            self.assertEqual(json.loads((relocated / 'train.json').read_bytes()), corpus['splits']['train'])
            self.assertEqual(json.loads((relocated / 'validation.json').read_bytes()), corpus['splits']['validation'])


if __name__ == '__main__':
    unittest.main()
