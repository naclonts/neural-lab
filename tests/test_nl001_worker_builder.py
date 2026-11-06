"""The generated entry point transports pinned bytes without running training."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts.build_nl001_worker import build, MAX_SCRIPT_BYTES
from neural_lab.queue import read_script, MAX_SCRIPT_BYTES as QUEUE_LIMIT


def sha(data):
    return hashlib.sha256(data).hexdigest()


class WorkerBuilderTest(unittest.TestCase):
    def test_disabled_before_reading_or_building(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / 'worker.py'
            with self.assertRaisesRegex(RuntimeError, 'mandatory selection audit gates'):
                build(root / 'missing', '0' * 64, root / 'candidate', output)
            self.assertFalse(output.exists())
            script = Path(__file__).resolve().parents[1] / 'scripts/build_nl001_worker.py'
            result = subprocess.run([sys.executable, str(script), '--config', str(root / 'missing'),
                '--config-sha256', '0' * 64, '--output', str(output)], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('mandatory selection audit gates', result.stderr)
            self.assertFalse(output.exists())

    @unittest.skip('legacy builder disabled pending integration of mandatory selection gates')
    def test_no_argument_entry_and_pins(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source'; source.mkdir()
            (source / '__init__.py').write_bytes(b'# fixture\n')
            inputs = {}
            for name in ('train', 'validation', 'manifest', 'audit'):
                path = root / (name + '.json'); path.write_text(json.dumps({'name': name}))
                inputs[name] = {'path': str(path), 'sha256': sha(path.read_bytes())}
            config = {'version': 'nl001-curve-candidate-v1', 'source_dir': str(source),
                      'source_sha256': {'__init__.py': sha((source / '__init__.py').read_bytes())},
                      'inputs': inputs, 'manifest_sha256': 'fixture', 'audit_sha256': 'fixture', 'curve': {}}
            config_path = root / 'config.json'; config_path.write_text(json.dumps(config))
            candidate = root / 'candidate.py'
            candidate.write_text('''import hashlib, json\nfrom pathlib import Path\ndef run(path, pin, output, *, worker_output):\n raw = Path(path).read_bytes()\n assert hashlib.sha256(raw).hexdigest() == pin and worker_output\n config = json.loads(raw)\n for item in config["inputs"].values():\n  assert hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest() == item["sha256"]\n assert hashlib.sha256((Path(config["source_dir"]) / "__init__.py").read_bytes()).hexdigest() == config["source_sha256"]["__init__.py"]\n (output / "fixture.txt").write_text("transport verified")\n''')
            worker = root / 'train.py'
            pin = sha(config_path.read_bytes())
            metadata = build(config_path, pin, candidate, worker)
            self.assertEqual(metadata['worker_sha256'], sha(worker.read_bytes()))
            self.assertEqual(MAX_SCRIPT_BYTES, QUEUE_LIMIT)
            self.assertEqual(read_script(worker), worker.read_bytes())
            second = root / 'second.py'
            build(config_path, pin, candidate, second)
            self.assertEqual(worker.read_bytes(), second.read_bytes())
            # Corruption must fail before the candidate can run.
            text = worker.read_text()
            start = text.index("PAYLOAD = '") + len("PAYLOAD = '")
            text = text[:start] + ('0' if text[start] != '0' else '1') + text[start + 1:]
            tampered = root / 'tampered.py'; tampered.write_text(text)
            tampered_output = root / 'tampered_output'; tampered_output.mkdir()
            env = dict(os.environ, NL001_WORKER_OUTPUT=str(tampered_output))
            failed = subprocess.run([sys.executable, '-B', str(tampered)], env=env, capture_output=True, timeout=30)
            self.assertNotEqual(failed.returncode, 0)
            self.assertFalse(list(tampered_output.iterdir()))
            output = root / 'output'; output.mkdir()
            env = dict(os.environ, NL001_WORKER_OUTPUT=str(output))
            result = subprocess.run([sys.executable, '-B', str(worker)], env=env, capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr.decode())
            self.assertEqual((output / 'fixture.txt').read_text(), 'transport verified')
            with self.assertRaises(FileExistsError): build(config_path, pin, candidate, worker)
            # Incompressible pinned data cannot bypass the queue boundary or leave an output.
            path = Path(inputs['train']['path']); path.write_bytes(os.urandom(100_000))
            inputs['train']['sha256'] = sha(path.read_bytes())
            config_path.write_text(json.dumps(config)); oversize = root / 'oversize.py'
            with self.assertRaisesRegex(ValueError, 'worker exceeds queue limit'):
                build(config_path, sha(config_path.read_bytes()), candidate, oversize)
            self.assertFalse(oversize.exists())
            config_path.write_text(json.dumps(config))
            pin = sha(config_path.read_bytes())
            (source / '__init__.py').write_text('changed')
            with self.assertRaises(ValueError): build(config_path, pin, candidate, root / 'bad.py')
            with self.assertRaises(ValueError): build(config_path, '0' * 64, candidate, root / 'bad.py')


    @unittest.skip('legacy builder disabled pending integration of mandatory selection gates')
    def test_regeneration_checks_real_corpus_before_candidate(self):
        from neural_lab.nl001_corpus import make_sampled_corpus
        parameters = {'split_seeds': {'train': [11], 'validation': [12], 'test': [13]},
                      'nodes': 10, 'edge_probability': 0.2, 'missing_fraction': 0.25,
                      'per_stratum': 2, 'sampling_seed': 42}
        corpus = make_sampled_corpus(**parameters)
        canonical = lambda x: json.dumps(x, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = Path(__file__).resolve().parents[1] / 'neural_lab'
            names = ['__init__.py', 'nl001_corpus.py', 'nl001_data.py', 'synthetic_graph.py']
            inputs = {name: {'sha256': sha(canonical(corpus['splits'][name]))} for name in ('train', 'validation')}
            for name in ('manifest', 'audit'):
                path = root / (name + '.json'); path.write_bytes(b'{}')
                inputs[name] = {'path': str(path), 'sha256': sha(path.read_bytes())}
            config = {'version': 'nl001-curve-candidate-v1', 'source_dir': str(source),
                      'source_sha256': {n: sha((source / n).read_bytes()) for n in names},
                      'inputs': inputs, 'manifest_sha256': 'fixture', 'audit_sha256': 'fixture', 'curve': {},
                      'regeneration': {'parameters': parameters, 'corpus_sha256': corpus['sha256']}}
            candidate = root / 'candidate.py'
            candidate.write_text('''import json
from pathlib import Path
def run(path, pin, output, *, worker_output):
 config = json.loads(Path(path).read_bytes())
 assert 'regeneration' not in config
 for name in ('train', 'validation'):
  assert json.loads(Path(config['inputs'][name]['path']).read_bytes())
 (output / 'ran').write_text('yes')
''')
            config_path = root / 'config.json'
            def run_worker(name):
                config_path.write_text(json.dumps(config))
                worker = root / (name + '.py')
                build(config_path, sha(config_path.read_bytes()), candidate, worker)
                output = root / name; output.mkdir()
                result = subprocess.run([sys.executable, '-B', str(worker)], env=dict(os.environ, NL001_WORKER_OUTPUT=str(output)), capture_output=True, timeout=60)
                return result, output
            result, output = run_worker('good')
            self.assertEqual(result.returncode, 0, result.stderr.decode())
            self.assertTrue((output / 'ran').exists())
            config['regeneration']['corpus_sha256'] = '0' * 64
            result, output = run_worker('bad_corpus')
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(list(output.iterdir()))
            config['regeneration']['corpus_sha256'] = corpus['sha256']
            config['inputs']['train']['sha256'] = '0' * 64
            result, output = run_worker('bad_split')
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(list(output.iterdir()))
            config['inputs']['train']['sha256'] = sha(canonical(corpus['splits']['train']))
            config['inputs']['validation']['sha256'] = '0' * 64
            result, output = run_worker('bad_validation')
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(list(output.iterdir()))
            config['inputs']['validation']['sha256'] = sha(canonical(corpus['splits']['validation']))
            removed = config['source_sha256'].pop('synthetic_graph.py')
            result, output = run_worker('missing_dependency')
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(list(output.iterdir()))
            config['source_sha256']['synthetic_graph.py'] = removed
            config['regeneration']['parameters']['nodes'] = 2
            result, output = run_worker('failed_generation')
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(list(output.iterdir()))


if __name__ == '__main__': unittest.main()
