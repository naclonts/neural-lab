"""Exercise custody and publication without running any model or training."""
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / 'scripts/nl001_curve_candidate.py'
VALIDATOR = ROOT / 'scripts/validate_nl001_output.py'


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return digest_bytes(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode())


class CandidateTest(unittest.TestCase):
    def test_legacy_candidate_fails_before_read_or_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            spec = importlib.util.spec_from_file_location('legacy_curve_candidate', CANDIDATE)
            candidate = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(candidate)
            with self.assertRaisesRegex(RuntimeError, 'mandatory selection audit gates'):
                candidate.run(root / 'missing', '0' * 64, root / 'direct')
            self.assertFalse((root / 'direct').exists())
            for worker in (False, True):
                output = root / ('worker' if worker else 'output')
                if worker:
                    output.mkdir()
                command = [sys.executable, str(CANDIDATE), '--config', str(root / 'missing'),
                           '--config-sha256', '0' * 64, '--output', str(output)]
                if worker:
                    command.append('--worker-output')
                result = subprocess.run(command, capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('mandatory selection audit gates are not integrated', result.stderr)
                self.assertEqual(list(output.iterdir()) if worker else [], [])
                if not worker:
                    self.assertFalse(output.exists())
                self.assertFalse(list(root.glob('.nl001-curve-*')))

    @unittest.skip('legacy candidate disabled pending integration of mandatory selection gates')
    def test_pinned_mock_run_publishes_and_tamper_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'source'
            source.mkdir()
            modules = {
                '__init__.py': '',
                'nl001_data.py': 'import hashlib,json\ndef digest(x): return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode()).hexdigest()\n',
                'nl001_manifest.py': 'def authenticate_split(*args): pass\n',
                'nl001_learning_curve.py': 'from pathlib import Path\ndef run_curve(*,output,**kwargs):\n p=Path(output); p.mkdir(); (p/"summary.json").write_text("{}")\n return {"fits": {}}\n',
                'artifacts.py': 'import json\nfrom pathlib import Path\nclass ArtifactWriter:\n def __init__(self, output): self.output=Path(output); self.items=[]\n def register_file(self, category,path,label,**kw):\n  self.items.append(label); (self.output/"artifacts.json").write_text(json.dumps({"items":self.items})); return {}\n',
            }
            pins = {}
            for name, text in modules.items():
                data = text.encode(); (source / name).write_bytes(data); pins[name] = digest_bytes(data)
            manifest = {'corpus_sha256': 'fixture'}
            manifest_pin = canonical(manifest)
            audit = {'version': 'nl001-corpus-audit-v2', 'regeneration': 'exact match using current generator and checker',
                     'manifest': manifest, 'manifest_sha256': manifest_pin, 'corpus_sha256': 'fixture',
                     'observed_view_audit': {'comparisons': 1, 'max_states_per_comparison': 1}}
            inputs = {}
            for name, value in {'train': [], 'validation': [], 'manifest': manifest, 'audit': audit}.items():
                path = root / (name + '.json'); data = json.dumps(value).encode(); path.write_bytes(data)
                inputs[name] = {'path': str(path), 'sha256': digest_bytes(data)}
            config = {'version': 'nl001-curve-candidate-v1', 'source_dir': str(source), 'source_sha256': pins,
                      'inputs': inputs, 'manifest_sha256': manifest_pin, 'audit_sha256': canonical(audit), 'curve': {}}
            config_path = root / 'config.json'; data = json.dumps(config).encode(); config_path.write_bytes(data)
            output = root / 'completed'
            command = [sys.executable, str(CANDIDATE), '--config', str(config_path),
                       '--config-sha256', digest_bytes(data), '--output', str(output)]
            subprocess.run(command, check=True, capture_output=True)
            subprocess.run([sys.executable, str(VALIDATOR), str(output)], check=True, capture_output=True)
            worker = root / 'worker-output'; worker.mkdir()
            subprocess.run(command[:-1] + [str(worker), '--worker-output'], check=True, capture_output=True)
            subprocess.run([sys.executable, str(VALIDATOR), str(worker / 'completed')], check=True, capture_output=True)
            self.assertEqual(json.loads((worker / 'result.json').read_text()),
                             json.loads((worker / 'completed' / 'result.json').read_text()))
            self.assertEqual(json.loads((worker / 'completed' / 'child-source-manifest.json').read_text())['package'],
                             'source/neural_lab')
            from neural_lab.runner import read_result
            self.assertIn('metrics', read_result(worker))
            # Fail or interrupt after the directory rename, at worker-result publication.
            for exception in ('OSError("injected link failure")', 'KeyboardInterrupt()'):
                failed_worker = root / ('link-failure-' + exception.split('(')[0])
                failed_worker.mkdir()
                injected = ('import os, runpy, sys\n'
                            'def fail(*args, **kwargs): raise ' + exception + '\n'
                            'os.link = fail\n'
                            'sys.argv = ' + repr(command[1:-1] + [str(failed_worker), '--worker-output']) + '\n'
                            'runpy.run_path(' + repr(str(CANDIDATE)) + ', run_name="__main__")\n')
                self.assertNotEqual(subprocess.run([sys.executable, '-c', injected], capture_output=True).returncode, 0)
                self.assertFalse((failed_worker / 'result.json').exists())
                self.assertFalse((failed_worker / 'completed' / 'nl001-calibration-complete.txt').exists())
                self.assertTrue((failed_worker / 'completed' / 'failure.txt').exists())
            (output / 'train.json').write_text('modified')
            self.assertNotEqual(subprocess.run([sys.executable, str(VALIDATOR), str(output)], capture_output=True).returncode, 0)
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
            (source / '__init__.py').write_text('changed')
            self.assertNotEqual(subprocess.run(command[:-1] + [str(root / 'other')], capture_output=True).returncode, 0)
            self.assertFalse((root / 'other').exists())
            # A run that fails after staging preserves diagnostics without publishing.
            (source / '__init__.py').write_text('')
            failing = 'def run_curve(**kwargs): raise RuntimeError("fixture failure")\n'
            (source / 'nl001_learning_curve.py').write_text(failing)
            config['source_sha256']['nl001_learning_curve.py'] = digest_bytes(failing.encode())
            data = json.dumps(config).encode(); config_path.write_bytes(data)
            failed = root / 'failed'
            command = [sys.executable, str(CANDIDATE), '--config', str(config_path),
                       '--config-sha256', digest_bytes(data), '--output', str(failed)]
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
            self.assertFalse(failed.exists())
            stages = list(root.glob('.nl001-curve-*'))
            self.assertEqual(len(stages), 1)
            self.assertIn('fixture failure', (stages[0] / 'failure.txt').read_text())
            self.assertFalse((stages[0] / 'nl001-calibration-complete.txt').exists())
            # Simulate a competing destination appearing after the run.
            racing = root / 'racing'
            runner = ('from pathlib import Path\ndef run_curve(*,output,**kwargs):\n'
                      ' p=Path(output); p.mkdir(); (p/"summary.json").write_text("{}")\n'
                      + ' Path(' + repr(str(racing)) + ').mkdir()\n return {"fits": {}}\n')
            (source / 'nl001_learning_curve.py').write_text(runner)
            config['source_sha256']['nl001_learning_curve.py'] = digest_bytes(runner.encode())
            data = json.dumps(config).encode(); config_path.write_bytes(data)
            command = [sys.executable, str(CANDIDATE), '--config', str(config_path),
                       '--config-sha256', digest_bytes(data), '--output', str(racing)]
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
            stages = list(root.glob('.nl001-curve-*'))
            self.assertEqual(len(stages), 2)
            self.assertTrue(all(not (stage / 'nl001-calibration-complete.txt').exists() for stage in stages))


if __name__ == '__main__':
    unittest.main()
