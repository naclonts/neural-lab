import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ChildAttestation(unittest.TestCase):
    def test_success_and_reject_changed_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = root / 'neural_lab'
            package.mkdir()
            (package / '__init__.py').write_text('')
            shutil.copy(ROOT / 'neural_lab/nl001_child_attestation.py', package)
            (package / 'nl001_selection_runner.py').write_text('from . import helper\n')
            (package / 'helper.py').write_text('value = 1\n')
            pins = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in package.iterdir()}
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps({'package': str(package), 'sha256': pins}))
            report = root / 'report.json'
            env = dict(os.environ, PYTHONPATH=str(root), PYTHONDONTWRITEBYTECODE='1')
            command = [sys.executable, '-m', 'neural_lab.nl001_child_attestation',
                       str(manifest), str(report), 'neural_lab.nl001_selection_runner']
            subprocess.run(command, cwd=root, env=env, check=True, capture_output=True)
            evidence = json.loads(report.read_text())
            self.assertIn('neural_lab.helper', evidence['imported_modules'])
            self.assertEqual(evidence['target_sha256'], pins['nl001_selection_runner.py'])
            report.unlink()
            (package / 'helper.py').write_text('value = 2\n')
            result = subprocess.run(command, cwd=root, env=env, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b'child import digest mismatch', result.stderr)
            self.assertFalse(report.exists())


if __name__ == '__main__':
    unittest.main()
