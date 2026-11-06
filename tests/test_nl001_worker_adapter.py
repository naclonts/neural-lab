"""CPU-only checks for the standalone mock worker adapter."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'jobs/drafts/nl001_curve_worker_diagnostic.py'
VALIDATOR = ROOT / 'scripts/validate_nl001_output.py'


class WorkerAdapterTest(unittest.TestCase):
    def test_publication_and_fail_closed(self):
        from neural_lab.runner import read_result
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'output'; output.mkdir()
            env = dict(os.environ, NL001_DIAGNOSTIC_OUTPUT=str(output))
            command = [sys.executable, '-B', str(SCRIPT)]
            first = subprocess.run(command, env=env, capture_output=True, timeout=30)
            self.assertEqual(first.returncode, 0, first.stderr.decode())
            result = read_result(output)
            self.assertEqual(result['metrics']['fixture_completed'], 1)
            self.assertIn('Mock-only', result['summary'])
            validate = [sys.executable, str(VALIDATOR), str(output / 'completed')]
            self.assertEqual(subprocess.run(validate, capture_output=True).returncode, 0)
            self.assertNotEqual(subprocess.run(command, env=env, capture_output=True, timeout=30).returncode, 0)
            (output / 'completed' / 'train.json').write_text('tampered')
            self.assertNotEqual(subprocess.run(validate, capture_output=True).returncode, 0)


if __name__ == '__main__': unittest.main()
