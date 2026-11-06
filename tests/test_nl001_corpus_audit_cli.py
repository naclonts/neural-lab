import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from neural_lab.nl001_corpus import make_sampled_corpus
from neural_lab.nl001_data import digest

ROOT = Path(__file__).resolve().parents[1]


class CorpusAuditCliTests(unittest.TestCase):
    def test_pinned_corpus_and_rejected_pin(self):
        corpus = make_sampled_corpus({'train': [11], 'validation': [12], 'test': [13]}, per_stratum=1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'corpus.json'
            source.write_text(json.dumps(corpus))
            output = root / 'report'
            cmd = [sys.executable, str(ROOT / 'scripts/audit_nl001_corpus.py'), str(source),
                   '--corpus-sha256', corpus['sha256'], '--output', str(output)]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            pins = json.loads(result.stdout)
            report = json.loads((output / 'corpus-audit.json').read_text())
            semantic = json.loads((output / 'semantic-audit.json').read_text())
            self.assertEqual(pins['corpus_audit_sha256'], digest(report))
            self.assertEqual(pins['semantic_audit_sha256'], digest(semantic))
            self.assertGreater(semantic['result']['claims'], 0)
            self.assertNotEqual(subprocess.run(cmd, capture_output=True).returncode, 0)
            capped = cmd[:-1] + [str(root / 'capped'), '--max-comparisons', '1']
            failure = subprocess.run(capped, capture_output=True, text=True)
            self.assertNotEqual(failure.returncode, 0)
            self.assertIn('aggregate comparison budget exhausted', failure.stderr)
            self.assertFalse((root / 'capped').exists())
            # Exercise CLI wiring and publication ordering with a deterministic
            # state-consuming comparator; real backtracking is covered separately.
            from unittest.mock import patch
            import runpy
            def consume(a, b, *, max_states, state_callback):
                state_callback()
                state_callback()
                return False
            state_output = root / 'state-capped'
            argv = cmd[1:-1] + [str(state_output), '--max-total-states', '1']
            with patch('sys.argv', argv), patch('neural_lab.nl001_split_audit.isomorphic', side_effect=consume):
                with self.assertRaisesRegex(RuntimeError, 'aggregate isomorphism state budget exhausted'):
                    runpy.run_path(str(ROOT / 'scripts/audit_nl001_corpus.py'), run_name='__main__')
            self.assertFalse(state_output.exists())
            cmd[cmd.index(corpus['sha256'])] = '0' * 64
            cmd[-1] = str(root / 'wrong-pin')
            self.assertNotEqual(subprocess.run(cmd, capture_output=True).returncode, 0)
            self.assertFalse((root / 'wrong-pin').exists())


if __name__ == '__main__':
    unittest.main()
