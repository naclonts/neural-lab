import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

DRAFT = Path(__file__).resolve().parents[1] / 'jobs/drafts/nl001_calibration_session11_guarded.py'
spec = importlib.util.spec_from_file_location('calibration_draft', DRAFT)
draft = importlib.util.module_from_spec(spec)
spec.loader.exec_module(draft)


class CalibrationCustody(unittest.TestCase):
    def test_snapshot_copies_verified_bytes_and_rejects_bad_pin(self):
        original_cwd = Path.cwd()
        original_path = list(__import__('sys').path)
        original_env = dict(os.environ)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                package = draft.snapshot_package(root / 'snapshot')
                self.assertEqual(set(p.name for p in package.iterdir()), set(draft.PACKAGE_SHA256))
                self.assertEqual(Path.cwd(), root / 'snapshot')
                self.assertEqual(os.environ['PYTHONPATH'], str(root / 'snapshot'))
                os.chdir(original_cwd)
                __import__('sys').path[:] = original_path
                with patch.dict(draft.PACKAGE_SHA256, {'__init__.py': 'bad'}):
                    with self.assertRaisesRegex(RuntimeError, 'digest mismatch'):
                        draft.snapshot_package(root / 'bad')
        finally:
            os.chdir(original_cwd)
            __import__('sys').path[:] = original_path
            os.environ.clear()
            os.environ.update(original_env)

    def test_failure_writes_inventory_of_partial_outputs_and_no_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'nested').mkdir()
            (root / 'nested' / 'inventory.json').write_text('partial')
            with patch.object(draft, 'OUTPUT', root), patch.object(draft, 'main', side_effect=RuntimeError('fixture failure')):
                with self.assertRaisesRegex(RuntimeError, 'fixture failure'):
                    draft.guarded_entry()
            self.assertFalse((root / 'result.json').exists())
            self.assertIn('fixture failure', (root / 'nl001-calibration-failure.txt').read_text())
            paths = {item['path'] for item in __import__('json').loads((root / 'failure-inventory.json').read_text())}
            self.assertIn('nested/inventory.json', paths)
            self.assertIn('nl001-calibration-failure.txt', paths)
            self.assertNotIn('failure-inventory.json', paths)


if __name__ == '__main__':
    unittest.main()
