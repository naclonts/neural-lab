import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location('output_validator', Path(__file__).resolve().parents[1] / 'scripts/validate_nl001_output.py')
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


def fixture(root):
    (root / 'result.json').write_text('{"metrics": {"checks": 1}, "summary": "fixture"}')
    (root / 'nested').mkdir()
    (root / 'nested' / 'inventory.json').write_text('evidence')
    entries = []
    for name in ('result.json', 'nested/inventory.json'):
        data = (root / name).read_bytes()
        entries.append({'path': name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
    (root / validator.INVENTORY).write_text(json.dumps(entries))
    mark(root)


def mark(root):
    (root / validator.MARKER).write_text(hashlib.sha256((root / validator.INVENTORY).read_bytes()).hexdigest() + '\n')


class OutputValidator(unittest.TestCase):
    def test_valid_and_mutations(self):
        for change in ('none', 'missing_marker', 'changed_file', 'missing_file', 'extra_file', 'bad_marker', 'symlink'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                fixture(root)
                if change == 'missing_marker': (root / validator.MARKER).unlink()
                if change == 'changed_file': (root / 'nested/inventory.json').write_text('changed')
                if change == 'missing_file': (root / 'nested/inventory.json').unlink()
                if change == 'extra_file': (root / 'extra').write_text('x')
                if change == 'bad_marker': (root / validator.MARKER).write_text('0' * 64 + '\n')
                if change == 'symlink': (root / 'link').symlink_to(root / 'result.json')
                if change == 'none':
                    self.assertEqual(validator.validate(root)['files_verified'], 2)
                else:
                    with self.assertRaises(ValueError): validator.validate(root)

    def test_rejects_invalid_worker_results_even_if_inventory_matches(self):
        for result in ({'metrics': {}, 'summary': 'ok'},
                       {'metrics': {'x': float('nan')}, 'summary': 'ok'},
                       {'metrics': {'x': True}, 'summary': 'ok'},
                       {'metrics': {'x': 1}, 'summary': ''},
                       {'metrics': {'x': 1}, 'summary': 'x' * 2001}):
            with self.subTest(result=result), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                fixture(root)
                (root / 'result.json').write_text(json.dumps(result))
                entries = json.loads((root / validator.INVENTORY).read_text())
                data = (root / 'result.json').read_bytes()
                entries[0].update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
                (root / validator.INVENTORY).write_text(json.dumps(entries))
                mark(root)
                with self.assertRaises(ValueError): validator.validate(root)

    def test_inventory_rejects_duplicate_and_traversal_even_with_matching_marker(self):
        for change in ('duplicate', 'traversal'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                fixture(root)
                entries = json.loads((root / validator.INVENTORY).read_text())
                if change == 'duplicate': entries.append(entries[0])
                else: entries[0]['path'] = '../result.json'
                (root / validator.INVENTORY).write_text(json.dumps(entries))
                mark(root)
                with self.assertRaises(ValueError): validator.validate(root)


if __name__ == '__main__':
    unittest.main()
