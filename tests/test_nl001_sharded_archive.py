import tempfile
import copy
import hashlib
import json
import unittest
from pathlib import Path

import torch

from neural_lab.nl001_sharded_archive import read_shards, verify_reconstruction, write_shards
from neural_lab.nl001_training import _batch


class ShardedArchiveTest(unittest.TestCase):
    def test_reconstruction_and_determinism(self):
        records = [({'adj': [[i, i + 1], [0, i]], 'query': i % 2},
                    {'label': i % 3, 'witness': [i, 0]}) for i in range(7)]
        expected = _batch(records, range(len(records)), 'cpu')
        with tempfile.TemporaryDirectory() as root:
            a, pin = write_shards(iter(records), Path(root) / 'a', shard_size=3, max_shard_bytes=100000)
            b, pin2 = write_shards(iter(records), Path(root) / 'b', shard_size=3, max_shard_bytes=100000)
            self.assertEqual((a, pin), (b, pin2))
            batches = list(read_shards(Path(root) / 'a', pin))
            self.assertEqual([s['count'] for s in a['shards']], [3, 3, 1])
            for group, original in zip(('inputs', 'targets'), expected):
                for key in original:
                    self.assertTrue(torch.equal(torch.cat([v[group][key] for v in batches]), original[key]))
            (Path(root) / 'a' / 'shard-000001.pt').write_bytes(b'corrupted')
            with self.assertRaises(ValueError):
                list(read_shards(Path(root) / 'a', pin))
            with self.assertRaises(ValueError):
                list(read_shards(Path(root) / 'b', '0' * 64))

    def test_failure_has_no_manifest_and_collision_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'archive'
            with self.assertRaises(ValueError):
                write_shards([({'x': [1]}, {'y': 1})], path, shard_size=1, max_shard_bytes=1)
            self.assertFalse((path / 'manifest.json').exists())
            with self.assertRaises(FileExistsError):
                write_shards([({'x': [1]}, {'y': 1})], path, shard_size=1, max_shard_bytes=100000)

    def test_full_reconstruction_rejects_late_damage_and_changed_records(self):
        records = [({'x': [i]}, {'y': i}) for i in range(5)]
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'archive'
            _, pin = write_shards(records, path, shard_size=2, max_shard_bytes=100000)
            self.assertEqual(verify_reconstruction(path, pin, iter(records)), 5)
            for bad in (records[:-1], records + [records[0]],
                        records[:-1] + [records[0]], records[::-1]):
                with self.assertRaises(ValueError):
                    verify_reconstruction(path, pin, iter(bad))
            (path / 'shard-000002.pt').write_bytes(b'corrupted')
            with self.assertRaises(ValueError):
                verify_reconstruction(path, pin, iter(records))

    def test_full_reconstruction_rejects_missing_late_shard(self):
        records = [({'x': [i]}, {'y': i}) for i in range(3)]
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'archive'
            _, pin = write_shards(records, path, shard_size=1, max_shard_bytes=100000)
            (path / 'shard-000002.pt').unlink()
            with self.assertRaises(FileNotFoundError):
                verify_reconstruction(path, pin, iter(records))

    def test_invalid_pinned_manifest_rejected_before_first_batch(self):
        records = [({'x': [i]}, {'y': i}) for i in range(3)]
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'archive'
            manifest, _ = write_shards(records, path, shard_size=2, max_shard_bytes=100000)
            variants = []
            for field, value in (('count', 4), ('shard_size', True),
                                 ('max_shard_bytes', 1), ('extra', 1)):
                bad = copy.deepcopy(manifest)
                bad[field] = value
                variants.append(bad)
            for field, value in (('start', 99), ('count', True), ('sha256', 'x' * 64),
                                 ('file', '../shard.pt'), ('extra', 1)):
                bad = copy.deepcopy(manifest)
                bad['shards'][1][field] = value
                variants.append(bad)
            for bad in variants:
                data = json.dumps(bad, sort_keys=True, separators=(',', ':')).encode()
                (path / 'manifest.json').write_bytes(data)
                pin = hashlib.sha256(data).hexdigest()
                with self.subTest(manifest=bad), self.assertRaises(ValueError):
                    next(read_shards(path, pin))

    def test_extra_fields_rejected_within_and_across_shards(self):
        records = [({'x': [i]}, {'y': i}) for i in range(4)]
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'archive'
            _, pin = write_shards(records, path, shard_size=2, max_shard_bytes=100000)
            for index in (1, 2):
                for group in (0, 1):
                    changed = list(records)
                    pair = list(changed[index])
                    pair[group] = {**pair[group], 'extra': 1}
                    changed[index] = tuple(pair)
                    with self.assertRaisesRegex(ValueError, 'inconsistent record fields'):
                        verify_reconstruction(path, pin, changed)
                    with self.assertRaisesRegex(ValueError, 'inconsistent record fields'):
                        write_shards(changed, Path(root) / f'bad-{index}-{group}',
                                     shard_size=2, max_shard_bytes=100000)


if __name__ == '__main__':
    unittest.main()
