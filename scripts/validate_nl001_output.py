#!/usr/bin/env python3
"""Verify a completed NL-001 calibration output, without importing its code.

This verifies consistency with the accompanying inventory, not authenticity or
scientific correctness. Protect the inventory digest independently if needed.
"""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path, PurePosixPath

INVENTORY = 'output-inventory.json'
MARKER = 'nl001-calibration-complete.txt'


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def validate(root):
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError('output root must be a real directory')
    for name in (INVENTORY, MARKER, 'result.json'):
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError('missing or invalid required file: ' + name)
    marker = (root / MARKER).read_text(encoding='ascii')
    if not re.fullmatch(r'[0-9a-f]{64}\n', marker) or marker.strip() != sha256(root / INVENTORY):
        raise ValueError('completion marker does not match inventory')
    entries = json.loads((root / INVENTORY).read_text(encoding='utf-8'))
    if not isinstance(entries, list):
        raise ValueError('inventory must be a list')
    expected = {}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {'path', 'bytes', 'sha256'}:
            raise ValueError('invalid inventory entry')
        name, size, digest = entry['path'], entry['bytes'], entry['sha256']
        if not isinstance(name, str):
            raise ValueError('invalid inventory path')
        parts = PurePosixPath(name)
        if (not name or parts.is_absolute() or str(parts) != name or
                any(p in ('', '.', '..') for p in name.split('/')) or name in (INVENTORY, MARKER)):
            raise ValueError('unsafe inventory path: ' + name)
        if name in expected or type(size) is not int or size < 0 or not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest):
            raise ValueError('invalid or duplicate inventory entry: ' + name)
        expected[name] = (size, digest)
    actual = set()
    for path in root.rglob('*'):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ValueError('symlink in output: ' + relative)
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError('non-regular output: ' + relative)
        if relative in (INVENTORY, MARKER):
            continue
        actual.add(relative)
        if relative not in expected or (path.stat().st_size, sha256(path)) != expected[relative]:
            raise ValueError('unexpected or changed output: ' + relative)
    if actual != set(expected) or 'result.json' not in expected:
        raise ValueError('missing inventoried files or result')
    result_path = root / 'result.json'
    if result_path.stat().st_size > 65536:
        raise ValueError('result exceeds worker size limit')
    result = json.loads(result_path.read_text(encoding='utf-8'))
    if not isinstance(result, dict):
        raise ValueError('invalid result contract')
    metrics, summary = result.get('metrics'), result.get('summary')
    if (not isinstance(metrics, dict) or not 1 <= len(metrics) <= 128 or
            any(not isinstance(k, str) or not 1 <= len(k) <= 80 or
                type(v) not in (int, float) or not math.isfinite(v)
                for k, v in metrics.items()) or
            not isinstance(summary, str) or not 1 <= len(summary) <= 2000):
        raise ValueError('invalid result contract')
    return {'inventory_sha256': marker.strip(), 'files_verified': len(actual)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(validate(args.output), sort_keys=True))
    except (ValueError, OSError, UnicodeError, json.JSONDecodeError) as exc:
        parser.exit(1, 'verification failed: ' + str(exc) + '\n')


if __name__ == '__main__':
    main()
