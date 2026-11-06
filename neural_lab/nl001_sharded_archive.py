"""Experimental bounded-example tensor archive; not a launch or custody gate.

The caller owns the input iterator and authenticates its source independently.
Only one shard of records is held at a time. Output requires a fresh directory.
"""
import hashlib
import io
import json
from pathlib import Path

import torch

from .nl001_training import _batch

VERSION = 'nl001-tensor-shards-v1'


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _record_schema(chunk, schema):
    """Reject fields that batching would otherwise silently discard."""
    for record in chunk:
        if (not isinstance(record, (tuple, list)) or len(record) != 2 or
                any(not isinstance(group, dict) or not group for group in record)):
            raise ValueError('invalid record')
        fields = tuple(frozenset(group) for group in record)
        if schema is None:
            schema = fields
        elif fields != schema:
            raise ValueError('inconsistent record fields')
    return schema


def write_shards(records, output, *, shard_size, max_shard_bytes):
    """Write ordered (inputs, targets) records; return manifest and its digest.

    Limits bound record count and serialized bytes per shard, not source memory,
    tensor construction peak, individual record size, total disk, or process RSS.
    An absent manifest means incomplete output; caller supplies external limits.
    """
    if any(type(x) is not int or x <= 0 for x in (shard_size, max_shard_bytes)):
        raise ValueError('shard limits must be positive integers')
    out = Path(output)
    out.mkdir(parents=False, exist_ok=False)
    iterator = iter(records)
    shards = []
    total = 0
    schema = None
    while True:
        chunk = []
        for _ in range(shard_size):
            try:
                chunk.append(next(iterator))
            except StopIteration:
                break
        if not chunk:
            break
        schema = _record_schema(chunk, schema)
        inputs, targets = _batch(chunk, range(len(chunk)), 'cpu')
        payload = {'inputs': inputs, 'targets': targets}
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        data = buffer.getvalue()
        if len(data) > max_shard_bytes:
            raise ValueError('serialized shard exceeds max_shard_bytes')
        name = f'shard-{len(shards):06d}.pt'
        with (out / name).open('xb') as stream:
            stream.write(data)
        shards.append({'file': name, 'start': total, 'count': len(chunk),
                       'bytes': len(data), 'sha256': _sha(data)})
        total += len(chunk)
    if not total:
        raise ValueError('empty records')
    manifest = {'version': VERSION, 'count': total, 'shard_size': shard_size,
                'max_shard_bytes': max_shard_bytes, 'shards': shards}
    data = json.dumps(manifest, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    with (out / 'manifest.json').open('xb') as stream:
        stream.write(data)
    return manifest, _sha(data)


def _validate_manifest(manifest):
    """Validate the complete pinned index before exposing any shard."""
    if (not isinstance(manifest, dict) or
            set(manifest) != {'version', 'count', 'shard_size', 'max_shard_bytes', 'shards'} or
            manifest['version'] != VERSION or
            any(type(manifest[k]) is not int or manifest[k] <= 0
                for k in ('count', 'shard_size', 'max_shard_bytes')) or
            not isinstance(manifest['shards'], list) or not manifest['shards']):
        raise ValueError('invalid manifest')
    offset = 0
    for index, shard in enumerate(manifest['shards']):
        if (not isinstance(shard, dict) or
                set(shard) != {'file', 'start', 'count', 'bytes', 'sha256'} or
                shard['file'] != f'shard-{index:06d}.pt' or
                type(shard['start']) is not int or shard['start'] != offset or
                type(shard['count']) is not int or not 0 < shard['count'] <= manifest['shard_size'] or
                (index < len(manifest['shards']) - 1 and shard['count'] != manifest['shard_size']) or
                type(shard['bytes']) is not int or not 0 < shard['bytes'] <= manifest['max_shard_bytes'] or
                not isinstance(shard['sha256'], str) or len(shard['sha256']) != 64 or
                any(c not in '0123456789abcdef' for c in shard['sha256'])):
            raise ValueError('invalid shard sequence')
        offset += shard['count']
    if offset != manifest['count']:
        raise ValueError('incomplete shard sequence')


def read_shards(output, expected_manifest_sha256):
    """Yield verified tensor batches in order; pin must come from independent custody."""
    out = Path(output)
    data = (out / 'manifest.json').read_bytes()
    if _sha(data) != expected_manifest_sha256:
        raise ValueError('manifest digest mismatch')
    manifest = json.loads(data)
    _validate_manifest(manifest)
    for index, shard in enumerate(manifest['shards']):
        raw = (out / shard['file']).read_bytes()
        if len(raw) != shard['bytes'] or _sha(raw) != shard['sha256']:
            raise ValueError('shard digest mismatch')
        payload = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
        if set(payload) != {'inputs', 'targets'} or any(
                not isinstance(group, dict) or not group or any(
                    not isinstance(t, torch.Tensor) or t.ndim == 0 or t.shape[0] != shard['count']
                    for t in group.values()) for group in payload.values()):
            raise ValueError('invalid tensor payload')
        yield payload


def verify_reconstruction(output, expected_manifest_sha256, records):
    """Exhaust the pinned archive and compare all ordered records exactly.

    The caller must independently custody the manifest pin and supply the
    authoritative records. This is a reconstruction check, not a publication,
    identity, resource-limit, or custody gate. It retains one shard at a time.
    """
    iterator = iter(records)
    total = 0
    schema = None
    for payload in read_shards(output, expected_manifest_sha256):
        count = next(iter(payload['inputs'].values())).shape[0]
        chunk = []
        for _ in range(count):
            try:
                chunk.append(next(iterator))
            except StopIteration:
                raise ValueError('expected records ended early') from None
        schema = _record_schema(chunk, schema)
        expected = _batch(chunk, range(count), 'cpu')
        for name, group in zip(('inputs', 'targets'), expected):
            actual = payload[name]
            if set(actual) != set(group):
                raise ValueError('reconstruction field mismatch')
            for key, tensor in group.items():
                other = actual[key]
                if (other.dtype != tensor.dtype or other.shape != tensor.shape or
                        not torch.equal(other, tensor)):
                    raise ValueError('reconstruction tensor mismatch')
        total += count
    try:
        next(iterator)
    except StopIteration:
        return total
    raise ValueError('expected records contain extra entries')
