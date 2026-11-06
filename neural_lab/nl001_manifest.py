"""Digest-pinned sampled-corpus split ownership.

The expected manifest digest must be frozen independently before using a split.
Hashes detect changes; they do not establish who created or audited a corpus.
"""
from .nl001_data import digest
from .nl001_corpus import VERSION


def make_manifest(corpus):
    if not isinstance(corpus, dict) or corpus.get('version') != VERSION:
        raise ValueError('unsupported corpus')
    body = {k: v for k, v in corpus.items() if k != 'sha256'}
    if corpus.get('sha256') != digest(body):
        raise ValueError('corpus digest mismatch')
    splits = corpus.get('splits')
    if not isinstance(splits, dict) or set(splits) != {'train', 'validation', 'test'}:
        raise ValueError('invalid corpus splits')
    seeds = []
    descriptors = {}
    for name, entries in splits.items():
        if not isinstance(entries, list) or not entries:
            raise ValueError('empty or invalid split')
        split_seeds = [e['seed'] for e in entries]
        if any(type(s) is not int for s in split_seeds):
            raise ValueError('invalid seed')
        seeds.extend(split_seeds)
        if corpus.get('split_sha256', {}).get(name) != digest(entries):
            raise ValueError('split digest mismatch')
        descriptors[name] = {'sha256': digest(entries), 'seeds': split_seeds}
    if len(set(seeds)) != len(seeds) or any(
        descriptors[name]['seeds'] != corpus['config']['split_seeds'][name] for name in splits
    ):
        raise ValueError('split seed ownership mismatch')
    return {'version': 'nl001-split-manifest-v1', 'corpus_sha256': corpus['sha256'], 'splits': descriptors}


def authenticate_split(entries, split, manifest, expected_manifest_sha256):
    if not isinstance(expected_manifest_sha256, str) or digest(manifest) != expected_manifest_sha256:
        raise ValueError('manifest does not match independently pinned digest')
    if not isinstance(manifest, dict) or manifest.get('version') != 'nl001-split-manifest-v1':
        raise ValueError('invalid manifest')
    if split not in ('train', 'validation', 'test'):
        raise ValueError('invalid split')
    try:
        descriptor = manifest['splits'][split]
        if digest(entries) != descriptor['sha256'] or [e['seed'] for e in entries] != descriptor['seeds']:
            raise ValueError('split payload does not match manifest')
    except (KeyError, TypeError) as exc:
        raise ValueError('invalid split manifest or payload') from exc
    return entries
