"""Independent literal-form alignment audit for nl001-sampled-crossed-v1.

This deliberately does not import generator templates or the checker. It is not
an arbitrary-language parser or a certificate, isomorphism, or custody audit.
"""
import hashlib
import json

FORMS = {
    ('train', 'calls'): ('Does ', ' call ', '?', False),
    ('train', 'reachable'): ('Can ', ' reach ', ' by calls?', False),
    ('train', 'unsupported'): ('Does ', ' eventually execute ', '?', False),
    ('test', 'calls'): ('Is there a direct call from ', ' to ', '?', False),
    ('test', 'reachable'): ('Is ', ' reachable from ', ' through one or more calls?', True),
    ('test', 'unsupported'): ('Will ', ' eventually execute ', '?', False),
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def parse(claim, symbols):
    if not isinstance(claim, str) or not isinstance(symbols, list) or not symbols or any(not isinstance(s, str) or not s for s in symbols) or len(set(symbols)) != len(symbols):
        raise ValueError('invalid claim or symbols')
    matches = []
    # Enumerating literal symbol pairs avoids regex delimiter and punctuation ambiguity.
    for (form, prop), (prefix, middle, suffix, reverse) in FORMS.items():
        for a in symbols:
            for b in symbols:
                first, second = (b, a) if reverse else (a, b)
                if prefix + first + middle + second + suffix == claim:
                    matches.append((form, prop, a, b))
    if len(matches) != 1:
        raise ValueError('unrecognized or ambiguous claim')
    return matches[0]


def positional_edges(variant, symbols):
    """Represent each graph view in symbol-index space for pairing checks."""
    positions = {symbol: i for i, symbol in enumerate(symbols)}
    result = []
    for name in ('full_edges', 'observed_incomplete_edges', 'removed_edges'):
        edges = variant[name]
        if not isinstance(edges, list):
            raise ValueError('invalid edge list')
        normalized = set()
        for edge in edges:
            if not isinstance(edge, list) or len(edge) != 2 or any(not isinstance(x, str) or x not in positions for x in edge):
                raise ValueError('invalid edge')
            normalized.add((positions[edge[0]], positions[edge[1]]))
        if len(normalized) != len(edges):
            raise ValueError('duplicate edge')
        result.append(frozenset(normalized))
    return tuple(result)


def audit(corpus, expected_digest, expected_counts):
    if not isinstance(corpus, dict) or corpus.get('version') != 'nl001-sampled-crossed-v1':
        raise ValueError('unsupported corpus')
    body = {k: v for k, v in corpus.items() if k != 'sha256'}
    if corpus.get('sha256') != expected_digest or digest(body) != expected_digest:
        raise ValueError('corpus digest mismatch')
    splits = corpus['splits']
    if set(splits) != {'train', 'validation', 'test'} or set(expected_counts) != set(splits):
        raise ValueError('invalid split coverage')
    seeds_seen = set()
    totals = {}
    for split, programs in splits.items():
        if not programs or len(programs) != len(expected_counts[split]):
            raise ValueError('program count mismatch')
        totals[split] = 0
        for program, expected in zip(programs, expected_counts[split]):
            seed = program['seed']
            if seed in seeds_seen or expected['seed'] != seed:
                raise ValueError('seed identity mismatch')
            seeds_seen.add(seed)
            variants = program['variants']
            order = [(False, False), (False, True), (True, False), (True, True)] if split == 'test' else [(False, False)]
            if len(variants) != len(order):
                raise ValueError('presentation count mismatch')
            reference = None
            reference_symbols = None
            reference_edges = None
            for variant, (heldout, renamed) in zip(variants, order):
                if variant['seed'] != seed or variant['presentation'] != {'heldout_template': heldout, 'renamed': renamed}:
                    raise ValueError('presentation identity mismatch')
                symbols = variant['symbols']
                if not renamed:
                    if reference_symbols is None:
                        reference_symbols = symbols
                    elif symbols != reference_symbols:
                        raise ValueError('unrenamed symbols mismatch')
                elif reference_symbols is None or len(symbols) != len(reference_symbols):
                    raise ValueError('renamed symbol count mismatch')
                if renamed and symbols != [f'z{i}' for i in range(len(symbols))]:
                    raise ValueError('renaming mismatch')
                # parse validates the symbol table before positional comparison.
                if not variant['questions'] or len(variant['questions']) != expected['questions_per_variant']:
                    raise ValueError('question count mismatch')
                entries = set()
                logical = set()
                pair_ids = set()
                for q in variant['questions']:
                    form, prop, a, b = parse(q['claim'], symbols)
                    if form != ('test' if heldout else 'train') or q['template_id'] != f'{form}:{prop}':
                        raise ValueError('template mismatch')
                    if q['query'] != {'property': prop, 'subject': a, 'object': b} or q['view'] not in ('complete', 'incomplete'):
                        raise ValueError('query or view mismatch')
                    if prop == 'unsupported' and q['label'] != 'Unknown':
                        raise ValueError('unsupported claim must be Unknown')
                    key = (q['view'], prop, symbols.index(a), symbols.index(b))
                    pair_id = q['pair_id']
                    if not isinstance(pair_id, str) or len(pair_id) != 64 or any(c not in '0123456789abcdef' for c in pair_id):
                        raise ValueError('invalid pair identity')
                    if key in logical or pair_id in pair_ids:
                        raise ValueError('duplicate question or pair identity')
                    logical.add(key)
                    pair_ids.add(pair_id)
                    entries.add((*key, pair_id))
                graph = positional_edges(variant, symbols)
                if reference_edges is not None and graph != reference_edges:
                    raise ValueError('paired graph mismatch')
                reference_edges = graph
                if reference is not None and entries != reference:
                    raise ValueError('paired questions mismatch')
                reference = entries
                totals[split] += len(entries)
    return {'version': 'nl001-alignment-audit-v2', 'corpus_sha256': expected_digest,
            'question_counts': totals, 'limitations': 'Literal controlled forms only; expected counts and corpus digest require external custody. Does not verify graph semantics, certificates, split isomorphism, or complete expected question identities.'}


def audit_expected_identities(corpus, expected_digest, expected_counts, expected_identities,
                              expected_identities_sha256):
    """Require externally pinned positional question identities for every program.

    The identity manifest maps each split to an ordered list of programs, each
    containing ``seed`` and ``questions``. Questions are lists of
    [view, property, subject index, object index, pair_id]. Identity order within
    a program is immaterial. Construct and retain the manifest and its digest
    independently; deriving it from the candidate corpus is not an audit.
    """
    if digest(expected_identities) != expected_identities_sha256:
        raise ValueError('expected identity manifest digest mismatch')
    report = audit(corpus, expected_digest, expected_counts)
    if not isinstance(expected_identities, dict) or set(expected_identities) != {'train', 'validation', 'test'}:
        raise ValueError('invalid identity split coverage')
    for split, programs in corpus['splits'].items():
        expected_programs = expected_identities[split]
        if not isinstance(expected_programs, list) or len(expected_programs) != len(programs):
            raise ValueError('identity program count mismatch')
        for program, expected in zip(programs, expected_programs):
            if not isinstance(expected, dict) or set(expected) != {'seed', 'questions'} or type(expected['seed']) is not int or expected['seed'] != program['seed']:
                raise ValueError('identity seed mismatch')
            identities = expected['questions']
            if not isinstance(identities, list):
                raise ValueError('invalid question identities')
            normalized = set()
            for identity in identities:
                if (not isinstance(identity, list) or len(identity) != 5
                        or identity[0] not in ('complete', 'incomplete')
                        or identity[1] not in ('calls', 'reachable', 'unsupported')
                        or type(identity[2]) is not int or type(identity[3]) is not int
                        or not 0 <= identity[2] < len(program['variants'][0]['symbols'])
                        or not 0 <= identity[3] < len(program['variants'][0]['symbols'])
                        or not isinstance(identity[4], str) or len(identity[4]) != 64
                        or any(c not in '0123456789abcdef' for c in identity[4])):
                    raise ValueError('invalid question identity')
                normalized.add(tuple(identity))
            if len(normalized) != len(identities):
                raise ValueError('duplicate expected question identity')
            variant = program['variants'][0]
            symbols = variant['symbols']
            actual = {(q['view'], q['query']['property'], symbols.index(q['query']['subject']),
                       symbols.index(q['query']['object']), q['pair_id']) for q in variant['questions']}
            if actual != normalized:
                raise ValueError('question identity mismatch')
    return {**report, 'version': 'nl001-alignment-identity-audit-v1',
            'expected_identities_sha256': expected_identities_sha256,
            'limitations': 'Requires independently constructed and retained identity manifest and digest; does not verify graph semantics, certificates, or split isomorphism.'}
