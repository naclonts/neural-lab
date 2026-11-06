"""Independently reconstruct sampled positional identities from a recipe.

No candidate corpus, generator, or checker is read. This mirrors the v1 recipe,
including Python's random.Random/shuffle and lexical symbol ordering; pin the
Python runtime and this file before use. It does not establish corpus custody.
"""
import hashlib
import json
import random


def _digest(value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def construct(split_seeds, *, nodes, edge_probability, missing_fraction, per_stratum, sampling_seed):
    if not isinstance(split_seeds, dict) or set(split_seeds) != {'train', 'validation', 'test'}:
        raise ValueError('invalid splits')
    if type(nodes) is not int or not 10 <= nodes <= 50:
        raise ValueError('invalid nodes')
    seeds = [s for split in ('train', 'validation', 'test') for s in split_seeds[split]]
    if (any(not isinstance(split_seeds[s], (list, tuple)) or not split_seeds[s] for s in split_seeds)
            or any(type(s) is not int for s in seeds) or len(seeds) != len(set(seeds))):
        raise ValueError('invalid seeds')
    if (isinstance(edge_probability, bool) or not 0 < edge_probability < 1
            or isinstance(missing_fraction, bool) or not 0 < missing_fraction <= 1
            or type(per_stratum) is not int or per_stratum <= 0 or type(sampling_seed) is not int):
        raise ValueError('invalid sampling configuration')
    result = {}
    for split in ('train', 'validation', 'test'):
        result[split] = []
        for seed in split_seeds[split]:
            names = [f'{split[0]}{seed}_{i}' for i in range(nodes)]
            rng = random.Random(seed)
            full = {(a, b) for a in names for b in names if rng.random() < edge_probability}
            if not full:
                raise ValueError('empty graph')
            shuffled = sorted(full)
            rng.shuffle(shuffled)
            observed = full - set(shuffled[:max(1, round(len(shuffled) * missing_fraction))])
            groups = {}
            for view, edges in [('complete', full), ('incomplete', observed)]:
                # Nonempty reachability: initialize from outgoing neighbors, not self.
                reach = {}
                for a in names:
                    seen, todo = set(), [b for x, b in edges if x == a]
                    while todo:
                        b = todo.pop()
                        if b not in seen:
                            seen.add(b)
                            todo.extend(y for x, y in edges if x == b and y not in seen)
                    reach[a] = seen
                for prop in ('calls', 'reachable', 'unsupported'):
                    pairs = [(names[0], names[-1])] if prop == 'unsupported' else [(a, b) for a in names for b in names]
                    for a, b in pairs:
                        positive = (a, b) in edges if prop == 'calls' else b in reach[a]
                        label = ('Unknown' if prop == 'unsupported' else 'Proven' if positive
                                 else 'Refuted' if view == 'complete' else 'Unknown')
                        groups.setdefault((view, prop, label), []).append((a, b))
            sample_rng = random.Random(sampling_seed + seed)
            identities = []
            source = f'nl001-whole-program-v1:split={split}:seed={seed}:nodes={nodes}:p={edge_probability}'
            positions = {name: i for i, name in enumerate(names)}
            for view, prop, label in sorted(groups):
                pairs = groups[(view, prop, label)][:]
                sample_rng.shuffle(pairs)
                for a, b in pairs[:per_stratum]:
                    query = {'property': prop, 'subject': a, 'object': b}
                    pair_id = _digest({'source': source, 'view': view, 'query': query})
                    identities.append([view, prop, positions[a], positions[b], pair_id])
            result[split].append({'seed': seed, 'questions': sorted(identities)})
    return result
