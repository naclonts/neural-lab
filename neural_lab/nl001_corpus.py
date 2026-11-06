"""Sampled NL-001 corpus with crossed language-shift factors.

Program splits are created and isomorphism-checked by nl001_data. Each test
program has four paired presentation variants; variants are never independent
programs. Sampling is performed before presentation changes and uses only gold
strata for constructing this synthetic fixture, never for model inputs.
"""
import random

from .nl001_data import TEMPLATES, digest, make_splits
from .synthetic_graph import Graph, check

VERSION = "nl001-sampled-crossed-v1"


def _sample(questions, seed, per_stratum):
    groups = {}
    for q in questions:
        key = (q['view'], q['query']['property'], q['label'])
        groups.setdefault(key, []).append(q)
    rng = random.Random(seed)
    selected = []
    for key in sorted(groups):
        group = groups[key][:]
        rng.shuffle(group)
        selected.extend(group[:per_stratum])
    return sorted(selected, key=lambda q: (q['view'], q['query']['property'],
                                          q['query']['subject'], q['query']['object']))


def _present(program, questions, template_split, renamed):
    old = program['symbols']
    names = [f'z{i}' for i in range(len(old))] if renamed else old[:]
    mapping = dict(zip(old, names))
    def edges(key):
        return [[mapping[a], mapping[b]] for a, b in program[key]]
    full, incomplete = edges('full_edges'), edges('observed_incomplete_edges')
    result = []
    for q in questions:
        query = {**q['query'], 'subject': mapping[q['query']['subject']],
                 'object': mapping[q['query']['object']]}
        claim = TEMPLATES[template_split][query['property']].format(a=query['subject'], b=query['object'])
        complete = q['view'] == 'complete'
        facts = full if complete else incomplete
        graph = Graph(tuple(names), frozenset(map(tuple, facts)), complete,
                      program['source'] + ('' if complete else ':incomplete'))
        answer = check(graph, query, claim=claim, expected_query=query)
        if answer['result'] != q['label']:
            raise AssertionError('presentation changed the truth label')
        result.append({**q, 'pair_id': digest({'source': program['source'], 'view': q['view'], 'query': q['query']}), 'claim': claim, 'query': query, 'label': answer['result'],
                       'certificate': answer['certificate'], 'reason': answer['reason'],
                       'template_id': f"{template_split}:{query['property']}"})
    return {'seed': program['seed'], 'source': program['source'], 'symbols': names,
            'full_edges': full, 'observed_incomplete_edges': incomplete,
            'removed_edges': edges('removed_edges'), 'questions': result,
            'presentation': {'heldout_template': template_split == 'test', 'renamed': renamed}}


def make_sampled_corpus(split_seeds, *, nodes=10, edge_probability=0.08,
                        missing_fraction=0.25, per_stratum=4, sampling_seed=20261106):
    """Construct paired test presentations and cap each nonempty label stratum.

    Strata are (view, property, label) within each program. Rare or absent
    strata are not resampled; counts therefore vary. Four test presentations
    share identical programs/questions, and must be analyzed as paired data.
    """
    if type(per_stratum) is not int or per_stratum <= 0 or type(sampling_seed) is not int:
        raise ValueError('per_stratum must be a positive integer and sampling_seed an integer')
    base = make_splits(split_seeds, nodes=nodes, edge_probability=edge_probability,
                       missing_fraction=missing_fraction)
    splits = {}
    for split, programs in base['splits'].items():
        splits[split] = []
        for program in programs:
            questions = _sample(program['questions'], sampling_seed + program['seed'], per_stratum)
            if split == 'test':
                variants = [_present(program, questions, template, renamed)
                            for template in ('train', 'test') for renamed in (False, True)]
            else:
                variants = [_present(program, questions, 'train', False)]
            splits[split].append({'seed': program['seed'], 'variants': variants})
    payload = {'version': VERSION, 'base_version': base['version'],
               'base_sha256': base['sha256'], 'base_split_sha256': base['split_sha256'],
               'config': {**base['config'], 'per_stratum': per_stratum, 'sampling_seed': sampling_seed},
               'templates': base['templates'], 'checker_version': base['checker_version'], 'splits': splits,
               'split_sha256': {split: digest(items) for split, items in splits.items()}}
    payload['sha256'] = digest(payload)
    return payload
