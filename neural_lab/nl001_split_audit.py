"""Fail-closed cross-split isomorphism audit of actual observed graph views.

This audit does not establish independent generation, claim correctness, or
completeness. The exact backtracking check may exhaust its explicit budget.
"""
from .nl001_data import isomorphic
from .synthetic_graph import Graph


class IsomorphicViewsError(ValueError):
    """Observed views overlap across splits."""


def check_comparison_budget(splits, max_comparisons):
    """Reject an audit whose complete cross-split comparison count exceeds its cap."""
    if type(max_comparisons) is not int or max_comparisons <= 0:
        raise ValueError('max_comparisons must be a positive integer')
    if set(splits) != {'train', 'validation', 'test'}:
        raise ValueError('expected train, validation, and test splits')
    counts = []
    for split in ('train', 'validation', 'test'):
        programs = splits[split]
        if not programs or any(not program.get('variants') for program in programs):
            raise ValueError('empty split or program without presentations')
        counts.append(2 * sum(len(program['variants']) for program in programs))
    required = counts[0] * counts[1] + counts[0] * counts[2] + counts[1] * counts[2]
    if required > max_comparisons:
        raise RuntimeError(f'aggregate comparison budget exhausted: {required} required, {max_comparisons} allowed; split is not validated')
    return required


def audit_observed_views(splits, *, max_states=1_000_000, max_comparisons=1_000_000,
                         max_total_states=1_000_000):
    """Reject cross-split isomorphic full/incomplete views, including cross-view pairs.

    Accepts the sampled corpus split payload (one program with presentation
    variants). Variants are paired presentations and are checked for consistency
    by comparing every presentation against other splits, not as independent
    samples. Count recursive backtracking states across all comparisons; degree
    filtering and candidate iteration are not counted. Budget exhaustion fails
    closed. Return comparison and state counts for audit reporting.
    """
    if set(splits) != {'train', 'validation', 'test'}:
        raise ValueError('expected train, validation, and test splits')
    if type(max_states) is not int or max_states <= 0:
        raise ValueError('max_states must be a positive integer')
    if type(max_total_states) is not int or max_total_states <= 0:
        raise ValueError('max_total_states must be a positive integer')
    required = check_comparison_budget(splits, max_comparisons)
    total_states = 0

    def count_state():
        nonlocal total_states
        total_states += 1
        if total_states > max_total_states:
            raise RuntimeError('aggregate isomorphism state budget exhausted; split is not validated')
    views = {}
    for split, programs in splits.items():
        views[split] = []
        if not programs:
            raise ValueError('empty split')
        for program in programs:
            if not program.get('variants'):
                raise ValueError('program without presentations')
            for variant in program['variants']:
                for edge_key, complete in (('full_edges', True), ('observed_incomplete_edges', False)):
                    graph = Graph(tuple(variant['symbols']), frozenset(map(tuple, variant[edge_key])),
                                  complete, variant['source'] + ':' + edge_key)
                    views[split].append(graph)
    comparisons = 0
    ordered = ('train', 'validation', 'test')
    for index, left in enumerate(ordered):
        for right in ordered[index + 1:]:
            for a in views[left]:
                for b in views[right]:
                    comparisons += 1
                    if isomorphic(a, b, max_states=max_states, state_callback=count_state):
                        raise IsomorphicViewsError(f'isomorphic observed views across {left} and {right}: {a.source}, {b.source}')
    return {'comparisons': comparisons, 'required_comparisons': required,
            'max_comparisons': max_comparisons, 'max_states_per_comparison': max_states,
            'total_states': total_states, 'max_total_states': max_total_states}
