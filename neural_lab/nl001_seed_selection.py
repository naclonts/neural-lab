"""Offline selection from a prospectively committed ordered list of corpora.

This does not freeze a protocol or isolate test data. The caller must retain the
commitment independently before generation and audit the selected corpus.
"""
from .nl001_corpus import make_sampled_corpus
from .nl001_data import digest, IsomorphicProgramsError
from .nl001_split_audit import audit_observed_views, IsomorphicViewsError


def select_corpus(plan, expected_plan_sha256):
    """Return the first candidate passing full and observed-view isolation.

    The plan includes ordered candidate seed maps, all generation parameters,
    and all observed-view isomorphism budgets. No quality or model metrics enter
    selection. Only proven graph overlap causes a retry; all other errors,
    including budget exhaustion, propagate and abort selection.
    """
    if digest(plan) != expected_plan_sha256:
        raise ValueError('seed selection plan digest mismatch')
    if not isinstance(plan, dict) or set(plan) != {'version', 'candidates', 'generation', 'max_states', 'max_comparisons', 'max_total_states'} or plan['version'] != 'nl001-seed-selection-v2':
        raise ValueError('invalid seed selection plan')
    candidates = plan['candidates']
    generation = plan['generation']
    if not isinstance(candidates, list) or not candidates or not isinstance(generation, dict) or set(generation) != {'nodes', 'edge_probability', 'missing_fraction', 'per_stratum', 'sampling_seed'}:
        raise ValueError('invalid seed selection plan')
    for key in ('max_states', 'max_comparisons', 'max_total_states'):
        if type(plan[key]) is not int or plan[key] <= 0:
            raise ValueError('invalid isomorphism budget: ' + key)
    for candidate in candidates:
        if not isinstance(candidate, dict) or set(candidate) != {'train', 'validation', 'test'} or any(not isinstance(v, list) or not v for v in candidate.values()):
            raise ValueError('invalid candidate seeds')
        seeds = [s for values in candidate.values() for s in values]
        if any(type(s) is not int for s in seeds) or len(seeds) != len(set(seeds)):
            raise ValueError('candidate seeds must be distinct integers')
    rejected = []
    for index, seeds in enumerate(candidates):
        try:
            corpus = make_sampled_corpus(seeds, **generation)
            audit = audit_observed_views(corpus['splits'], max_states=plan['max_states'],
                                         max_comparisons=plan['max_comparisons'],
                                         max_total_states=plan['max_total_states'])
        except (IsomorphicProgramsError, IsomorphicViewsError) as error:
            rejected.append({'index': index, 'reason': str(error)})
            continue
        return corpus, {'version': 'nl001-seed-selection-result-v2',
                        'plan_sha256': expected_plan_sha256, 'selected_index': index,
                        'rejected': rejected, 'corpus_sha256': corpus['sha256'],
                        'observed_view_audit': audit}
    raise ValueError('all precommitted candidates have isomorphic graph overlap')
