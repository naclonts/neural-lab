"""Offline audit report for an exact, independently pinned sampled corpus.

This regenerates with the same generator and runs a bounded graph-isomorphism
check. It is not an independent semantic audit. The input includes test data
and must be handled outside the selection environment; its digest and report
must be independently retained.
"""
from .nl001_data import digest
from .nl001_manifest import make_manifest
from .nl001_split_audit import audit_observed_views, check_comparison_budget
from .nl001_corpus import make_sampled_corpus


def audit_corpus(corpus, expected_corpus_sha256, *, max_states=1_000_000, max_comparisons=1_000_000, max_total_states=1_000_000):
    manifest = make_manifest(corpus)
    if corpus['sha256'] != expected_corpus_sha256:
        raise ValueError('corpus does not match independently retained digest')
    check_comparison_budget(corpus['splits'], max_comparisons)
    config = corpus['config']
    regenerated = make_sampled_corpus(
        config['split_seeds'], nodes=config['nodes'],
        edge_probability=config['edge_probability'], missing_fraction=config['missing_fraction'],
        per_stratum=config['per_stratum'], sampling_seed=config['sampling_seed'])
    if regenerated != corpus:
        raise ValueError('corpus does not match deterministic regeneration')
    comparison = audit_observed_views(corpus['splits'], max_states=max_states, max_comparisons=max_comparisons,
                                      max_total_states=max_total_states)
    return {'version': 'nl001-corpus-audit-v2', 'corpus_sha256': corpus['sha256'],
            'manifest': manifest, 'manifest_sha256': digest(manifest),
            'regeneration': 'exact match using current generator and checker',
            'observed_view_audit': comparison,
            'limitations': 'Regeneration shares the generator and checker; it does not independently establish semantic correctness or historical provenance. The aggregate state cap applies only to the observed-view audit, not regeneration, preprocessing, memory, disk, or wall time.'}
