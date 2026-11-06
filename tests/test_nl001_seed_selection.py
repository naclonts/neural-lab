import unittest
from unittest.mock import patch

from neural_lab.nl001_data import digest, IsomorphicProgramsError
from neural_lab.nl001_corpus import make_sampled_corpus
from neural_lab.nl001_seed_selection import select_corpus
from neural_lab.nl001_split_audit import IsomorphicViewsError, audit_observed_views


class SeedSelectionTests(unittest.TestCase):
    def setUp(self):
        self.plan = {'version': 'nl001-seed-selection-v2',
                     'candidates': [{'train': [11], 'validation': [12], 'test': [13]},
                                    {'train': [21], 'validation': [22], 'test': [23]}],
                     'generation': {'nodes': 10, 'edge_probability': .08, 'missing_fraction': .25,
                                    'per_stratum': 1, 'sampling_seed': 20261106}, 'max_states': 1000000,
                     'max_comparisons': 1000000, 'max_total_states': 1000000}

    def test_real_corpus_is_deterministic_and_all_views_audited(self):
        plan = {**self.plan, 'candidates': self.plan['candidates'][:1]}
        corpus, report = select_corpus(plan, digest(plan))
        self.assertEqual(report['selected_index'], 0)
        self.assertEqual(report['corpus_sha256'], corpus['sha256'])
        self.assertEqual(report['observed_view_audit'], audit_observed_views(corpus['splits']))
        self.assertGreater(report['observed_view_audit']['comparisons'], 0)
        self.assertEqual((corpus, report), select_corpus(plan, digest(plan)))

    def test_only_proven_overlap_can_advance_to_next_candidate(self):
        corpus = {'sha256': 'selected', 'splits': {}}
        for failure in (IsomorphicProgramsError('full overlap'), IsomorphicViewsError('view overlap')):
            with self.subTest(failure=failure), patch('neural_lab.nl001_seed_selection.make_sampled_corpus', side_effect=[failure, corpus]), patch('neural_lab.nl001_seed_selection.audit_observed_views', return_value={'comparisons': 1}):
                _, report = select_corpus(self.plan, digest(self.plan))
                self.assertEqual(report['selected_index'], 1)
                self.assertEqual(report['rejected'], [{'index': 0, 'reason': str(failure)}])
        with patch('neural_lab.nl001_seed_selection.make_sampled_corpus', side_effect=RuntimeError('budget exhausted')) as generate:
            with self.assertRaisesRegex(RuntimeError, 'budget exhausted'):
                select_corpus(self.plan, digest(self.plan))
            self.assertEqual(generate.call_count, 1)
        with patch('neural_lab.nl001_seed_selection.make_sampled_corpus', return_value=corpus), patch('neural_lab.nl001_seed_selection.audit_observed_views', side_effect=RuntimeError('budget exhausted')):
            with self.assertRaisesRegex(RuntimeError, 'budget exhausted'):
                select_corpus(self.plan, digest(self.plan))

    def test_digest_and_entire_plan_validated_before_generation(self):
        with patch('neural_lab.nl001_seed_selection.make_sampled_corpus') as generate:
            with self.assertRaisesRegex(ValueError, 'digest mismatch'):
                select_corpus(self.plan, 'bad')
            bad = {**self.plan, 'candidates': [self.plan['candidates'][0], {'train': [1], 'validation': [1], 'test': [2]}]}
            with self.assertRaisesRegex(ValueError, 'distinct'):
                select_corpus(bad, digest(bad))
            generate.assert_not_called()

    def test_exhaustion_fails(self):
        with patch('neural_lab.nl001_seed_selection.make_sampled_corpus', side_effect=IsomorphicProgramsError('overlap')):
            with self.assertRaisesRegex(ValueError, 'all precommitted candidates'):
                select_corpus(self.plan, digest(self.plan))

    def test_committed_limits_are_enforced_and_never_trigger_retry(self):
        for key, limit, message in [('max_comparisons', 1, 'comparison budget exhausted'),
                                    ('max_total_states', 1, 'state budget exhausted')]:
            plan = {**self.plan, key: limit}
            def consume_states(a, b, *, max_states, state_callback):
                state_callback()
                state_callback()
                return False
            with self.subTest(key=key), patch('neural_lab.nl001_seed_selection.make_sampled_corpus', wraps=make_sampled_corpus) as generate, patch('neural_lab.nl001_split_audit.isomorphic', side_effect=consume_states):
                with self.assertRaisesRegex(RuntimeError, message):
                    select_corpus(plan, digest(plan))
                self.assertEqual(generate.call_count, 1)

    def test_missing_or_invalid_limits_rejected_before_generation(self):
        with patch('neural_lab.nl001_seed_selection.make_sampled_corpus') as generate:
            for key in ('max_states', 'max_comparisons', 'max_total_states'):
                for invalid in (0, True):
                    plan = {**self.plan, key: invalid}
                    with self.subTest(key=key, invalid=invalid), self.assertRaisesRegex(ValueError, 'budget'):
                        select_corpus(plan, digest(plan))
                plan = {k: v for k, v in self.plan.items() if k != key}
                with self.assertRaisesRegex(ValueError, 'plan'):
                    select_corpus(plan, digest(plan))
            generate.assert_not_called()


if __name__ == '__main__':
    unittest.main()
