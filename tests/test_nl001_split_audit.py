import unittest
from neural_lab.nl001_split_audit import audit_observed_views


def program(seed, full, incomplete):
    names = [f'{seed}_{i}' for i in range(3)]
    return {'variants': [{'symbols': names, 'source': seed,
                          'full_edges': [[names[a], names[b]] for a, b in full],
                          'observed_incomplete_edges': [[names[a], names[b]] for a, b in incomplete]}]}


class SplitAuditTests(unittest.TestCase):
    def test_detects_cross_view_overlap_despite_different_full_graphs(self):
        splits = {'train': [program('a', [(0, 1), (1, 2)], [(0, 1)])],
                  'validation': [program('b', [(0, 1)], [])],
                  'test': [program('c', [(0, 0), (1, 1), (2, 2)], [(0, 0), (1, 1)])]}
        with self.assertRaisesRegex(ValueError, 'isomorphic observed views'):
            audit_observed_views(splits)

    def test_budget_exhaustion_is_not_accepted(self):
        splits = {name: [program(name, [], [])] for name in ('train', 'validation', 'test')}
        with self.assertRaisesRegex(RuntimeError, 'budget exhausted'):
            audit_observed_views(splits, max_states=1)

    def test_distinct_views_pass(self):
        splits = {'train': [program('a', [], [])],
                  'validation': [program('b', [(0, 1)], [(0, 1)])],
                  'test': [program('c', [(0, 0), (1, 1)], [(0, 0), (1, 1)])]}
        self.assertEqual(audit_observed_views(splits)['comparisons'], 12)

class AggregateBudgetTests(unittest.TestCase):
    def test_complete_count_cap_is_enforced_before_any_comparison(self):
        from unittest.mock import patch
        splits = {'train': [program('a', [], [])],
                  'validation': [program('b', [(0, 1)], [(0, 1)])],
                  'test': [program('c', [(0, 0), (1, 1)], [(0, 0), (1, 1)])]}
        with patch('neural_lab.nl001_split_audit.isomorphic') as compare:
            with self.assertRaisesRegex(RuntimeError, '12 required, 11 allowed'):
                audit_observed_views(splits, max_comparisons=11)
            compare.assert_not_called()
        report = audit_observed_views(splits, max_comparisons=12)
        self.assertEqual((report['comparisons'], report['required_comparisons'], report['max_comparisons']), (12, 12, 12))
        for invalid in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                audit_observed_views(splits, max_comparisons=invalid)

class AggregateStateBudgetTests(unittest.TestCase):
    def test_total_state_limit_across_comparisons_and_boundary(self):
        from unittest.mock import patch
        splits = {'train': [program('a', [], [])],
                  'validation': [program('b', [(0, 1)], [(0, 1)])],
                  'test': [program('c', [(0, 0), (1, 1)], [(0, 0), (1, 1)])]}

        def comparison(a, b, *, max_states, state_callback):
            state_callback()
            state_callback()
            return False

        with patch('neural_lab.nl001_split_audit.isomorphic', side_effect=comparison):
            report = audit_observed_views(splits, max_states=3, max_total_states=24)
            self.assertEqual((report['comparisons'], report['total_states']), (12, 24))
            with self.assertRaisesRegex(RuntimeError, 'aggregate isomorphism state budget exhausted'):
                audit_observed_views(splits, max_states=3, max_total_states=23)
        for value in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                audit_observed_views(splits, max_total_states=value)

    def test_real_isomorphism_visits_are_counted(self):
        from neural_lab.nl001_data import isomorphic
        from neural_lab.synthetic_graph import Graph
        names = tuple('abcdef')
        one_cycle = Graph(names, frozenset((names[i], names[(i + 1) % 6]) for i in range(6)), True, 'one')
        two_cycles = Graph(names, frozenset((names[i], names[(i // 3) * 3 + (i + 1) % 3]) for i in range(6)), True, 'two')
        states = []
        self.assertFalse(isomorphic(one_cycle, two_cycles, state_callback=lambda: states.append(1)))
        self.assertGreater(len(states), 1)

    def test_real_audit_exact_aggregate_boundary(self):
        cycle = [(i, (i + 1) % 6) for i in range(6)]
        triangles = [(i, (i // 3) * 3 + (i + 1) % 3) for i in range(6)]
        def six(name, edges):
            names = [f'{name}_{i}' for i in range(6)]
            return {'variants': [{'symbols': names, 'source': name,
                                  'full_edges': [[names[a], names[b]] for a, b in edges],
                                  'observed_incomplete_edges': [[names[a], names[b]] for a, b in edges]}]}
        splits = {'train': [six('a', cycle)], 'validation': [six('b', triangles)],
                  'test': [program('c', [], [])]}
        total = audit_observed_views(splits)['total_states']
        self.assertGreater(total, 1)
        self.assertEqual(audit_observed_views(splits, max_total_states=total)['total_states'], total)
        with self.assertRaisesRegex(RuntimeError, 'aggregate isomorphism state budget exhausted'):
            audit_observed_views(splits, max_total_states=total - 1)
