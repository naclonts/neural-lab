import unittest

from neural_lab.nl001_training_data import example, verify_selected
from neural_lab.synthetic_graph import Graph, replay


class TrainingDataTests(unittest.TestCase):
    def setUp(self):
        self.program = {'symbols': ['n0', 'n1', 'n2'], 'full_edges': [['n0', 'n1'], ['n1', 'n2']],
                        'observed_incomplete_edges': [['n0', 'n1']], 'source': 'fixture'}
        self.q = {'view': 'complete', 'claim': 'Can n0 reach n2 by calls?',
                  'query': {'property': 'reachable', 'subject': 'n0', 'object': 'n2'},
                  'label': 'Proven', 'certificate': {'kind': 'path', 'nodes': ['n0', 'n1', 'n2']}}

    def test_witness_and_input_separation(self):
        inputs, targets = example(self.program, self.q)
        self.assertEqual(set(inputs), {'tokens', 'pointers', 'adjacency', 'node_count', 'complete'})
        self.assertEqual(targets['witness'][0][1], 1)
        self.assertEqual(targets['witness'][1][2], 1)
        bad = dict(self.q, certificate={'kind': 'path', 'nodes': ['n0', 'n2']})
        with self.assertRaises(ValueError):
            example(self.program, bad)

    def test_selected_positive_and_full_graph_negative(self):
        raw = {'claim': self.q['claim'], 'symbols': self.program['symbols'],
               'observed_edges': self.program['full_edges'], 'complete': True}
        q = self.q['query']
        self.assertEqual(verify_selected(raw, q, [])[ 'result'], 'Unknown')
        answer = verify_selected(raw, q, raw['observed_edges'])
        graph = Graph(tuple(raw['symbols']), frozenset(map(tuple, raw['observed_edges'])), True, 'fixture')
        self.assertTrue(replay(graph, q, answer['result'], answer['certificate']))
        negative = {'property': 'reachable', 'subject': 'n2', 'object': 'n0'}
        answer = verify_selected(raw, negative, [])
        self.assertEqual(answer['result'], 'Refuted')
        self.assertTrue(replay(graph, negative, answer['result'], answer['certificate']))
        raw['complete'] = False
        self.assertEqual(verify_selected(raw, negative, [])['result'], 'Unknown')
        with self.assertRaises(ValueError):
            verify_selected(raw, q, [['n0', 'n2']])

if __name__ == '__main__':
    unittest.main()
