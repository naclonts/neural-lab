import copy
import itertools
import unittest

from neural_lab.nl001_independent_oracle import audit, certificate_valid, truth
from neural_lab.nl001_corpus import make_sampled_corpus
from neural_lab.synthetic_graph import Graph, check


class IndependentOracleTests(unittest.TestCase):
    def test_exhaustive_through_three_nodes(self):
        cases = 0
        for n in range(1, 4):
            names = [str(i) for i in range(n)]
            possible = list(itertools.product(names, repeat=2))
            for bits in range(1 << len(possible)):
                edges = [list(edge) for i, edge in enumerate(possible) if bits & (1 << i)]
                for complete in (False, True):
                    graph = Graph(tuple(names), frozenset(map(tuple, edges)), complete, 'independent-test')
                    for prop, a, b in itertools.product(('calls', 'reachable'), names, names):
                        q = {'property': prop, 'subject': a, 'object': b}
                        answer = check(graph, q, claim='fixture', expected_query=q)
                        self.assertEqual(answer['result'], truth(names, edges, complete, q))
                        self.assertTrue(certificate_valid(names, edges, complete, q, answer['result'], answer['certificate']))
                        cases += 1
        self.assertEqual(cases, 18696)

    def test_adversarial(self):
        names = ['a', 'b', 'c']
        edges = [['a', 'b'], ['b', 'a']]
        q = {'property': 'reachable', 'subject': 'a', 'object': 'a'}
        self.assertEqual(truth(names, edges, True, q), 'Proven')
        self.assertFalse(certificate_valid(names, edges, True, q, 'Proven', {'kind': 'path', 'nodes': ['a', 'a']}))
        self.assertFalse(certificate_valid(names, edges, False, q, 'Refuted', {'kind': 'closed_set', 'nodes': ['a']}))
        for bad in ({'property': 'calls', 'subject': 'z', 'object': 'a'}, {'property': 'unsupported', 'subject': 'a', 'object': 'a'}, {'property': 'calls'}):
            self.assertEqual(truth(names, edges, True, bad), 'Unknown')
            self.assertTrue(certificate_valid(names, edges, True, bad, 'Unknown', None))
        negative = {'property': 'reachable', 'subject': 'a', 'object': 'b'}
        self.assertFalse(certificate_valid(names, [], True, negative, 'Refuted', {'kind': 'closed_set', 'nodes': ['a', 'b']}))
        self.assertEqual(truth(names, [], False, q), 'Unknown')
        self.assertFalse(certificate_valid(names, [], False, q, 'Unknown', {'kind': 'closed_set', 'nodes': ['a']}))
        with self.assertRaises(ValueError):
            truth(names, [['a', 'z']], True, q)

    def test_generated_and_tampered_corpus(self):
        corpus = make_sampled_corpus({'train': [11], 'validation': [12], 'test': [13]}, per_stratum=2)
        self.assertGreater(audit(corpus)['claims'], 0)
        broken = copy.deepcopy(corpus)
        broken['splits']['train'][0]['variants'][0]['questions'][0]['label'] = 'Supported'
        with self.assertRaises(ValueError):
            audit(broken)
        broken = copy.deepcopy(corpus)
        broken['splits']['train'][0]['variants'][0]['removed_edges'] = []
        with self.assertRaises(ValueError):
            audit(broken)


if __name__ == '__main__':
    unittest.main()
