import unittest
from neural_lab.nl001_corpus import make_sampled_corpus
from neural_lab.nl001_data import digest, model_input
from neural_lab.synthetic_graph import Graph, replay


class SampledCorpusTests(unittest.TestCase):
    def test_crossed_variants_are_paired_and_replay(self):
        args = {'train': [11], 'validation': [12], 'test': [13]}
        corpus = make_sampled_corpus(args, per_stratum=2)
        self.assertEqual(corpus, make_sampled_corpus(args, per_stratum=2))
        variants = corpus['splits']['test'][0]['variants']
        self.assertEqual({(v['presentation']['heldout_template'], v['presentation']['renamed']) for v in variants},
                         {(False, False), (False, True), (True, False), (True, True)})
        self.assertEqual(len({len(v['questions']) for v in variants}), 1)
        base = variants[0]
        for variant in variants:
            mapping = dict(zip(base['symbols'], variant['symbols']))
            for key in ('full_edges', 'observed_incomplete_edges', 'removed_edges'):
                self.assertEqual([[mapping[a], mapping[b]] for a, b in base[key]], variant[key])
            for left, right in zip(base['questions'], variant['questions']):
                self.assertEqual(left['pair_id'], right['pair_id'])
                self.assertEqual(left['view'], right['view'])
                self.assertEqual(mapping[left['query']['subject']], right['query']['subject'])
                self.assertEqual(mapping[left['query']['object']], right['query']['object'])
                template = corpus['templates']['test' if variant['presentation']['heldout_template'] else 'train'][right['query']['property']]
                self.assertEqual(right['claim'], template.format(a=right['query']['subject'], b=right['query']['object']))
        self.assertEqual(len({q['pair_id'] for q in base['questions']}), len(base['questions']))
        for split, programs in corpus['splits'].items():
            self.assertEqual(corpus['split_sha256'][split], digest(programs))
        self.assertEqual(corpus['sha256'], digest({k: v for k, v in corpus.items() if k != 'sha256'}))
        for rows in zip(*(v['questions'] for v in variants)):
            self.assertEqual(len({(q['view'], q['query']['property'], q['label']) for q in rows}), 1)
            self.assertNotEqual(rows[0]['claim'], rows[2]['claim'])
        for variant in variants:
            for q in variant['questions']:
                item = model_input(variant, q)
                graph = Graph(tuple(item['symbols']), frozenset(map(tuple, item['observed_edges'])),
                              item['complete'], variant['source'])
                if q['label'] in ('Proven', 'Refuted'):
                    self.assertTrue(replay(graph, q['query'], q['label'], q['certificate']))

    def test_invalid_sampling_parameters(self):
        seeds = {'train': [11], 'validation': [12], 'test': [13]}
        for value in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                make_sampled_corpus(seeds, per_stratum=value)

if __name__ == '__main__':
    unittest.main()
