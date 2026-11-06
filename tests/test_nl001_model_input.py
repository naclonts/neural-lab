import unittest
from neural_lab.nl001_data import make_splits, model_input
from neural_lab.nl001_model_input import encode_input


class ModelInputTests(unittest.TestCase):
    def test_generator_inputs_and_renaming(self):
        payload = make_splits({'train': [11], 'validation': [12], 'test': [13]})
        for programs in payload['splits'].values():
            p = programs[0]
            for q in p['questions']:
                encoded = encode_input(model_input(p, q))
                self.assertEqual(sum(map(sum, encoded['adjacency'])), len(p['full_edges'] if q['view'] == 'complete' else p['observed_incomplete_edges']))
        item = {'claim': 'Can a reach b by calls?', 'symbols': ['a', 'b'], 'observed_edges': [['a', 'b']], 'complete': True}
        renamed = {'claim': 'Can x reach y by calls?', 'symbols': ['x', 'y'], 'observed_edges': [['x', 'y']], 'complete': True}
        for key in encode_input(item):
            self.assertEqual(encode_input(item)[key], encode_input(renamed)[key])

    def test_rejects_extra_fields_and_invalid_edges(self):
        item = {'claim': 'Does a call b?', 'symbols': ['a', 'b'], 'observed_edges': [], 'complete': True}
        with self.assertRaises(ValueError):
            encode_input({**item, 'label': 'Proven'})
        with self.assertRaises(ValueError):
            encode_input({**item, 'observed_edges': [['a', 'c']]})

if __name__ == '__main__':
    unittest.main()
