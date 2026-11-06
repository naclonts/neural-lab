import copy
import unittest
from unittest.mock import patch

from neural_lab.nl001_corpus import make_sampled_corpus
from neural_lab.nl001_corpus_audit import audit_corpus
from neural_lab.nl001_data import digest


class CorpusRegenerationTests(unittest.TestCase):
    def test_rehashed_label_tampering_is_rejected(self):
        corpus = make_sampled_corpus({'train': [11], 'validation': [12], 'test': [13]}, per_stratum=1)
        with patch('neural_lab.nl001_corpus_audit.audit_observed_views', return_value={'comparisons': 0}):
            self.assertIn('exact match', audit_corpus(corpus, corpus['sha256'])['regeneration'])
            altered = copy.deepcopy(corpus)
            question = altered['splits']['train'][0]['variants'][0]['questions'][0]
            question['label'] = 'Refuted' if question['label'] != 'Refuted' else 'Proven'
            altered['split_sha256']['train'] = digest(altered['splits']['train'])
            altered['sha256'] = digest({k: v for k, v in altered.items() if k != 'sha256'})
            with self.assertRaisesRegex(ValueError, 'deterministic regeneration'):
                audit_corpus(altered, altered['sha256'])


if __name__ == '__main__':
    unittest.main()
