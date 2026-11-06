import unittest

from neural_lab.nl001_alignment_audit import audit_expected_identities, digest
from neural_lab.nl001_corpus import make_sampled_corpus
from neural_lab.nl001_identity_manifest import construct


class IdentityManifestTests(unittest.TestCase):
    def test_independent_recipe_matches_sampled_fixture(self):
        seeds = {'train': [100], 'validation': [200], 'test': [300]}
        config = dict(nodes=10, edge_probability=0.08, missing_fraction=0.25,
                      per_stratum=4, sampling_seed=20261106)
        identities = construct(seeds, **config)
        corpus = make_sampled_corpus(seeds, **config)
        counts = {split: [{'seed': p['seed'], 'questions_per_variant': len(p['variants'][0]['questions'])}
                          for p in programs] for split, programs in corpus['splits'].items()}
        report = audit_expected_identities(corpus, corpus['sha256'], counts, identities, digest(identities))
        self.assertEqual(report['expected_identities_sha256'], digest(identities))
        identities['test'][0]['questions'].pop()
        with self.assertRaisesRegex(ValueError, 'question identity mismatch'):
            audit_expected_identities(corpus, corpus['sha256'], counts, identities, digest(identities))

    def test_invalid_recipe_rejected(self):
        with self.assertRaises(ValueError):
            construct({'train': [1], 'validation': [1], 'test': [2]}, nodes=10,
                      edge_probability=.08, missing_fraction=.25, per_stratum=4, sampling_seed=0)


if __name__ == '__main__':
    unittest.main()
