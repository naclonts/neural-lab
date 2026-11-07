"""Per-seed node-size mapping for NL-001 corpus generation and identity recipe."""
import json
import unittest

from neural_lab.nl001_alignment_audit import audit_expected_identities, digest
from neural_lab.nl001_corpus import make_sampled_corpus
from neural_lab.nl001_data import make_splits, node_sizes
from neural_lab.nl001_identity_manifest import construct


SEEDS = {'train': [100, 101], 'validation': [200], 'test': [300]}
NODES = {100: 10, 101: 12, 200: 14, 300: 16}
BASE_CONFIG = dict(edge_probability=0.08, missing_fraction=0.25)
CONFIG = dict(BASE_CONFIG, per_stratum=2, sampling_seed=20261106)


class NodeSizeMappingTests(unittest.TestCase):
    def test_uniform_int_path_is_byte_identical_to_uniform_mapping(self):
        uniform = make_splits(SEEDS, nodes=10, **BASE_CONFIG)
        mapped = make_splits(SEEDS, nodes={seed: 10 for seeds in SEEDS.values() for seed in seeds},
                             **BASE_CONFIG)
        # Only the stored config may differ; programs and split digests are identical.
        self.assertEqual(uniform['splits'], mapped['splits'])
        self.assertEqual(uniform['split_sha256'], mapped['split_sha256'])
        self.assertNotEqual(uniform['sha256'], mapped['sha256'])

    def test_per_seed_sizes_control_symbol_counts(self):
        payload = make_splits(SEEDS, nodes=NODES, **BASE_CONFIG)
        for split, programs in payload['splits'].items():
            for program in programs:
                self.assertEqual(len(program['symbols']), NODES[program['seed']])
        self.assertEqual(payload['config']['nodes'], NODES)

    def test_string_keys_regenerate_identical_payload_after_json_roundtrip(self):
        corpus = make_sampled_corpus(SEEDS, nodes=NODES, **CONFIG)
        reloaded = json.loads(json.dumps(corpus))
        config = reloaded['config']
        regenerated = make_sampled_corpus(config['split_seeds'], nodes=config['nodes'],
                                          edge_probability=config['edge_probability'],
                                          missing_fraction=config['missing_fraction'],
                                          per_stratum=config['per_stratum'],
                                          sampling_seed=config['sampling_seed'])
        self.assertEqual(regenerated, reloaded)
        body = {key: value for key, value in reloaded.items() if key != 'sha256'}
        self.assertEqual(digest(body), corpus['sha256'])

    def test_invalid_mappings_rejected(self):
        for bad in ({}, {100: 10}, {**{s: 10 for seeds in SEEDS.values() for s in seeds}, 999: 10},
                    {**NODES, 100: 9}, {**NODES, 100: 51}, {**NODES, 100: True},
                    {'x': 10, 101: 12, 200: 14, 300: 16}, {100.0: 10, 101: 12, 200: 14, 300: 16},
                    {'100': 10, 100: 10, 101: 12, 200: 14, 300: 16}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                node_sizes(bad, [100, 101, 200, 300])
        with self.assertRaises(ValueError):
            make_splits(SEEDS, nodes={100: 10, 101: 12, 200: 14, 300: 16, 301: 10}, **BASE_CONFIG)

    def test_identity_manifest_matches_multisize_corpus(self):
        identities = construct(SEEDS, nodes=NODES, **CONFIG)
        corpus = make_sampled_corpus(SEEDS, nodes=NODES, **CONFIG)
        counts = {split: [{'seed': p['seed'], 'questions_per_variant': len(p['variants'][0]['questions'])}
                          for p in programs] for split, programs in corpus['splits'].items()}
        report = audit_expected_identities(corpus, corpus['sha256'], counts, identities, digest(identities))
        self.assertEqual(report['version'], 'nl001-alignment-identity-audit-v1')
        with self.assertRaises(ValueError):
            construct(SEEDS, nodes={100: 10, 101: 12, 200: 14}, **CONFIG)


if __name__ == '__main__':
    unittest.main()
