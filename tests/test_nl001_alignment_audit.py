import copy
import unittest
from neural_lab.nl001_alignment_audit import audit, digest, parse


def fixture():
    splits = {}
    counts = {}
    for i, split in enumerate(('train', 'validation', 'test')):
        variants = []
        for held, renamed in ([(False, False), (False, True), (True, False), (True, True)] if split == 'test' else [(False, False)]):
            names = ['z0', 'z1'] if renamed else ['a?', 'b!']
            a, b = names
            claim = f'Is {b} reachable from {a} through one or more calls?' if held else f'Can {a} reach {b} by calls?'
            variants.append({'seed': i, 'symbols': names, 'presentation': {'heldout_template': held, 'renamed': renamed},
                'full_edges': [[a, b]], 'observed_incomplete_edges': [], 'removed_edges': [[a, b]],
                'questions': [{'claim': claim, 'query': {'property': 'reachable', 'subject': a, 'object': b}, 'view': 'complete', 'label': 'Proven', 'pair_id': 'a' * 64, 'template_id': ('test' if held else 'train') + ':reachable'}]})
        splits[split] = [{'seed': i, 'variants': variants}]
        counts[split] = [{'seed': i, 'questions_per_variant': 1}]
    corpus = {'version': 'nl001-sampled-crossed-v1', 'splits': splits}
    corpus['sha256'] = digest(corpus)
    return corpus, counts


def repin(c):
    c['sha256'] = digest({k: v for k, v in c.items() if k != 'sha256'})
    return c['sha256']


class AlignmentTests(unittest.TestCase):
    def test_literal_and_ambiguity(self):
        self.assertEqual(parse('Is b! reachable from a? through one or more calls?', ['a?', 'b!']), ('test', 'reachable', 'a?', 'b!'))
        with self.assertRaises(ValueError):
            parse('Does a call b call c?', ['a', 'a call b', 'b call c', 'c'])
        with self.assertRaises(ValueError):
            parse('Does a call x?', ['a', 'b'])

    def test_mutations_fail(self):
        base, counts = fixture()
        audit(base, base['sha256'], counts)
        mutations = [
            lambda q: q.update(claim='Can b! reach a? by calls?'),
            lambda q: q.update(claim='Does a? call b!?'),
            lambda q: q.update(claim='Does a? eventually execute b!?'),
            lambda q: q.update(claim='Can a? reach b! by calls.'),
            lambda q: q['query'].update(subject='b!', object='a?'),
            lambda q: q['query'].update(property='calls'),
            lambda q: q.update(view='bad'),
        ]
        for mutate in mutations:
            c = copy.deepcopy(base); mutate(c['splits']['train'][0]['variants'][0]['questions'][0])
            with self.assertRaises(ValueError): audit(c, repin(c), counts)
        c = copy.deepcopy(base); c['splits']['test'][0]['variants'][0]['presentation']['renamed'] = True
        with self.assertRaises(ValueError): audit(c, repin(c), counts)
        c = copy.deepcopy(base); c['splits']['test'][0]['variants'][0]['questions'].clear()
        with self.assertRaises(ValueError): audit(c, repin(c), counts)
        c = copy.deepcopy(base); c['splits']['test'][0]['variants'][0]['questions'].append(copy.deepcopy(c['splits']['test'][0]['variants'][0]['questions'][0]))
        with self.assertRaises(ValueError): audit(c, repin(c), counts)
        for field in ('full_edges', 'observed_incomplete_edges', 'removed_edges'):
            c = copy.deepcopy(base)
            c['splits']['test'][0]['variants'][1][field] = [['z1', 'z0']]
            with self.assertRaises(ValueError): audit(c, repin(c), counts)
        c = copy.deepcopy(base)
        c['splits']['test'][0]['variants'][1]['full_edges'].append(['z0', 'z1'])
        with self.assertRaises(ValueError): audit(c, repin(c), counts)
        c = copy.deepcopy(base); c['splits']['train'][0]['variants'][0]['questions'][0]['claim'] += 'x'
        with self.assertRaises(ValueError): audit(c, base['sha256'], counts)

    def test_unsupported(self):
        c, counts = fixture(); q = c['splits']['train'][0]['variants'][0]['questions'][0]
        q.update(claim='Does a? eventually execute b!?', query={'property': 'unsupported', 'subject': 'a?', 'object': 'b!'}, template_id='train:unsupported', label='Unknown')
        audit(c, repin(c), counts)
        q['label'] = 'Proven'
        with self.assertRaises(ValueError): audit(c, repin(c), counts)

    def test_graph_pairing_edge_order_and_invalid_endpoints(self):
        c, counts = fixture()
        for variant in c['splits']['test'][0]['variants']:
            a, b = variant['symbols']
            variant['full_edges'] = [[a, b], [b, a]]
        c['splits']['test'][0]['variants'][1]['full_edges'].reverse()
        audit(c, repin(c), counts)
        for invalid in (['z0', 'absent'], ['z0'], ['z0', 1]):
            mutated = copy.deepcopy(c)
            mutated['splits']['test'][0]['variants'][1]['full_edges'][0] = invalid
            with self.assertRaises(ValueError):
                audit(mutated, repin(mutated), counts)

class IdentityAuditTests(unittest.TestCase):
    def test_external_identities_reject_coordinated_substitution(self):
        from neural_lab.nl001_alignment_audit import audit_expected_identities
        corpus, counts = fixture()
        expected = {s: [{'seed': i, 'questions': [['complete', 'reachable', 0, 1, 'a' * 64]]}]
                    for i, s in enumerate(('train', 'validation', 'test'))}
        anchor = digest(expected)
        audit_expected_identities(corpus, corpus['sha256'], counts, expected, anchor)
        for variant in corpus['splits']['test'][0]['variants']:
            a, b = variant['symbols']
            q = variant['questions'][0]
            q['query'].update(subject=b, object=a)
            q['claim'] = (f'Is {a} reachable from {b} through one or more calls?'
                          if variant['presentation']['heldout_template'] else f'Can {b} reach {a} by calls?')
        repin(corpus)
        audit(corpus, corpus['sha256'], counts)
        with self.assertRaisesRegex(ValueError, 'question identity mismatch'):
            audit_expected_identities(corpus, corpus['sha256'], counts, expected, anchor)
        expected['test'][0]['questions'][0][2] = 1
        with self.assertRaisesRegex(ValueError, 'manifest digest mismatch'):
            audit_expected_identities(corpus, corpus['sha256'], counts, expected, anchor)

    def test_invalid_and_duplicate_manifest_entries_fail(self):
        from neural_lab.nl001_alignment_audit import audit_expected_identities
        corpus, counts = fixture()
        expected = {s: [{'seed': i, 'questions': [['complete', 'reachable', 0, 1, 'a' * 64]]}]
                    for i, s in enumerate(('train', 'validation', 'test'))}
        for bad in (['complete', 'reachable', True, 1, 'a' * 64],
                    ['complete', 'reachable', 0, 2, 'a' * 64]):
            altered = copy.deepcopy(expected)
            altered['train'][0]['questions'] = [bad]
            with self.assertRaises(ValueError):
                audit_expected_identities(corpus, corpus['sha256'], counts, altered, digest(altered))
        expected['train'][0]['questions'] *= 2
        with self.assertRaisesRegex(ValueError, 'duplicate expected'):
            audit_expected_identities(corpus, corpus['sha256'], counts, expected, digest(expected))


if __name__ == '__main__':
    unittest.main()
