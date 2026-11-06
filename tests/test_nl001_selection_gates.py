import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import numpy as np

from neural_lab.artifacts import ArtifactWriter, BestCheckpoint
from neural_lab.nl001_data import digest
from neural_lab.nl001_corpus import make_sampled_corpus
from neural_lab.nl001_manifest import make_manifest
from neural_lab.nl001_models import CalibrationModel
from neural_lab.nl001_selection_runner import run, _audit_gates, _check_audit_details
from neural_lab.nl001_training import _save_training_artifacts, _save_best_checkpoint


class SelectionGates(unittest.TestCase):
    def test_independent_claim_verification_is_checked_separately(self):
        expected = {'status': 'Proven', 'query_aligned': False, 'replayed': True, 'correct': False}
        detail = dict(expected, identity='row', claim_verified=False)
        _check_audit_details([detail], [{'identity': 'row'}], [expected])
        for changed in (dict(detail, claim_verified=True), dict(detail, replayed=False),
                        dict(detail, identity='other'), dict(detail, unexpected=True)):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                _check_audit_details([changed], [{'identity': 'row'}], [expected])

    def test_complete_reports_are_required_and_bound(self):
        import copy
        manifest = {'corpus_sha256': 'corpus', 'splits': {n: {'seeds': [i]} for i, n in enumerate(('train', 'validation', 'test'))}}
        audit = {'version': 'nl001-corpus-audit-v2', 'regeneration': 'exact match using current generator and checker',
                 'manifest': manifest, 'manifest_sha256': digest(manifest), 'corpus_sha256': 'corpus',
                 'observed_view_audit': {'comparisons': 36, 'required_comparisons': 36, 'max_comparisons': 36, 'max_states_per_comparison': 10}}
        identity = {'version': 'nl001-alignment-identity-audit-v1', 'corpus_sha256': 'corpus',
                    'expected_identities_sha256': 'a' * 64, 'question_counts': {'train': 1, 'validation': 1, 'test': 4}}
        semantic = {'version': 'nl001-semantic-count-attestation-v1', 'corpus_sha256': 'corpus',
                    'semantic_audit_sha256': 'b' * 64, 'claims': 6, 'question_counts': identity['question_counts'].copy()}
        def check(a, i, s):
            _audit_gates(manifest, digest(manifest), a, digest(a), i, digest(i), 'a' * 64, s, digest(s))
        check(audit, identity, semantic)
        for target, path, value in ((0, ('observed_view_audit', 'comparisons'), 1),
                                    (0, ('observed_view_audit', 'required_comparisons'), 1),
                                    (1, ('corpus_sha256',), 'other'),
                                    (1, ('version',), 'old'),
                                    (2, ('claims',), 5),
                                    (2, ('corpus_sha256',), 'other')):
            reports = copy.deepcopy([audit, identity, semantic])
            obj = reports[target]
            for key in path[:-1]: obj = obj[key]
            obj[path[-1]] = value
            with self.subTest(path=path), self.assertRaises(ValueError): check(*reports)
        with self.assertRaises(ValueError):
            _audit_gates(manifest, digest(manifest), audit, digest(audit), identity, 'bad', 'a' * 64, semantic, digest(semantic))
        with self.assertRaises(ValueError):
            _audit_gates(manifest, digest(manifest), audit, digest(audit), identity, digest(identity), 'a' * 64, semantic, 'bad')

    def test_real_audits_are_accepted_without_test_label_breakdown(self):
        from neural_lab.nl001_identity_manifest import construct
        from neural_lab.nl001_alignment_audit import audit_expected_identities
        from neural_lab.nl001_corpus_audit import audit_corpus
        from neural_lab.nl001_independent_oracle import audit as audit_semantics
        seeds = {'train': [11], 'validation': [12], 'test': [13]}
        recipe = dict(nodes=10, edge_probability=.08, missing_fraction=.25, per_stratum=1, sampling_seed=20261106)
        identities = construct(seeds, **recipe)
        corpus = make_sampled_corpus(seeds, **recipe)
        manifest = make_manifest(corpus)
        counts = {n: [{'seed': p['seed'], 'questions_per_variant': len(p['questions'])} for p in ps]
                  for n, ps in identities.items()}
        identity = audit_expected_identities(corpus, corpus['sha256'], counts, identities, digest(identities))
        audit = audit_corpus(corpus, corpus['sha256'])
        semantic_result = audit_semantics(corpus)
        totals = {n: sum(row['count'] for row in semantic_result['counts'] if row['split'] == n) for n in seeds}
        semantic = {'version': 'nl001-semantic-count-attestation-v1', 'corpus_sha256': corpus['sha256'],
                    'claims': semantic_result['claims'], 'question_counts': totals, 'semantic_audit_sha256': 'b' * 64}
        _audit_gates(manifest, digest(manifest), audit, digest(audit), identity, digest(identity),
                     digest(identities), semantic, digest(semantic))
        semantic['result'] = semantic_result
        with self.assertRaises(ValueError):
            _audit_gates(manifest, digest(manifest), audit, digest(audit), identity, digest(identity),
                         digest(identities), semantic, digest(semantic))

    def test_saved_checkpoint_pipeline_without_training(self):
        corpus = make_sampled_corpus({'train': [11], 'validation': [12], 'test': [13]}, per_stratum=1)
        manifest = make_manifest(corpus)
        pin = digest(manifest)
        # This fabricated audit exercises wiring only; it is not an independent corpus audit.
        audit = {'version': 'nl001-corpus-audit-v2', 'regeneration': 'exact match using current generator and checker',
                 'manifest': manifest, 'manifest_sha256': pin, 'corpus_sha256': manifest['corpus_sha256'],
                 'observed_view_audit': {'comparisons': 1, 'max_states_per_comparison': 10}, 'question_counts': {'train': 0, 'validation': 0}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, value in [('train', corpus['splits']['train']), ('validation', corpus['splits']['validation']),
                                ('manifest', manifest), ('audit', audit)]:
                (root / name).write_text(json.dumps(value))
            def untrained_fit(**kwargs):
                entry = ArtifactWriter(kwargs['output']).save_checkpoint(
                    'best-' + kwargs['arm'], CalibrationModel(kwargs['arm']), arm=kwargs['arm'],
                    seed=kwargs['seed'], selection={'scope': 'untrained fixture'})
                return {'checkpoint_sha256': entry['sha256']}
            audit['question_counts'] = {n: sum(len(v['questions']) for p in corpus['splits'][n] for v in p['variants']) for n in ('train', 'validation')}
            (root / 'audit').write_text(json.dumps(audit))
            with patch('neural_lab.nl001_selection_runner._audit_gates'), patch('neural_lab.nl001_selection_runner.fit_fixed_updates', side_effect=untrained_fit):
                result = run(train_path=root/'train', validation_path=root/'validation', manifest_path=root/'manifest',
                             manifest_sha256=pin, audit_path=root/'audit', audit_sha256=digest(audit), identity_audit_path=root/'audit', identity_audit_sha256='pin', expected_identities_sha256='pin', semantic_audit_path=root/'audit', semantic_audit_sha256='pin',
                             output=root/'out', seed=1, batch_size=1, updates=1, learning_rate=.001, device='cpu')
            self.assertGreater(result['reports']['B']['evaluation']['counts']['claims'], 0)
            for arm in ('A', 'B'):
                self.assertTrue((root/'out'/f'{arm}-independent-audit.json').is_file())
                self.assertTrue((root/'out'/f'{arm}-replay.json').is_file())

    def test_subprocess_failures_and_prediction_corruption_never_publish_summary(self):
        for failure in ('inference_error', 'inference_timeout', 'changed_prediction', 'replay_error', 'audit_error'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest = {'identity': 'fixture'}
                pin = digest(manifest)
                audit = {'version': 'nl001-corpus-audit-v2', 'regeneration': 'exact match using current generator and checker',
                         'manifest': manifest, 'manifest_sha256': pin,
                         'observed_view_audit': {'comparisons': 1, 'max_states_per_comparison': 10}, 'question_counts': {'train': 0, 'validation': 0}}
                for name, value in [('train', []), ('validation', []), ('manifest', manifest), ('audit', audit)]:
                    (root / name).write_text(json.dumps(value))
                def infer(**kwargs):
                    kwargs['output'].write_text('{"predictions": []}')
                    return digest({'predictions': []})
                def subprocess_call(command, **kwargs):
                    stage = 'inference' if '--infer' in command else ('replay' if '--replay' in command else 'audit')
                    if failure == stage + '_error':
                        raise subprocess.CalledProcessError(1, command)
                    if failure == 'inference_timeout' and stage == 'inference':
                        raise subprocess.TimeoutExpired(command, 120)
                    if stage == 'inference':
                        request = json.loads(Path(command[-1]).read_text())
                        Path(request['output']).write_text(json.dumps({'predictions': [1] if failure == 'changed_prediction' else []}))
                    else:
                        Path(command[-1]).write_text(json.dumps({'counts': {key: 0 for key in ('claims', 'replay_attempts', 'replay_passed', 'replay_rejected', 'aligned_label_conflicts')}, 'details': [], 'predictions': []}))
                with patch('neural_lab.nl001_selection_runner._audit_gates'), patch('neural_lab.nl001_selection_runner.authenticate_split'), \
                     patch('neural_lab.nl001_selection_runner.evaluate_split', return_value={'counts': {'aligned_label_conflicts': 0}}), \
                     patch('neural_lab.nl001_selection_runner.fit_fixed_updates', return_value={'checkpoint_sha256': 'pin'}), \
                     patch('neural_lab.nl001_selection_runner.infer_checkpoint', side_effect=infer), \
                     patch('neural_lab.nl001_selection_runner.subprocess.run', side_effect=subprocess_call):
                    with self.assertRaises((ValueError, subprocess.CalledProcessError, subprocess.TimeoutExpired)):
                        run(train_path=root/'train', validation_path=root/'validation', manifest_path=root/'manifest',
                            manifest_sha256=pin, audit_path=root/'audit', audit_sha256=digest(audit), identity_audit_path=root/'audit', identity_audit_sha256='pin', expected_identities_sha256='pin', semantic_audit_path=root/'audit', semantic_audit_sha256='pin', output=root/'out',
                            seed=1, batch_size=1, updates=1, learning_rate=.001, device='cpu')
                self.assertTrue((root/'out'/'protocol.json').exists())
                self.assertFalse((root/'out'/'selection-summary.json').exists())

    def test_best_checkpoint_copies_pre_update_weights_and_keeps_earliest_tie(self):
        model = torch.nn.Linear(1, 1, bias=False)
        best = BestCheckpoint()
        for step, (weight, loss) in enumerate(((1, 3), (2, 1), (3, 1), (4, 2))):
            with torch.no_grad():
                model.weight.fill_(weight)
            best.observe(model, loss, step)
        self.assertEqual(best.step, 1)
        self.assertEqual(best.value, 1)
        self.assertEqual(best.state['weight'].item(), 2)
        self.assertEqual(best.observed_steps, [0, 1, 2, 3])

    def test_training_tensor_artifact_preserves_order_and_checkpoint_binding(self):
        records = [({'tokens': [1, 2], 'complete': True}, {'status': 2}),
                   ({'tokens': [3, 0], 'complete': False}, {'status': 1})]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            writer = ArtifactWriter(root)
            source, tensors = _save_training_artifacts(writer, [{'fixture': 1}], {'manifest': 1}, 'manifest-pin', records)
            with np.load(root / tensors['path'], allow_pickle=False) as data:
                np.testing.assert_array_equal(data['input_tokens'], [[1, 2], [3, 0]])
                np.testing.assert_array_equal(data['input_complete'], [True, False])
                np.testing.assert_array_equal(data['target_status'], [2, 1])
                self.assertEqual(set(data.files), {'input_tokens', 'input_complete', 'target_status'})
            self.assertEqual(tensors['source']['programs_sha256'], source['sha256'])
            self.assertEqual(tensors['source']['manifest_sha256'], 'manifest-pin')
            model = torch.nn.Linear(1, 1)
            best = BestCheckpoint()
            best.observe(model, 0.5, 0)
            checkpoint = _save_best_checkpoint(best, writer, 'A', 7, {}, source, tensors)
            self.assertEqual(checkpoint['datasets'], [source['sha256'], tensors['sha256']])
            manifest = json.loads((root / 'artifacts.json').read_text())
            self.assertEqual(manifest['checkpoints'][0]['datasets'], checkpoint['datasets'])

    def test_audit_gate_rejects_mismatched_pin_before_fit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, value in [('train', []), ('validation', []), ('manifest', {}), ('audit', {})]:
                (root / name).write_text(json.dumps(value))
            with patch('neural_lab.nl001_selection_runner.fit_fixed_updates') as fit:
                with self.assertRaisesRegex(ValueError, 'corpus audit'):
                    run(train_path=root/'train', validation_path=root/'validation', manifest_path=root/'manifest',
                        manifest_sha256=digest({}), audit_path=root/'audit', audit_sha256='bad', identity_audit_path=root/'audit', identity_audit_sha256='pin', expected_identities_sha256='pin', semantic_audit_path=root/'audit', semantic_audit_sha256='pin', output=root/'out',
                        seed=1, batch_size=1, updates=1, learning_rate=.001, device='cpu')
                fit.assert_not_called()

    def test_runner_wires_audited_inputs_to_mocked_pipeline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {'identity': 'fixture'}
            manifest_pin = digest(manifest)
            audit = {'version': 'nl001-corpus-audit-v2', 'regeneration': 'exact match using current generator and checker', 'manifest': manifest,
                     'manifest_sha256': manifest_pin,
                     'observed_view_audit': {'comparisons': 1, 'max_states_per_comparison': 10}, 'question_counts': {'train': 0, 'validation': 0}}
            for name, value in [('train', []), ('validation', []), ('manifest', manifest), ('audit', audit)]:
                (root / name).write_text(json.dumps(value))
            def fake_fit(**kwargs):
                return {'checkpoint_sha256': 'fixture-checkpoint'}
            def fake_infer(**kwargs):
                kwargs['output'].write_text('{"predictions": []}')
                return digest({'predictions': []})
            def fake_replay(command, **kwargs):
                if '--infer' in command:
                    request = json.loads(Path(command[-1]).read_text())
                    Path(request['output']).write_text('{"predictions": []}')
                    return
                Path(command[-1]).write_text(json.dumps({'counts': {key: 0 for key in ('claims', 'replay_attempts', 'replay_passed', 'replay_rejected', 'aligned_label_conflicts')}, 'details': [], 'predictions': []}))
            with patch('neural_lab.nl001_selection_runner._audit_gates'), patch('neural_lab.nl001_selection_runner.authenticate_split'), \
                 patch('neural_lab.nl001_selection_runner.evaluate_split', return_value={'counts': {'aligned_label_conflicts': 0}}), \
                 patch('neural_lab.nl001_selection_runner.fit_fixed_updates', side_effect=fake_fit) as fit, \
                 patch('neural_lab.nl001_selection_runner.infer_checkpoint', side_effect=fake_infer) as infer, \
                 patch('neural_lab.nl001_selection_runner.subprocess.run', side_effect=fake_replay):
                result = run(train_path=root/'train', validation_path=root/'validation', manifest_path=root/'manifest',
                             manifest_sha256=manifest_pin, audit_path=root/'audit', audit_sha256=digest(audit), identity_audit_path=root/'audit', identity_audit_sha256='pin', expected_identities_sha256='pin', semantic_audit_path=root/'audit', semantic_audit_sha256='pin',
                             output=root/'out', seed=1, batch_size=1, updates=1, learning_rate=.001, device='cpu')
            self.assertEqual(fit.call_count, 2)
            self.assertEqual(infer.call_count, 2)
            self.assertTrue(all('best-' in str(call.kwargs['checkpoint']) for call in infer.call_args_list))
            self.assertEqual(result['protocol']['audit_sha256'], digest(audit))
            self.assertTrue((root/'out'/'selection-summary.json').is_file())


if __name__ == '__main__':
    unittest.main()
