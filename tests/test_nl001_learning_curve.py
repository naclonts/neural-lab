import tempfile
import unittest
from pathlib import Path

from neural_lab.nl001_corpus import make_sampled_corpus
from neural_lab.nl001_data import digest
from neural_lab.nl001_inference import _checkpoint_model
from neural_lab.nl001_learning_curve import fit_curve
from neural_lab.nl001_manifest import make_manifest


class CurveTest(unittest.TestCase):
    def test_exact_steps_and_reload(self):
        corpus = make_sampled_corpus({'train': [11], 'validation': [12], 'test': [13]}, per_stratum=1)
        manifest = make_manifest(corpus)
        with tempfile.TemporaryDirectory() as temp:
            args = dict(programs=corpus['splits']['train'], manifest=manifest,
                        manifest_sha256=digest(manifest), arm='A', seed=41, batch_size=4,
                        steps=[1, 3], learning_rate=0.001, device='cpu')
            result = fit_curve(output=Path(temp) / 'fit', **args)
            self.assertEqual(list(result['checkpoints']), ['0', '1', '3'])
            self.assertEqual(len(result['losses']), 3)
            self.assertEqual(result['completed_updates'], 3)
            self.assertEqual(result['actual_model_devices'], ['cpu'])
            self.assertEqual(result['exposures'], sum(result['batch_sizes']))
            for step, entry in result['checkpoints'].items():
                model = _checkpoint_model(Path(temp) / 'fit' / entry['path'], entry['sha256'], 'A')
                self.assertEqual(model.arm, 'A')
            self.assertNotEqual(result['checkpoints']['0']['sha256'], result['checkpoints']['3']['sha256'])
            with self.assertRaises(ValueError):
                fit_curve(output=Path(temp) / 'bad', **{**args, 'steps': [3, 1]})
            with self.assertRaises(ValueError):
                fit_curve(output=Path(temp) / 'bad2', **{**args, 'manifest_sha256': 'bad'})

    def test_incomplete_schedule_fails_closed(self):
        from unittest.mock import patch
        corpus = make_sampled_corpus({'train': [11], 'validation': [12], 'test': [13]}, per_stratum=1)
        manifest = make_manifest(corpus)
        with tempfile.TemporaryDirectory() as temp, patch(
                'neural_lab.nl001_learning_curve.batch_indices', return_value=iter([[0]])), patch(
                'neural_lab.nl001_learning_curve.update', return_value=0.0):
            with self.assertRaisesRegex(ValueError, 'completed update count'):
                fit_curve(programs=corpus['splits']['train'], manifest=manifest,
                    manifest_sha256=digest(manifest), output=Path(temp) / 'fit', arm='A',
                    seed=41, batch_size=4, steps=[3], learning_rate=0.001, device='cpu')

    def test_validation_runner_only_accepts_bound_validation(self):
        from neural_lab.nl001_learning_curve import run_curve
        corpus = make_sampled_corpus({'train': [11], 'validation': [12], 'test': [13]}, per_stratum=1)
        manifest = make_manifest(corpus)
        args = dict(train=corpus['splits']['train'], validation=corpus['splits']['validation'],
                    manifest=manifest, manifest_sha256=digest(manifest), seeds=[41], batch_size=4,
                    steps=[1], learning_rate=0.001, device='cpu')
        with tempfile.TemporaryDirectory() as temp:
            report = run_curve(output=Path(temp) / 'run', **args)
            self.assertEqual(set(report['fits']['41']), {'A', 'B'})
            self.assertEqual(report['protocol']['evaluation_split'], 'validation')
            for arm in ('A', 'B'):
                check = report['fits']['41'][arm]['validation']['1']['verification']
                self.assertGreaterEqual(check['fresh_process_inference_seconds'], 0)
                self.assertEqual(check['independent_audit']['arm'], arm)
            with self.assertRaises(ValueError):
                run_curve(output=Path(temp) / 'bad', **{**args, 'validation': corpus['splits']['test']})

    def test_fixed_states_match_independent_update_loop(self):
        import torch
        from neural_lab.nl001_models import CalibrationModel, update
        from neural_lab.nl001_training import _batch, _flatten, batch_indices
        corpus = make_sampled_corpus({'train': [11], 'validation': [12], 'test': [13]}, per_stratum=1)
        manifest = make_manifest(corpus)
        records = _flatten(corpus['splits']['train'])
        with tempfile.TemporaryDirectory() as temp:
            for arm in ('A', 'B'):
                result = fit_curve(programs=corpus['splits']['train'], manifest=manifest,
                    manifest_sha256=digest(manifest), output=Path(temp) / arm, arm=arm, seed=41,
                    batch_size=7, steps=[1, 3], learning_rate=0.001, device='cpu')
                torch.manual_seed(41)
                model = CalibrationModel(arm, hidden=192 if arm == 'A' else 128, embedding=32)
                optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999),
                                              eps=1e-8, weight_decay=0.01, amsgrad=False)
                schedule = list(batch_indices(len(records), 7, 3, 41))
                self.assertEqual(result['batch_sizes'], list(map(len, schedule)))
                observed = []
                for step, indices in enumerate(schedule, 1):
                    inputs, targets = _batch(records, indices, 'cpu')
                    def capture(current, ce):
                        observed.append((ce, {k: v.clone() for k, v in current.state_dict().items()}))
                    update(model, optimizer, inputs, targets, observe=capture)
                    if step in (1, 3):
                        entry = result['checkpoints'][str(step)]
                        loaded = _checkpoint_model(Path(temp) / arm / entry['path'], entry['sha256'], arm)
                        for key, value in model.state_dict().items():
                            self.assertTrue(torch.equal(value, loaded.state_dict()[key]), (arm, step, key))
                selected = result['best_checkpoint']
                minimum = min(range(len(observed)), key=lambda i: observed[i][0])
                self.assertEqual(selected['selection']['step'], minimum)
                self.assertEqual(selected['selection']['observations'], 3)
                self.assertEqual(selected['selection']['value'], observed[minimum][0])
                self.assertEqual([x['cross_entropy'] for x in result['training_ce_observations']],
                                 [x[0] for x in observed])
                loaded = _checkpoint_model(Path(temp) / arm / selected['path'], selected['sha256'], arm)
                for key, value in observed[minimum][1].items():
                    self.assertTrue(torch.equal(value, loaded.state_dict()[key]), (arm, key))

    def test_fresh_process_mismatch_fails_closed(self):
        import json
        from unittest.mock import patch
        from neural_lab.nl001_curve_replay import verify_checkpoint
        def corrupt(module, args, report):
            request = json.loads(Path(args[1]).read_text())
            Path(request['output']).write_text('{}')
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch('neural_lab.nl001_curve_replay._child', side_effect=corrupt):
                with self.assertRaisesRegex(ValueError, 'fresh-process'):
                    verify_checkpoint(programs=[], manifest={}, manifest_sha256='x',
                        checkpoint=root / 'checkpoint', checkpoint_sha256='x',
                        predictions=root / 'predictions', prediction_sha256='x', arm='A',
                        edge_threshold=0.5, evaluation={}, directory=root, step=1)

    def test_all_fits_finish_before_validation(self):
        from unittest.mock import patch
        import neural_lab.nl001_learning_curve as curve
        corpus = make_sampled_corpus({'train': [11], 'validation': [12], 'test': [13]}, per_stratum=1)
        manifest = make_manifest(corpus)
        original = curve.fit_curve
        completed = []
        def fit(**kwargs):
            result = original(**kwargs)
            completed.append((kwargs['seed'], kwargs['arm']))
            return result
        from neural_lab.nl001_run_evaluation import evaluate_split
        def control(**kwargs):
            self.assertEqual(len(completed), 4)
            return evaluate_split(**kwargs)
        with tempfile.TemporaryDirectory() as temp, patch.object(curve, 'fit_curve', side_effect=fit), \
                patch('neural_lab.nl001_run_evaluation.evaluate_split', side_effect=control):
            curve.run_curve(train=corpus['splits']['train'], validation=corpus['splits']['validation'],
                manifest=manifest, manifest_sha256=digest(manifest), output=Path(temp) / 'run',
                seeds=[41, 42], batch_size=4, steps=[1], learning_rate=0.001, device='cpu')
