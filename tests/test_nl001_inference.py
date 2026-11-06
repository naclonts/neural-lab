import hashlib
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from neural_lab.artifacts import ArtifactWriter, BestCheckpoint, load_json_state
from neural_lab.nl001_corpus import make_sampled_corpus
from neural_lab.nl001_data import digest
from neural_lab.nl001_manifest import make_manifest
from neural_lab.nl001_models import CalibrationModel
from neural_lab.nl001_inference import _checkpoint_model, infer_checkpoint, replay_saved
from neural_lab.nl001_prediction_audit import audit_predictions
from neural_lab.nl001_training import _flatten, _save_training_artifacts, _save_best_checkpoint


class InferenceTests(unittest.TestCase):
    def test_independent_audit_rejects_repinned_claim_query_mismatch_for_both_arms(self):
        corpus = make_sampled_corpus({'train': [11], 'validation': [12], 'test': [13]}, per_stratum=1)
        # Repinning makes custody valid: semantic disagreement must still fail.
        bad = copy.deepcopy(corpus)
        question = bad['splits']['test'][0]['variants'][2]['questions'][0]
        symbols = bad['splits']['test'][0]['variants'][2]['symbols']
        question['query']['subject'] = next(s for s in symbols if s != question['query']['subject'])
        bad['split_sha256']['test'] = digest(bad['splits']['test'])
        bad['sha256'] = digest({k: v for k, v in bad.items() if k != 'sha256'})
        manifest = make_manifest(bad)
        programs = bad['splits']['test']
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for arm in ('A', 'B'):
                entry = ArtifactWriter(root / arm).save_checkpoint('model', CalibrationModel(arm), arm=arm, seed=1, selection={})
                output = root / (arm + '.json')
                args = dict(programs=programs, split='test', manifest=manifest,
                            manifest_sha256=digest(manifest), checkpoint_sha256=entry['sha256'], arm=arm)
                pin = infer_checkpoint(**args, checkpoint=root / arm / entry['path'], output=output)
                with self.subTest(arm=arm), self.assertRaisesRegex(ValueError, 'claim/query alignment mismatch'):
                    audit_predictions(**args, predictions_path=output, predictions_sha256=pin)

    def test_independent_audit_rejects_repinned_invalid_view_for_both_arms(self):
        corpus = make_sampled_corpus({'train': [11], 'validation': [12], 'test': [13]}, per_stratum=1)
        bad = copy.deepcopy(corpus)
        bad['splits']['test'][0]['variants'][0]['questions'][0]['view'] = 'invalid'
        bad['split_sha256']['test'] = digest(bad['splits']['test'])
        bad['sha256'] = digest({k: v for k, v in bad.items() if k != 'sha256'})
        manifest = make_manifest(bad)
        programs = bad['splits']['test']
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for arm in ('A', 'B'):
                entry = ArtifactWriter(root / arm).save_checkpoint('model', CalibrationModel(arm), arm=arm, seed=1, selection={})
                output = root / (arm + '.json')
                args = dict(programs=programs, split='test', manifest=manifest,
                            manifest_sha256=digest(manifest), checkpoint_sha256=entry['sha256'], arm=arm)
                envelope = {'version': 'nl001-inference-v1', 'split': 'test', 'split_sha256': digest(programs),
                            'manifest_sha256': digest(manifest), 'checkpoint_sha256': entry['sha256'],
                            'arm': arm, 'edge_threshold': 0.5,
                            'predictions': [{'identity': digest({'program': v, 'question': q}), 'prediction': {}}
                                            for p in programs for v in p['variants'] for q in v['questions']]}
                output.write_text(json.dumps(envelope))
                pin = digest(envelope)
                with self.subTest(arm=arm), self.assertRaisesRegex(ValueError, 'invalid question view'):
                    audit_predictions(**args, predictions_path=output, predictions_sha256=pin)

    def test_checkpoint_rejects_corrupted_tensor_even_with_matching_pin(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            entry = ArtifactWriter(root).save_checkpoint('model', CalibrationModel('A'), arm='A', seed=1, selection={})
            path = root / entry['path']
            original = torch.load(path, map_location='cpu', weights_only=True)
            name = next(iter(original['tensors']))
            for corruption in ('dtype', 'shape', 'nonfinite', 'missing', 'extra'):
                value = {key: (value.clone() if isinstance(value, torch.Tensor) else value) for key, value in original.items()}
                value['tensors'] = {key: tensor.clone() for key, tensor in original['tensors'].items()}
                if corruption == 'dtype':
                    value['tensors'][name] = value['tensors'][name].to(torch.float64)
                elif corruption == 'shape':
                    value['tensors'][name] = value['tensors'][name][:1]
                elif corruption == 'nonfinite':
                    value['tensors'][name] = value['tensors'][name].fill_(float('inf'))
                elif corruption == 'missing':
                    del value['tensors'][name]
                elif corruption == 'extra':
                    value['tensors']['unexpected'] = torch.zeros(1)
                torch.save(value, path)
                pin = hashlib.sha256(path.read_bytes()).hexdigest()
                with self.subTest(corruption=corruption), self.assertRaises(ValueError):
                    _checkpoint_model(path, pin, 'A')

    @classmethod
    def setUpClass(cls):
        corpus = make_sampled_corpus({'train': [11], 'validation': [12], 'test': [13]}, per_stratum=1)
        cls.programs = corpus['splits']['validation']
        cls.manifest = make_manifest(corpus)
        cls.pin = digest(cls.manifest)

    def test_untrained_checkpoints_and_bound_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for arm in ('A', 'B'):
                writer = ArtifactWriter(root / arm)
                entry = writer.save_checkpoint('model', CalibrationModel(arm), arm=arm, seed=1, selection={})
                checkpoint = root / arm / entry['path']
                output = root / (arm + '.json')
                args = dict(programs=self.programs, split='validation', manifest=self.manifest,
                            manifest_sha256=self.pin, checkpoint_sha256=entry['sha256'], arm=arm)
                pin = infer_checkpoint(**args, checkpoint=checkpoint, output=output)
                report = replay_saved(**args, predictions_path=output, predictions_sha256=pin)
                self.assertGreater(report.get('claims', report.get('counts', {}).get('claims', 0)), 0)
                with self.assertRaises(ValueError):
                    replay_saved(**{**args, 'checkpoint_sha256': '0' * 64}, predictions_path=output, predictions_sha256=pin)
                envelope = json.loads(output.read_text())
                envelope['predictions'].reverse()
                output.write_text(json.dumps(envelope))
                with self.assertRaises(ValueError):
                    replay_saved(**args, predictions_path=output, predictions_sha256=pin)
                with self.assertRaises(ValueError):
                    replay_saved(**args, predictions_path=output, predictions_sha256=digest(envelope))
                checkpoint.write_bytes(checkpoint.read_bytes() + b' ')
                with self.assertRaises(ValueError):
                    infer_checkpoint(**args, checkpoint=checkpoint, output=root / 'unused.json')

    def test_corrupted_certificate_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            writer = ArtifactWriter(root)
            entry = writer.save_checkpoint('model', CalibrationModel('B'), arm='B', seed=1, selection={})
            output = root / 'out.json'
            args = dict(programs=self.programs, split='validation', manifest=self.manifest,
                        manifest_sha256=self.pin, checkpoint_sha256=entry['sha256'], arm='B')
            infer_checkpoint(**args, checkpoint=root / entry['path'], output=output)
            envelope = json.loads(output.read_text())
            envelope['predictions'][0]['prediction'] = {'query': self.programs[0]['variants'][0]['questions'][0]['query'], 'result': 'Proven', 'certificate': {'nodes': ['bad']}}
            output.write_text(json.dumps(envelope))
            report = replay_saved(**args, predictions_path=output, predictions_sha256=digest(envelope))
            self.assertGreater(report['counts']['replay_rejected'], 0)
            independent = audit_predictions(**args, predictions_path=output, predictions_sha256=digest(envelope))
            for key in ('claims', 'replay_attempts', 'replay_passed', 'replay_rejected', 'aligned_label_conflicts'):
                self.assertEqual(independent['counts'][key], report['counts'][key])
            with self.assertRaises(ValueError):
                audit_predictions(**args, predictions_path=output, predictions_sha256='0' * 64)

    def test_independent_audit_replays_positive_and_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            writer = ArtifactWriter(root)
            entry = writer.save_checkpoint('model', CalibrationModel('B'), arm='B', seed=1, selection={})
            output = root / 'out.json'
            args = dict(programs=self.programs, split='validation', manifest=self.manifest,
                        manifest_sha256=self.pin, checkpoint_sha256=entry['sha256'], arm='B')
            infer_checkpoint(**args, checkpoint=root / entry['path'], output=output)
            envelope = json.loads(output.read_text())
            items = [(v, q) for p in self.programs for v in p['variants'] for q in v['questions']]
            index = next(i for i, (_, q) in enumerate(items) if q['label'] == 'Proven')
            question = items[index][1]
            envelope['predictions'][index]['prediction'] = {'query': question['query'],
                'result': 'Proven', 'certificate': question['certificate']}
            output.write_text(json.dumps(envelope))
            good = audit_predictions(**args, predictions_path=output, predictions_sha256=digest(envelope))
            self.assertGreater(good['counts']['replay_passed'], 0)
            envelope['predictions'][index]['prediction']['certificate'] = {'kind': 'path', 'nodes': ['invalid', 'invalid']}
            output.write_text(json.dumps(envelope))
            bad = audit_predictions(**args, predictions_path=output, predictions_sha256=digest(envelope))
            self.assertEqual(bad['counts']['replay_rejected'], good['counts']['replay_rejected'] + 1)
            self.assertEqual(bad['counts']['replay_passed'], good['counts']['replay_passed'] - 1)

    def test_independent_subprocess_negative_and_wrong_query_certificates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            writer = ArtifactWriter(root)
            entry = writer.save_checkpoint('model', CalibrationModel('B'), arm='B', seed=1, selection={})
            output = root / 'out.json'
            args = dict(programs=self.programs, split='validation', manifest=self.manifest,
                        manifest_sha256=self.pin, checkpoint_sha256=entry['sha256'], arm='B')
            infer_checkpoint(**args, checkpoint=root / entry['path'], output=output)
            envelope = json.loads(output.read_text())
            items = [(v, q) for p in self.programs for v in p['variants'] for q in v['questions']]
            # Remove untrained predictions to make all replay outcomes deterministic.
            for row in envelope['predictions']:
                row['prediction'] = {'query': None, 'result': 'Unknown', 'certificate': None}
            indices = []
            for kind in ('absent_edge', 'closed_set'):
                index = next(i for i, (_, q) in enumerate(items)
                             if q['label'] == 'Refuted' and q['certificate']['kind'] == kind)
                q = items[index][1]
                envelope['predictions'][index]['prediction'] = {'query': q['query'], 'result': 'Refuted', 'certificate': q['certificate']}
                indices.append(index)
            # Use a valid negative proof for a different query on the same graph.
            index = indices[0]
            program, q = items[index]
            other = next(i for i, (v, candidate) in enumerate(items)
                         if v == program and candidate['view'] == q['view'] and candidate['query'] != q['query'] and i not in indices)
            envelope['predictions'][other]['prediction'] = envelope['predictions'][index]['prediction']
            output.write_text(json.dumps(envelope))
            request = {**args, 'predictions_path': str(output), 'predictions_sha256': digest(envelope), 'edge_threshold': 0.5}
            request_path, report_path = root / 'request.json', root / 'audit.json'
            request_path.write_text(json.dumps(request))
            subprocess.run([sys.executable, '-m', 'neural_lab.nl001_prediction_audit', str(request_path), str(report_path)], check=True)
            audit = json.loads(report_path.read_text())
            report_path.unlink()
            bad_request = {**request, 'predictions_sha256': '0' * 64}
            request_path.write_text(json.dumps(bad_request))
            failed = subprocess.run([sys.executable, '-m', 'neural_lab.nl001_prediction_audit', str(request_path), str(report_path)], capture_output=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertFalse(report_path.exists())
            production = replay_saved(**request)
            self.assertEqual(audit['counts']['replay_passed'], 3)
            self.assertEqual(audit['counts']['verified_claims'], 2)
            self.assertTrue(audit['details'][indices[0]]['claim_verified'])
            self.assertTrue(audit['details'][other]['replayed'])
            self.assertFalse(audit['details'][other]['claim_verified'])
            self.assertFalse(audit['details'][other]['query_aligned'])
            self.assertFalse(audit['details'][other]['correct'])
            for actual, expected in zip(audit['details'], production['details']):
                self.assertEqual({k: v for k, v in actual.items() if k not in ('identity', 'claim_verified')}, expected)
            envelope['predictions'][indices[1]]['prediction']['certificate'] = {'kind': 'closed_set', 'nodes': []}
            output.write_text(json.dumps(envelope))
            request['predictions_sha256'] = digest(envelope)
            corrupted = audit_predictions(**request)
            self.assertEqual(corrupted['details'][indices[1]]['status'], 'Unknown')
            self.assertFalse(corrupted['details'][indices[1]]['replayed'])

    def test_fresh_process_replay_and_bad_pin_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for arm in ('A', 'B'):
                writer = ArtifactWriter(root / arm)
                entry = writer.save_checkpoint('model', CalibrationModel(arm), arm=arm, seed=1, selection={})
                predictions = root / (arm + '-predictions.json')
                args = dict(programs=self.programs, split='validation', manifest=self.manifest,
                            manifest_sha256=self.pin, checkpoint_sha256=entry['sha256'], arm=arm)
                pin = infer_checkpoint(**args, checkpoint=root / arm / entry['path'], output=predictions)
                request = {**args, 'predictions_path': str(predictions), 'predictions_sha256': pin,
                           'edge_threshold': 0.5}
                request_path = root / (arm + '-request.json')
                report_path = root / (arm + '-report.json')
                request_path.write_text(json.dumps(request))
                command = [sys.executable, '-m', 'neural_lab.nl001_selection_runner', '--replay',
                           str(request_path), '--replay-output', str(report_path)]
                subprocess.run(command, check=True, capture_output=True, text=True)
                self.assertEqual(json.loads(report_path.read_text()), replay_saved(**request))
                report_path.unlink()
                request['predictions_sha256'] = '0' * 64
                request_path.write_text(json.dumps(request))
                failed = subprocess.run(command, capture_output=True, text=True)
                self.assertNotEqual(failed.returncode, 0)
                self.assertFalse(report_path.exists())

    def test_selected_training_artifacts_reload_and_replay_without_training(self):
        corpus = make_sampled_corpus({'train': [11], 'validation': [12], 'test': [13]}, per_stratum=1)
        train = corpus['splits']['train']
        with tempfile.TemporaryDirectory() as tmp:
            for arm, hidden in (('A', 192), ('B', 128)):
                root = Path(tmp) / arm
                writer = ArtifactWriter(root)
                source, tensors = _save_training_artifacts(writer, train, self.manifest, self.pin, _flatten(train))
                torch.manual_seed(17)
                model = CalibrationModel(arm, hidden=hidden, embedding=32)
                expected = {name: tensor.clone() for name, tensor in model.state_dict().items()}
                best = BestCheckpoint()
                best.observe(model, 1.25, 100)
                with torch.no_grad():
                    next(model.parameters()).add_(1)
                best.observe(model, 2.0, 200)
                entry = _save_best_checkpoint(best, writer, arm, 17,
                    {'hidden': hidden, 'embedding': 32}, source, tensors)
                checkpoint = root / entry['path']
                loaded = load_json_state(checkpoint)
                self.assertEqual(set(loaded), set(expected))
                for name in expected:
                    self.assertTrue(torch.equal(loaded[name], expected[name]), name)
                restored = CalibrationModel(arm, hidden=hidden, embedding=32)
                restored.load_state_dict(loaded, strict=True)
                self.assertEqual(entry['selection']['step'], 100)
                self.assertEqual(entry['datasets'], [source['sha256'], tensors['sha256']])
                recorded = json.loads((root / 'artifacts.json').read_text())
                self.assertEqual(recorded['checkpoints'], [entry])
                self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(), entry['sha256'])
                args = dict(programs=self.programs, split='validation', manifest=self.manifest,
                            manifest_sha256=self.pin, checkpoint_sha256=entry['sha256'], arm=arm)
                output = root / 'predictions.json'
                pin = infer_checkpoint(**args, checkpoint=checkpoint, output=output)
                report = replay_saved(**args, predictions_path=output, predictions_sha256=pin)
                self.assertGreater(report.get('claims', report.get('counts', {}).get('claims', 0)), 0)


if __name__ == '__main__':
    unittest.main()
