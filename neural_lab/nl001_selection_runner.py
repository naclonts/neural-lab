"""Unreviewed validation-only integration; not an authorized queue entry point.

Inputs are separate train and validation files. Test is deliberately not accepted.
Digests must be retained independently; this interface cannot protect trust anchors
or prove that test data was not accessed elsewhere.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .artifacts import ArtifactWriter
from .nl001_data import digest
from .nl001_inference import infer_checkpoint, replay_saved
from .nl001_manifest import authenticate_split
from .nl001_run_evaluation import evaluate_split
from .nl001_training import fit_fixed_updates


def _read(path):
    return json.loads(Path(path).read_text())


def _write(path, obj):
    Path(path).write_text(json.dumps(obj, sort_keys=True, allow_nan=False, indent=2))


def _check_audit_details(details, rows, expected_details):
    """Compare shared replay fields and independently check claim verification."""
    fields = {'status', 'query_aligned', 'replayed', 'correct'}
    if len(details) != len(rows) or len(details) != len(expected_details):
        raise ValueError('independent per-claim audit disagrees with replay')
    for detail, row, expected in zip(details, rows, expected_details):
        if (set(detail) != fields | {'identity', 'claim_verified'} or set(expected) != fields
                or detail['identity'] != row['identity']
                or {key: detail[key] for key in fields} != expected
                or type(detail['claim_verified']) is not bool
                or detail['claim_verified'] != (expected['replayed'] and expected['query_aligned'])):
            raise ValueError('independent per-claim audit disagrees with replay')



def _child(module, args, report):
    """Optionally require a child import report under calibration custody."""
    manifest = os.environ.get('NL001_CHILD_ATTESTATION_MANIFEST')
    if manifest:
        command = [sys.executable, '-m', 'neural_lab.nl001_child_attestation',
                   manifest, str(report), module, *args]
    else:
        command = [sys.executable, '-m', module, *args]
    subprocess.run(command, check=True, timeout=120)
    if manifest:
        attestation = _read(report)
        if attestation.get('target') != module or not attestation.get('imported_modules'):
            raise ValueError('missing or invalid child import attestation')


def _audit_gates(manifest, manifest_sha256, audit, audit_sha256,
                 identity, identity_sha256, expected_identities_sha256, semantic, semantic_sha256):
    """Validate retained reports; custody and audit execution remain external."""
    if digest(audit) != audit_sha256 or audit.get('version') != 'nl001-corpus-audit-v2' or audit.get('regeneration') != 'exact match using current generator and checker':
        raise ValueError('corpus audit does not match independently retained digest')
    if (audit.get('manifest') != manifest or audit.get('manifest_sha256') != manifest_sha256
            or audit.get('corpus_sha256') != manifest.get('corpus_sha256')):
        raise ValueError('corpus audit and split manifest disagree')
    try:
        splits = manifest['splits']
        if set(splits) != {'train', 'validation', 'test'}:
            raise ValueError('invalid split coverage')
        counts = [len(splits[name]['seeds']) for name in ('train', 'validation', 'test')]
        if any(n <= 0 for n in counts):
            raise ValueError('empty split')
        # Sampled v1 has one/one/four presentations and two views each.
        a, b, c = 2 * counts[0], 2 * counts[1], 8 * counts[2]
        required = a*b + a*c + b*c
        comparison = audit['observed_view_audit']
        if (type(comparison['comparisons']) is not int or comparison['comparisons'] != required
                or type(comparison['required_comparisons']) is not int or comparison['required_comparisons'] != required
                or type(comparison['max_comparisons']) is not int or comparison['max_comparisons'] < required
                or type(comparison['max_states_per_comparison']) is not int or comparison['max_states_per_comparison'] <= 0):
            raise ValueError('incomplete corpus audit comparisons')
        if (digest(identity) != identity_sha256 or identity['version'] != 'nl001-alignment-identity-audit-v1'
                or identity['corpus_sha256'] != manifest['corpus_sha256']
                or identity['expected_identities_sha256'] != expected_identities_sha256
                or not isinstance(identity['expected_identities_sha256'], str)
                or len(identity['expected_identities_sha256']) != 64
                or any(c not in '0123456789abcdef' for c in identity['expected_identities_sha256'])):
            raise ValueError('invalid identity audit')
        questions = identity['question_counts']
        if set(questions) != set(splits) or any(type(v) is not int or v <= 0 for v in questions.values()):
            raise ValueError('invalid identity question counts')
        if (digest(semantic) != semantic_sha256 or semantic['version'] != 'nl001-semantic-count-attestation-v1'
                or semantic['corpus_sha256'] != manifest['corpus_sha256']
                or set(semantic) != {'version', 'corpus_sha256', 'question_counts', 'claims', 'semantic_audit_sha256'}
                or not isinstance(semantic['semantic_audit_sha256'], str)
                or len(semantic['semantic_audit_sha256']) != 64
                or any(c not in '0123456789abcdef' for c in semantic['semantic_audit_sha256'])):
            raise ValueError('invalid semantic audit')
        if (semantic['question_counts'] != questions or type(semantic['claims']) is not int
                or semantic['claims'] != sum(questions.values())):
            raise ValueError('incomplete semantic counts')
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError('invalid or incomplete audit report') from exc


def run(*, train_path, validation_path, manifest_path, manifest_sha256, output,
        seed, batch_size, updates, learning_rate, device, audit_path, audit_sha256,
        identity_audit_path, identity_audit_sha256, expected_identities_sha256,
        semantic_audit_path, semantic_audit_sha256, edge_threshold=0.5):
    """Run training-loss-selected checkpoints and validation controls, never test selection.

    This consumes no test file and has no test-stage mode. It does not itself
    establish corpus provenance, protect source bytes, or authorize a launch.
    """
    train, validation, manifest, audit, identity, semantic = map(
        _read, (train_path, validation_path, manifest_path, audit_path, identity_audit_path, semantic_audit_path))
    _audit_gates(manifest, manifest_sha256, audit, audit_sha256, identity, identity_audit_sha256, expected_identities_sha256,
                 semantic, semantic_audit_sha256)
    authenticate_split(train, 'train', manifest, manifest_sha256)
    authenticate_split(validation, 'validation', manifest, manifest_sha256)
    for name, programs in (('train', train), ('validation', validation)):
        if sum(len(v['questions']) for p in programs for v in p['variants']) != identity['question_counts'][name]:
            raise ValueError('identity audit question counts disagree with split')
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError('output directory must be empty')
    protocol = {'version': 'nl001-selection-integration-v1', 'manifest_sha256': manifest_sha256,
                'seed': seed, 'batch_size': batch_size, 'updates': updates,
                'learning_rate': learning_rate, 'device': device, 'edge_threshold': edge_threshold,
                'audit_sha256': audit_sha256, 'identity_audit_sha256': identity_audit_sha256,
                'expected_identities_sha256': expected_identities_sha256,
                'semantic_audit_sha256': semantic_audit_sha256,
                'checkpoint_selection': 'minimum full-training-set CE after updates 100, 200, ..., and final; B sums property/subject/object CE, excludes witness BCE; earliest tie', 'evaluation_split': 'validation'}
    _write(out / 'protocol.json', protocol)
    writer = ArtifactWriter(out)
    writer.register_file('datasets', out / 'protocol.json', 'protocol', format='json')
    reports = {}
    for mode in ('oracle', 'parser'):
        reports[mode] = evaluate_split(programs=validation, split='validation', manifest=manifest,
            manifest_sha256=manifest_sha256, output=out / mode, mode=mode)
        if reports[mode]['counts']['aligned_label_conflicts']:
            raise ValueError('control found aligned label conflict')
    for arm in ('A', 'B'):
        training = fit_fixed_updates(programs=train, manifest=manifest, manifest_sha256=manifest_sha256,
            output=out / arm, arm=arm, seed=seed, batch_size=batch_size, updates=updates,
            learning_rate=learning_rate, device=device)
        checkpoint = out / arm / 'checkpoints' / ('best-' + arm + '.pt')
        predictions = out / (arm + '-predictions.json')
        start = time.perf_counter()
        prediction_pin = infer_checkpoint(programs=validation, split='validation', manifest=manifest,
            manifest_sha256=manifest_sha256, checkpoint=checkpoint,
            checkpoint_sha256=training['checkpoint_sha256'], arm=arm, output=predictions,
            edge_threshold=edge_threshold)
        inference_seconds = time.perf_counter() - start
        inference_request = out / (arm + '-inference-request.json')
        fresh_predictions = out / (arm + '-fresh-predictions.json')
        _write(inference_request, {'programs': validation, 'split': 'validation', 'manifest': manifest,
            'manifest_sha256': manifest_sha256, 'checkpoint': str(checkpoint.resolve()),
            'checkpoint_sha256': training['checkpoint_sha256'], 'arm': arm,
            'output': str(fresh_predictions.resolve()), 'edge_threshold': edge_threshold})
        start = time.perf_counter()
        _child('neural_lab.nl001_selection_runner', ['--infer', str(inference_request)],
               out / (arm + '-infer-imports.json'))
        fresh_inference_seconds = time.perf_counter() - start
        if digest(_read(fresh_predictions)) != prediction_pin or _read(fresh_predictions) != _read(predictions):
            raise ValueError('fresh-process checkpoint inference disagrees with predictions')
        replay_input = {'programs': validation, 'split': 'validation', 'manifest': manifest,
            'manifest_sha256': manifest_sha256, 'checkpoint_sha256': training['checkpoint_sha256'],
            'predictions_path': str(predictions.resolve()), 'predictions_sha256': prediction_pin,
            'arm': arm, 'edge_threshold': edge_threshold}
        replay_request = out / (arm + '-replay-request.json')
        replay_report = out / (arm + '-replay.json')
        _write(replay_request, replay_input)
        start = time.perf_counter()
        _child('neural_lab.nl001_selection_runner', ['--replay', str(replay_request), '--replay-output', str(replay_report)],
               out / (arm + '-replay-imports.json'))
        replay_seconds = time.perf_counter() - start
        report = _read(replay_report)
        independent_path = out / (arm + '-independent-audit.json')
        start = time.perf_counter()
        _child('neural_lab.nl001_prediction_audit', [str(replay_request), str(independent_path)],
               out / (arm + '-audit-imports.json'))
        independent_audit_seconds = time.perf_counter() - start
        independent = _read(independent_path)
        if arm == 'B' and any(independent['counts'][key] != report['counts'][key] for key in
                              ('claims', 'replay_attempts', 'replay_passed', 'replay_rejected', 'aligned_label_conflicts')):
            raise ValueError('independent certificate audit disagrees with replay')
        if arm == 'B':
            saved_rows = _read(predictions)['predictions']
            details = independent['details']
            expected_details = report['details']
            _check_audit_details(details, saved_rows, expected_details)
        if arm == 'B' and report['counts']['aligned_label_conflicts']:
            raise ValueError('replay found aligned label conflict')
        reports[arm] = {'training': training, 'inference_seconds': inference_seconds,
                        'fresh_process_inference_seconds': fresh_inference_seconds,
                        'fresh_process_replay_seconds': replay_seconds,
                        'independent_audit_seconds': independent_audit_seconds,
                        'prediction_sha256': prediction_pin, 'evaluation': report, 'independent_audit': independent}
        for path in (predictions, inference_request, fresh_predictions, replay_request, replay_report, independent_path):
            writer.register_file('datasets', path, path.stem, format='json')
    summary = {'protocol': protocol, 'reports': reports,
               'limitations': ['Unreviewed integration; not a queue executable.',
                   'Source, environment and pins require an independently protected freeze.',
                   'Audit report pin is caller supplied; provenance, semantics and physical test isolation remain unverified.',
                   'Model capacities and measured compute differ; premise recall is unresolved.',
                   'Fresh-process replay uses the same checker; independent audit checks certificate semantics, not claim interpretation; claim-level Wilson bounds ignore clustering.',
                   'CUDA determinism and target-device feasibility are untested.']}
    _write(out / 'selection-summary.json', summary)
    writer.register_file('datasets', out / 'selection-summary.json', 'selection-summary', format='json')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay')
    parser.add_argument('--replay-output')
    parser.add_argument('--infer')
    parser.add_argument('--config', help='JSON object of run() arguments')
    args = parser.parse_args()
    if args.config and not args.replay and not args.replay_output and not args.infer:
        run(**_read(args.config))
    elif args.infer and not args.config and not args.replay and not args.replay_output:
        infer_checkpoint(**_read(args.infer))
    elif args.replay and args.replay_output and not args.config and not args.infer:
        report = replay_saved(**_read(args.replay))
        with Path(args.replay_output).open('x') as f:
            json.dump(report, f, sort_keys=True, allow_nan=False, indent=2)
    else:
        parser.error('supply --config or both --replay and --replay-output')


if __name__ == '__main__':
    main()
