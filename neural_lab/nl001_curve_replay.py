"""Fresh-process checks for curve checkpoints; not a custody or launch boundary."""
import json
import time

from .nl001_data import digest
from .nl001_selection_runner import _child, _read


def verify_checkpoint(*, programs, manifest, manifest_sha256, checkpoint, checkpoint_sha256,
                      predictions, prediction_sha256, arm, edge_threshold, evaluation, directory, step):
    def write(path, obj):
        path.write_text(json.dumps(obj, sort_keys=True, allow_nan=False, indent=2))
    stem = str(step)
    fresh = directory / ('fresh-predictions-' + stem + '.json')
    request = directory / ('infer-request-' + stem + '.json')
    write(request, {'programs': programs, 'split': 'validation', 'manifest': manifest,
        'manifest_sha256': manifest_sha256, 'checkpoint': str(checkpoint.resolve()),
        'checkpoint_sha256': checkpoint_sha256, 'arm': arm, 'output': str(fresh.resolve()),
        'edge_threshold': edge_threshold})
    start = time.perf_counter()
    _child('neural_lab.nl001_selection_runner', ['--infer', str(request.resolve())],
           directory / ('infer-imports-' + stem + '.json'))
    inference_seconds = time.perf_counter() - start
    if digest(_read(fresh)) != prediction_sha256 or _read(fresh) != _read(predictions):
        raise ValueError('fresh-process checkpoint inference disagrees')
    request = directory / ('audit-request-' + stem + '.json')
    report = directory / ('audit-' + stem + '.json')
    write(request, {'programs': programs, 'split': 'validation', 'manifest': manifest,
        'manifest_sha256': manifest_sha256, 'checkpoint_sha256': checkpoint_sha256,
        'predictions_path': str(predictions.resolve()), 'predictions_sha256': prediction_sha256,
        'arm': arm, 'edge_threshold': edge_threshold})
    start = time.perf_counter()
    _child('neural_lab.nl001_prediction_audit', [str(request.resolve()), str(report.resolve())],
           directory / ('audit-imports-' + stem + '.json'))
    audit_seconds = time.perf_counter() - start
    audit = _read(report)
    if arm == 'B':
        rows, details = _read(predictions)['predictions'], audit['details']
        # The audit enriches each row with identity and derived fields the replay
        # evaluation does not emit; compare only the schema both sides share so a
        # genuine status/replay/alignment/correctness disagreement still fails.
        if (len(details) != len(rows) or len(details) != len(evaluation['details']) or
            any(detail.get('identity') != row['identity'] or
                {k: detail[k] for k in detail.keys() & expected.keys() if k != 'identity'} !=
                {k: expected[k] for k in detail.keys() & expected.keys() if k != 'identity'}
                for detail, row, expected in zip(details, rows, evaluation['details']))):
            raise ValueError('independent per-claim audit disagrees with replay')
    return {'fresh_process_inference_seconds': inference_seconds,
            'independent_audit_seconds': audit_seconds, 'independent_audit': audit}
