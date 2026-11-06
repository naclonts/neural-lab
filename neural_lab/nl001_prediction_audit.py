"""Independent certificate audit of pinned NL-001 prediction envelopes.

Uses the separate Warshall oracle, not the production graph checker. Input
identity and semantic labels still depend on the caller's authenticated split.
Claim/query alignment is checked independently for the supported literal forms.
"""
import argparse
import json
from pathlib import Path

from .nl001_data import digest
from .nl001_alignment_audit import parse
from .nl001_manifest import authenticate_split
from .nl001_independent_oracle import certificate_valid


def audit_predictions(*, programs, split, manifest, manifest_sha256, checkpoint_sha256,
                      predictions_path, predictions_sha256, arm, edge_threshold=0.5):
    authenticate_split(programs, split, manifest, manifest_sha256)
    envelope = json.loads(Path(predictions_path).read_text())
    if digest(envelope) != predictions_sha256:
        raise ValueError('prediction digest mismatch')
    expected = {'version': 'nl001-inference-v1', 'split': split, 'split_sha256': digest(programs),
                'manifest_sha256': manifest_sha256, 'checkpoint_sha256': checkpoint_sha256,
                'arm': arm, 'edge_threshold': edge_threshold}
    if arm not in ('A', 'B') or any(envelope.get(k) != v for k, v in expected.items()):
        raise ValueError('prediction binding mismatch')
    items = [(v, q) for p in programs for v in p['variants'] for q in v['questions']]
    rows = envelope.get('predictions')
    if not isinstance(rows, list) or len(rows) != len(items):
        raise ValueError('prediction count mismatch')
    counts = {'claims': len(items), 'replay_attempts': 0, 'replay_passed': 0, 'replay_rejected': 0,
              'aligned_label_conflicts': 0, 'verified_claims': 0}
    details = []
    for (program, question), row in zip(items, rows):
        if not isinstance(row, dict) or set(row) != {'identity', 'prediction'} or row['identity'] != digest({'program': program, 'question': question}):
            raise ValueError('prediction identity mismatch')
        form, prop, subject, obj = parse(question['claim'], program['symbols'])
        expected_form = 'test' if program['presentation']['heldout_template'] else 'train'
        if (form != expected_form or question['template_id'] != f'{form}:{prop}'
                or question['query'] != {'property': prop, 'subject': subject, 'object': obj}):
            raise ValueError('claim/query alignment mismatch')
        if question.get('view') not in ('complete', 'incomplete'):
            raise ValueError('invalid question view')
        if arm == 'A':
            continue
        pred = row['prediction'] if isinstance(row['prediction'], dict) else {}
        status, query = pred.get('result'), pred.get('query')
        aligned = isinstance(query, dict) and query == question['query']
        if status not in ('Proven', 'Refuted'):
            details.append({'identity': row['identity'], 'status': 'Unknown', 'query_aligned': aligned,
                            'replayed': False, 'claim_verified': False, 'correct': False})
            continue
        counts['replay_attempts'] += 1
        complete = question['view'] == 'complete'
        edges = program['full_edges'] if complete else program['observed_incomplete_edges']
        try:
            valid = certificate_valid(program['symbols'], edges, complete, query, status, pred.get('certificate'))
        except (TypeError, ValueError, KeyError):
            valid = False
        counts['replay_passed' if valid else 'replay_rejected'] += 1
        if valid and aligned:
            counts['verified_claims'] += 1
        if valid and aligned and status != question['label']:
            counts['aligned_label_conflicts'] += 1
        details.append({'identity': row['identity'], 'status': status if valid else 'Unknown',
                        'query_aligned': aligned, 'replayed': bool(valid), 'claim_verified': bool(valid and aligned),
                        'correct': bool(valid and aligned and status == question['label'])})
    return {'version': 'nl001-independent-prediction-audit-v3', 'arm': arm,
            'predictions_sha256': predictions_sha256, 'counts': counts, 'details': details,
            'scope': 'independent graph certificate semantics and controlled literal claim/query alignment; split identity and labels are trusted',
            'metric_definitions': {'replay_passed': 'valid certificate for the predicted query, which may differ from the claim',
                                   'replayed': 'valid certificate for the predicted query, which may differ from the claim',
                                   'verified_claims': 'valid certificates whose predicted queries align with their claims',
                                   'claim_verified': 'valid certificate whose predicted query aligns with its claim'}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('request')
    parser.add_argument('output')
    args = parser.parse_args()
    report = audit_predictions(**json.loads(Path(args.request).read_text()))
    with Path(args.output).open('x') as f:
        json.dump(report, f, sort_keys=True, allow_nan=False, indent=2)


if __name__ == '__main__':
    main()
