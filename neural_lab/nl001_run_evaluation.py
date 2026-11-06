"""No-training evaluation of frozen predictions and deterministic controls.

Only independently pinned split payloads are accepted. Model inference and
checkpoint selection remain the caller's responsibility.
"""
import json
from pathlib import Path

from .artifacts import ArtifactWriter
from .nl001_data import digest, model_input
from .nl001_evaluation import evaluate
from .nl001_manifest import authenticate_split
from .nl001_parser import parse_claim
from .synthetic_graph import Graph, check


def evaluate_split(*, programs, split, manifest, manifest_sha256, output, mode, predictions=None):
    """Evaluate parser/oracle controls, checked predictions, or direct answers.

    predictions are ordered by program, presentation, question. Direct answers
    are never treated as certificates. Existing output must be empty.
    """
    authenticate_split(programs, split, manifest, manifest_sha256)
    if mode not in ('parser', 'oracle', 'checked', 'direct'):
        raise ValueError('invalid evaluation mode')
    items = [(v, q) for p in programs for v in p['variants'] for q in v['questions']]
    if mode in ('checked', 'direct') and (not isinstance(predictions, list) or len(predictions) != len(items)):
        raise ValueError('predictions must match all claims')
    records, saved = [], []
    for i, (program, question) in enumerate(items):
        raw = model_input(program, question)
        graph = Graph(tuple(raw['symbols']), frozenset(map(tuple, raw['observed_edges'])), raw['complete'], program['source'])
        if mode in ('parser', 'oracle'):
            query = question['query'] if mode == 'oracle' else parse_claim(raw['claim'], raw['symbols'])
            if query is None:
                pred = {'query': None, 'result': 'Unknown', 'certificate': None}
            else:
                result = check(graph, query, claim=raw['claim'])
                pred = {'query': query, 'result': result['query_result'], 'certificate': result['certificate']}
        else:
            pred = predictions[i]
        saved.append({'source': program['source'], 'presentation': program['presentation'],
                      'pair_id': question['pair_id'], 'claim': raw['claim'], 'view': question['view'],
                      'expected_query': question['query'], 'label': question['label'], 'prediction': pred})
        records.append({'graph': graph, 'expected_query': question['query'], 'label': question['label'], 'prediction': pred})
    if mode == 'direct':
        correct = sum(isinstance(p, str) and p in ('Proven', 'Refuted') and p == q['label']
                      for p, (_, q) in zip(predictions, items))
        answered = sum(isinstance(p, str) and p in ('Proven', 'Refuted') for p in predictions)
        report = {'claims': len(items), 'correct_answers': correct, 'answered': answered,
                  'answer_accuracy': correct / len(items) if items else None,
                  'conditional_answer_accuracy': correct / answered if answered else None,
                  'certifications': 0, 'scope': 'direct predictions; no proofs'}
    else:
        report = evaluate(records)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError('output directory must be empty')
    writer = ArtifactWriter(out)
    for filename, value in [('split.json', programs), ('manifest.json', manifest),
                            ('predictions.json', saved), ('evaluation.json', report)]:
        path = out / filename
        path.write_text(json.dumps(value, sort_keys=True, allow_nan=False, indent=2))
        writer.register_file('datasets', path, filename, format='json', canonical_sha256=digest(value))
    return report
