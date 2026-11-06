"""Explicit NL-001 training examples and verifier boundary for calibration.

This module is a building block, not an approved experiment or queue job.
Supervision is kept outside the inference input. Positive witness targets are
one deterministic checker path, not the set of all valid proofs.
"""
from .nl001_data import model_input
from .nl001_model_input import encode_input, MAX_NODES
from .synthetic_graph import Graph, check, replay

PROPERTIES = ('calls', 'reachable', 'unsupported')
STATUSES = ('Proven', 'Refuted', 'Unknown')


def example(program, question):
    """Return separately encoded observed inputs and training-only targets."""
    raw = model_input(program, question)
    encoded = encode_input(raw)
    q, label = question['query'], question['label']
    if q['property'] not in PROPERTIES or label not in STATUSES:
        raise ValueError('unsupported training target')
    symbols = raw['symbols']
    subject, obj = symbols.index(q['subject']), symbols.index(q['object'])
    witness = [[0] * MAX_NODES for _ in range(MAX_NODES)]
    graph = Graph(tuple(symbols), frozenset(map(tuple, raw['observed_edges'])), raw['complete'], program['source'])
    if label == 'Proven':
        cert = question['certificate']
        if not replay(graph, q, label, cert):
            raise ValueError('invalid positive training certificate')
        for a, b in zip(cert['nodes'], cert['nodes'][1:]):
            witness[symbols.index(a)][symbols.index(b)] = 1
    return encoded, {'status': STATUSES.index(label), 'property': PROPERTIES.index(q['property']),
                     'subject': subject, 'object': obj, 'witness': witness,
                     'has_witness': label == 'Proven'}


def verify_selected(raw, query, selected_edges):
    """Check a translated query; positives use selected facts, negatives all facts.

    Results are query-level only: semantic alignment must be measured later.
    Selected edges are rejected unless they are observed. A failed selected proof
    is Unknown unless a complete full graph independently refutes the query.
    """
    encode_input(raw)  # validate the inference-only boundary and bounds
    facts = frozenset(map(tuple, raw['observed_edges']))
    selected = frozenset(map(tuple, selected_edges))
    if not selected <= facts:
        raise ValueError('selected edges must be observed')
    symbols = tuple(raw['symbols'])
    chosen = Graph(symbols, selected, False, 'selected-observed-facts')
    full = Graph(symbols, facts, raw['complete'], 'full-observed-facts')
    pos = check(chosen, query, claim=raw['claim'])
    if pos['query_result'] == 'Proven':
        return {'query': query, 'result': 'Proven', 'certificate': pos['certificate']}
    negative = check(full, query, claim=raw['claim'])
    if negative['query_result'] == 'Refuted':
        return {'query': query, 'result': 'Refuted', 'certificate': negative['certificate']}
    return {'query': query, 'result': 'Unknown', 'certificate': None}
