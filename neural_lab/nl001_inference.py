"""Bounded checkpoint inference and persisted replay, not a launch protocol.

Pins must be independently frozen by the caller. Replay is a separate invocation;
it verifies certificates, but does not establish claim semantics or provenance.
"""
import hashlib
import json
from pathlib import Path

import torch

from .nl001_data import digest, model_input
from .nl001_evaluation import evaluate
from .nl001_manifest import authenticate_split
from .nl001_model_input import encode_input
from .nl001_models import CalibrationModel
from .nl001_training_data import PROPERTIES, STATUSES, verify_selected
from .synthetic_graph import Graph


def _items(programs):
    return [(v, q) for p in programs for v in p['variants'] for q in v['questions']]


def _identity(program, question):
    # Bind the entire trusted question and presentation, including observation.
    return digest({'program': program, 'question': question})


def _checkpoint_model(path, checkpoint_sha256, arm):
    """Authenticate and decode the same checkpoint bytes, then validate its state.

    Accepts legacy JSON state dicts and current .pt checkpoints. The .pt path
    uses torch.load(weights_only=True), so only tensors and plain containers
    are reconstructed and no model code executes.
    """
    payload = Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != checkpoint_sha256:
        raise ValueError('checkpoint digest mismatch')

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate checkpoint key')
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError('non-finite checkpoint value: ' + value)

    import io as _io

    if Path(path).suffix == '.pt':
        meta = torch.load(_io.BytesIO(payload), map_location='cpu', weights_only=True)
        if not isinstance(meta, dict) or meta.get('format') != 'neural-lab-state-dict-pt-v1' or meta.get('arm') != arm:
            raise ValueError('checkpoint format or arm mismatch')
        tensors = meta.get('tensors')
        if not isinstance(tensors, dict) or any(not isinstance(t, torch.Tensor) for t in tensors.values()):
            raise ValueError('checkpoint tensor mapping is invalid')
    else:
        meta = json.loads(payload, object_pairs_hook=unique, parse_constant=invalid_constant)
        if not isinstance(meta, dict) or meta.get('format') != 'neural-lab-state-dict-v1' or meta.get('arm') != arm:
            raise ValueError('checkpoint format or arm mismatch')
        tensors = meta.get('tensors')
        if not isinstance(tensors, dict):
            raise ValueError('checkpoint tensor mapping is invalid')
    config = meta.get('config', {})
    if not isinstance(config, dict):
        raise ValueError('invalid checkpoint configuration')
    hidden, embedding = config.get('hidden', 128), config.get('embedding', 32)
    if type(hidden) is not int or type(embedding) is not int or hidden <= 0 or embedding <= 0:
        raise ValueError('invalid checkpoint architecture')
    model = CalibrationModel(arm, hidden=hidden, embedding=embedding)
    expected = model.state_dict()
    if set(tensors) != set(expected):
        raise ValueError('checkpoint tensor names mismatch')
    state = {}
    for name, reference in expected.items():
        tensor = tensors[name]
        if Path(path).suffix == '.pt':
            if tensor.dtype != reference.dtype or tensor.shape != reference.shape:
                raise ValueError('checkpoint tensor metadata mismatch: ' + name)
            value = tensor
        else:
            if (not isinstance(tensor, dict) or set(tensor) != {'dtype', 'shape', 'data'}
                    or tensor['dtype'] != str(reference.dtype).removeprefix('torch.')
                    or tensor['shape'] != list(reference.shape)):
                raise ValueError('checkpoint tensor metadata mismatch: ' + name)
            try:
                value = torch.tensor(tensor['data'], dtype=reference.dtype)
            except (TypeError, ValueError, RuntimeError, OverflowError) as exc:
                raise ValueError('invalid checkpoint tensor: ' + name) from exc
        if value.shape != reference.shape or (value.is_floating_point() and not torch.isfinite(value).all()):
            raise ValueError('invalid checkpoint tensor: ' + name)
        state[name] = value
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def infer_checkpoint(*, programs, split, manifest, manifest_sha256, checkpoint, checkpoint_sha256, arm, output, edge_threshold=0.5):
    """Run a pinned JSON state dictionary on CPU, writing bound predictions.

    Threshold is fixed by caller before evaluation. No checkpoint selection occurs.
    Returns the digest that must be independently retained for replay.
    """
    authenticate_split(programs, split, manifest, manifest_sha256)
    if arm not in ('A', 'B') or isinstance(edge_threshold, bool) or not isinstance(edge_threshold, (int, float)) or not 0 <= edge_threshold <= 1:
        raise ValueError('invalid inference configuration')
    model = _checkpoint_model(checkpoint, checkpoint_sha256, arm)
    predictions = []
    with torch.inference_mode():
        for program, question in _items(programs):
            raw = model_input(program, question)
            encoded = encode_input(raw)
            inputs = {k: torch.tensor([v]) for k, v in encoded.items()}
            result = model(inputs)
            if arm == 'A':
                prediction = STATUSES[int(result['status'].argmax(dim=1).item())]
            else:
                symbols = raw['symbols']
                query = {'property': PROPERTIES[int(result['property'].argmax(dim=1).item())],
                         'subject': symbols[int(result['subject'].argmax(dim=1).item())],
                         'object': symbols[int(result['object'].argmax(dim=1).item())]}
                scores = torch.sigmoid(result['edges'][0])
                selected = [e for e in raw['observed_edges'] if scores[symbols.index(e[0]), symbols.index(e[1])].item() >= edge_threshold]
                prediction = verify_selected(raw, query, selected)
            predictions.append({'identity': _identity(program, question), 'prediction': prediction})
    envelope = {'version': 'nl001-inference-v1', 'split': split, 'split_sha256': digest(programs),
                'manifest_sha256': manifest_sha256, 'checkpoint_sha256': checkpoint_sha256,
                'arm': arm, 'edge_threshold': edge_threshold, 'predictions': predictions}
    out = Path(output)
    with out.open('x') as f:
        json.dump(envelope, f, sort_keys=True, allow_nan=False, indent=2)
    return digest(envelope)


def replay_saved(*, programs, split, manifest, manifest_sha256, checkpoint_sha256, predictions_path, predictions_sha256, arm, edge_threshold=0.5):
    """Reload pinned predictions and independently replay against pinned graphs."""
    authenticate_split(programs, split, manifest, manifest_sha256)
    envelope = json.loads(Path(predictions_path).read_text())
    if digest(envelope) != predictions_sha256:
        raise ValueError('prediction digest mismatch')
    expected = {'version': 'nl001-inference-v1', 'split': split, 'split_sha256': digest(programs),
                'manifest_sha256': manifest_sha256, 'checkpoint_sha256': checkpoint_sha256,
                'arm': arm, 'edge_threshold': edge_threshold}
    if any(envelope.get(k) != v for k, v in expected.items()):
        raise ValueError('prediction binding mismatch')
    items = _items(programs)
    rows = envelope.get('predictions')
    if not isinstance(rows, list) or len(rows) != len(items):
        raise ValueError('prediction count mismatch')
    records, direct = [], []
    for (program, question), row in zip(items, rows):
        if not isinstance(row, dict) or set(row) != {'identity', 'prediction'} or row['identity'] != _identity(program, question):
            raise ValueError('prediction identity mismatch')
        raw = model_input(program, question)
        graph = Graph(tuple(raw['symbols']), frozenset(map(tuple, raw['observed_edges'])), raw['complete'], program['source'])
        records.append({'graph': graph, 'expected_query': question['query'], 'label': question['label'], 'prediction': row['prediction']})
        direct.append(row['prediction'])
    if arm == 'B':
        return evaluate(records)
    if arm != 'A':
        raise ValueError('invalid arm')
    correct = sum(p in ('Proven', 'Refuted') and p == q['label'] for p, (_, q) in zip(direct, items))
    answered = sum(p in ('Proven', 'Refuted') for p in direct)
    return {'claims': len(items), 'correct_answers': correct, 'answered': answered,
            'answer_accuracy': correct / len(items) if items else None,
            'conditional_answer_accuracy': correct / answered if answered else None,
            'certifications': 0, 'scope': 'direct predictions; no proofs'}
