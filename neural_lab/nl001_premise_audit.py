"""Measure selected-edge recall against one recorded positive witness.

This is a diagnostic, not a certificate or a queue entry point. Pins and the
choice of threshold must be frozen independently before examining the split.
"""
import argparse
import json
from pathlib import Path

import torch

from .nl001_data import digest, model_input
from .nl001_inference import _checkpoint_model, _identity, _items
from .nl001_manifest import authenticate_split
from .nl001_model_input import encode_input
from .synthetic_graph import Graph, replay


def audit(*, programs, split, manifest, manifest_sha256, checkpoint,
          checkpoint_sha256, edge_threshold=0.5):
    authenticate_split(programs, split, manifest, manifest_sha256)
    if (isinstance(edge_threshold, bool) or not isinstance(edge_threshold, (float, int))
            or not 0 <= edge_threshold <= 1):
        raise ValueError('invalid edge threshold')
    model = _checkpoint_model(checkpoint, checkpoint_sha256, 'B')
    rows = []
    with torch.inference_mode():
        for program, question in _items(programs):
            if question['label'] != 'Proven':
                continue
            raw = model_input(program, question)
            graph = Graph(tuple(raw['symbols']), frozenset(map(tuple, raw['observed_edges'])),
                          raw['complete'], program['source'])
            cert = question['certificate']
            if not replay(graph, question['query'], 'Proven', cert):
                raise ValueError('invalid reference witness')
            witness = set(zip(cert['nodes'], cert['nodes'][1:]))
            encoded = encode_input(raw)
            output = model({key: torch.tensor([value]) for key, value in encoded.items()})
            scores = torch.sigmoid(output['edges'][0])
            indices = {symbol: i for i, symbol in enumerate(raw['symbols'])}
            selected = {(a, b) for a, b in raw['observed_edges']
                        if scores[indices[a], indices[b]].item() >= edge_threshold}
            rows.append({'identity': _identity(program, question),
                         'witness_edges': len(witness),
                         'recalled_edges': len(witness & selected),
                         'selected_edges': len(selected),
                         'full_witness_selected': witness <= selected})
    total = sum(r['witness_edges'] for r in rows)
    recalled = sum(r['recalled_edges'] for r in rows)
    return {'version': 'nl001-premise-audit-v1', 'split': split,
            'split_sha256': digest(programs), 'manifest_sha256': manifest_sha256,
            'checkpoint_sha256': checkpoint_sha256, 'edge_threshold': edge_threshold,
            'positive_claims': len(rows), 'witness_edges': total, 'recalled_edges': recalled,
            'micro_witness_recall': recalled / total if total else None,
            'full_witnesses_selected': sum(r['full_witness_selected'] for r in rows),
            'rows': rows,
            'scope': 'Recall of one recorded valid witness on positive claims; alternative proofs are not counted. Repeated presentations and views are separate claims.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('request', help='JSON arguments to audit')
    parser.add_argument('output')
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text())
    result = audit(**request)
    with Path(args.output).open('x') as out:
        json.dump(result, out, sort_keys=True, allow_nan=False, indent=2)


if __name__ == '__main__':
    main()
