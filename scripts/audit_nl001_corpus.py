#!/usr/bin/env python3
"""Audit a pinned full corpus offline, outside the model-selection environment.

The digest is the canonical corpus body digest, retained independently before
this command. Output contains test-set aggregate information; keep it away from
model selection. This command does not establish custody or text alignment.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neural_lab.nl001_corpus_audit import audit_corpus
from neural_lab.nl001_independent_oracle import audit as semantic_audit
from neural_lab.nl001_data import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('corpus', type=Path)
    parser.add_argument('--corpus-sha256', required=True, help='independently retained canonical corpus digest')
    parser.add_argument('--output', type=Path, required=True, help='new output directory')
    parser.add_argument('--max-states', type=int, default=1_000_000)
    parser.add_argument('--max-comparisons', type=int, default=1_000_000, help='aggregate observed-view comparison cap')
    parser.add_argument('--max-total-states', type=int, default=1_000_000, help='aggregate isomorphism backtracking state cap')
    args = parser.parse_args()
    corpus = json.loads(args.corpus.read_text(encoding='utf-8'))
    # Perform every check before publishing any report; exhaustion propagates.
    report = audit_corpus(corpus, args.corpus_sha256, max_states=args.max_states, max_comparisons=args.max_comparisons,
                          max_total_states=args.max_total_states)
    semantics = semantic_audit(corpus)
    semantic_report = {'version': 'nl001-independent-semantic-audit-v1',
                       'corpus_sha256': args.corpus_sha256, 'result': semantics,
                       'limitations': 'Does not establish text/query alignment, corpus custody, split independence, or presentation identity.'}
    totals = {name: 0 for name in ('train', 'validation', 'test')}
    for row in semantics['counts']:
        totals[row['split']] += row['count']
    attestation = {'version': 'nl001-semantic-count-attestation-v1',
                   'corpus_sha256': args.corpus_sha256, 'question_counts': totals,
                   'claims': semantics['claims'], 'semantic_audit_sha256': digest(semantic_report)}
    args.output.mkdir()
    for name, value in [('corpus-audit.json', report), ('semantic-audit.json', semantic_report),
                        ('semantic-attestation.json', attestation)]:
        with (args.output / name).open('x', encoding='utf-8') as f:
            json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
            f.write('\n')
    print(json.dumps({'corpus_audit_sha256': digest(report),
                      'semantic_audit_sha256': digest(semantic_report),
                      'semantic_attestation_sha256': digest(attestation)}, sort_keys=True))


if __name__ == '__main__':
    main()
