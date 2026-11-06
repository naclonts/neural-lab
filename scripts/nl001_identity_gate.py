#!/usr/bin/env python3
"""Create recipe-only identity anchors or audit a corpus against retained anchors.

Run construct before inspecting/generating the candidate; retain its digest and
source/runtime pins separately. This command cannot establish independent custody.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neural_lab.nl001_identity_manifest import construct
from neural_lab.nl001_alignment_audit import audit_expected_identities, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    create = sub.add_parser('construct')
    create.add_argument('recipe', type=Path)
    create.add_argument('output', type=Path)
    audit = sub.add_parser('audit')
    audit.add_argument('corpus', type=Path)
    audit.add_argument('identities', type=Path)
    audit.add_argument('--corpus-sha256', required=True)
    audit.add_argument('--identities-sha256', required=True)
    audit.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'construct':
        recipe = json.loads(args.recipe.read_text(encoding='utf-8'))
        value = construct(**recipe)
    else:
        corpus = json.loads(args.corpus.read_text(encoding='utf-8'))
        identities = json.loads(args.identities.read_text(encoding='utf-8'))
        counts = {split: [{'seed': p['seed'], 'questions_per_variant': len(p['questions'])}
                          for p in programs] for split, programs in identities.items()}
        value = audit_expected_identities(corpus, args.corpus_sha256, counts, identities, args.identities_sha256)
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n')
    print(digest(value))


if __name__ == '__main__':
    main()
