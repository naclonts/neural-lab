#!/usr/bin/env python3
"""Audit a pinned controlled corpus against independently retained counts."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neural_lab.nl001_alignment_audit import audit, digest

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('corpus', type=Path)
p.add_argument('--corpus-sha256', required=True)
p.add_argument('--expected-counts', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
report = audit(json.loads(a.corpus.read_text()), a.corpus_sha256, json.loads(a.expected_counts.read_text()))
with a.output.open('x', encoding='utf-8') as f:
    json.dump(report, f, sort_keys=True, indent=2, allow_nan=False)
    f.write('\n')
print(json.dumps({'alignment_audit_sha256': digest(report)}, sort_keys=True))
