"""Versioned, reproducible whole-program splits for NL-001 (standard library only).

Ground truth and observed inputs are stored together for audit, so training loaders
must explicitly exclude hidden edges, labels, certificates and held-out splits.
"""
from __future__ import annotations

import hashlib
import json
import random

from .synthetic_graph import CHECKER_VERSION, Graph, check

VERSION = "nl001-whole-program-v1"


class IsomorphicProgramsError(ValueError):
    """A generated corpus contains duplicate full graphs."""


TEMPLATES = {
    "train": {"calls": "Does {a} call {b}?", "reachable": "Can {a} reach {b} by calls?", "unsupported": "Does {a} eventually execute {b}?"},
    "validation": {"calls": "Does {a} call {b}?", "reachable": "Can {a} reach {b} by calls?", "unsupported": "Does {a} eventually execute {b}?"},
    "test": {"calls": "Is there a direct call from {a} to {b}?", "reachable": "Is {b} reachable from {a} through one or more calls?", "unsupported": "Will {a} eventually execute {b}?"},
}


def _bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(_bytes(value)).hexdigest()


def isomorphic(left, right, *, max_states=1_000_000, state_callback=None):
    """Exact directed-graph isomorphism, including loops; fail closed on budget.

    Node names and edge ordering have no significance. Degree filtering and
    adjacency-preserving backtracking are exact but may be expensive on symmetric
    graphs. A budget exception is not evidence of non-isomorphism.
    """
    if len(left.symbols) != len(right.symbols) or len(left.edges) != len(right.edges):
        return False
    def signature(g, x):
        return (sum(a == x for a, b in g.edges), sum(b == x for a, b in g.edges), (x, x) in g.edges)
    ls = {x: signature(left, x) for x in left.symbols}
    rs = {x: signature(right, x) for x in right.symbols}
    if sorted(ls.values()) != sorted(rs.values()):
        return False
    candidates = {x: [y for y in right.symbols if ls[x] == rs[y]] for x in left.symbols}
    order = sorted(left.symbols, key=lambda x: (len(candidates[x]), x))
    mapping, used, states = {}, set(), 0
    def visit(i):
        nonlocal states
        states += 1
        if state_callback is not None:
            state_callback()
        if states > max_states:
            raise RuntimeError("isomorphism state budget exhausted; split is not validated")
        if i == len(order):
            return True
        x = order[i]
        for y in candidates[x]:
            if y in used or any(((x, u) in left.edges) != ((y, v) in right.edges) or
                                ((u, x) in left.edges) != ((v, y) in right.edges)
                                for u, v in mapping.items()):
                continue
            mapping[x] = y
            used.add(y)
            if visit(i + 1):
                return True
            used.remove(y)
            del mapping[x]
        return False
    return visit(0)


def node_sizes(nodes, seeds):
    """Resolve per-seed program sizes; a plain int keeps uniform v1 sizing.

    A mapping assigns each generator seed its own program size within the same
    10..50 bound, so one corpus can span several node sizes. Mapping keys may
    be ints or ASCII digit strings so a JSON round trip of the stored config
    regenerates an identical payload; the mapping is stored as passed.
    """
    if type(nodes) is int:
        if not 10 <= nodes <= 50:
            raise ValueError("nodes must be an integer between 10 and 50")
        return None
    if not isinstance(nodes, dict) or not nodes:
        raise ValueError("nodes must be an integer between 10 and 50 or a per-seed mapping")
    sizes = {}
    for key, value in nodes.items():
        if type(key) is int:
            seed = key
        elif isinstance(key, str) and key.isascii() and key.isdigit():
            seed = int(key)
        else:
            raise ValueError("per-seed node mapping keys must be generator seeds")
        if type(value) is not int or not 10 <= value <= 50:
            raise ValueError("per-seed node counts must be integers between 10 and 50")
        if seed in sizes:
            raise ValueError("duplicate seed in per-seed node mapping")
        sizes[seed] = value
    if set(sizes) != set(seeds):
        raise ValueError("per-seed node mapping must cover exactly the generator seeds")
    return sizes


def make_splits(split_seeds, *, nodes=10, edge_probability=0.08, missing_fraction=0.25):
    """Build exact auditable payload; reject seed reuse and any cross-split iso.

    Each program has a complete view and a deterministic incomplete view with at
    least one removed edge. All ordered pairs are queried in both views. This is
    a small correctness fixture, not a sampled/balanced substantive protocol.
    """
    if set(split_seeds) != set(TEMPLATES):
        raise ValueError("train, validation and test seed lists are required")
    if any(not isinstance(v, (list, tuple)) or not v for v in split_seeds.values()):
        raise ValueError("each split needs at least one seed")
    seeds = [seed for values in split_seeds.values() for seed in values]
    if any(type(s) is not int for s in seeds) or len(seeds) != len(set(seeds)):
        raise ValueError("generator seeds must be distinct integers")
    sizes = node_sizes(nodes, seeds)
    if (isinstance(edge_probability, bool) or not 0 < edge_probability < 1 or
        isinstance(missing_fraction, bool) or not 0 < missing_fraction <= 1):
        raise ValueError("probability must be in (0,1), missing fraction in (0,1]")
    graphs, programs = {}, {}
    for split in ("train", "validation", "test"):
        graphs[split], programs[split] = [], []
        for seed in split_seeds[split]:
            rng = random.Random(seed)
            size = nodes if sizes is None else sizes[seed]
            # Split-specific names ensure renamed-symbol evaluation across splits.
            names = tuple(f"{split[0]}{seed}_{i}" for i in range(size))
            edges = frozenset((a, b) for a in names for b in names if rng.random() < edge_probability)
            if not edges:
                raise ValueError(f"seed {seed} generated no edge to remove; choose another seed")
            source = f"{VERSION}:split={split}:seed={seed}:nodes={size}:p={edge_probability}"
            full = Graph(names, edges, True, source)
            for previous in graphs.values():
                # Same-split duplicates are harmless to isolation; compare all
                # prior graphs anyway so the full corpus has unique programs.
                for other in previous:
                    if isomorphic(full, other):
                        raise IsomorphicProgramsError("isomorphic programs detected; choose different seeds")
            graphs[split].append(full)
            shuffled = sorted(edges)
            rng.shuffle(shuffled)
            removed = frozenset(shuffled[:max(1, round(len(shuffled) * missing_fraction))])
            views = [full, Graph(names, edges - removed, False, source + ":incomplete")]
            records = []
            for graph in views:
                for prop in ("calls", "reachable", "unsupported"):
                    # One unsupported claim per view suffices for the fixture.
                    pairs = [(names[0], names[-1])] if prop == "unsupported" else [(a, b) for a in names for b in names]
                    for a, b in pairs:
                        query = {"property": prop, "subject": a, "object": b}
                        claim = TEMPLATES[split][prop].format(a=a, b=b)
                        checked = check(graph, query, claim=claim, expected_query=query)
                        records.append({"view": "complete" if graph.complete else "incomplete", "claim": claim,
                                        "template_id": f"{split}:{prop}", "query": query, "label": checked["result"],
                                        "certificate": checked["certificate"], "reason": checked["reason"]})
            programs[split].append({"seed": seed, "source": source, "symbols": list(names),
                                    "full_edges": sorted([list(e) for e in edges]),
                                    "observed_incomplete_edges": sorted([list(e) for e in edges - removed]),
                                    "removed_edges": sorted([list(e) for e in removed]), "questions": records})
    payload = {"version": VERSION, "config": {"nodes": nodes, "edge_probability": edge_probability,
               "missing_fraction": missing_fraction, "split_seeds": split_seeds},
               "templates": TEMPLATES, "checker_version": CHECKER_VERSION, "splits": programs,
               "split_sha256": {split: digest(items) for split, items in programs.items()}}
    payload["sha256"] = digest(payload)
    return payload


def model_input(program, question):
    """Return only inference-time inputs, excluding hidden facts and gold data."""
    view = question["view"]
    if view not in ("complete", "incomplete"):
        raise ValueError("unknown observation view")
    edges = program["full_edges"] if view == "complete" else program["observed_incomplete_edges"]
    return {"claim": question["claim"], "symbols": list(program["symbols"]),
            "observed_edges": [list(edge) for edge in edges], "complete": view == "complete"}
