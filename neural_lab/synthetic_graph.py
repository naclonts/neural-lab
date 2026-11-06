"""NL-001 synthetic graph semantics and independently replayable certificates.

Version 1 uses exact, case-sensitive symbol names, directed edges, and paths of
one or more edges. A node reaches itself only through a self-loop or cycle.
Completeness describes the entire declared graph, not just selected premises.
No claim is made about runtime calls or properties other than these graph facts.
"""
from collections import deque
from dataclasses import dataclass
import random

CHECKER_VERSION = "nl001-checker-v1"
GENERATOR_VERSION = "nl001-generator-v1"


@dataclass(frozen=True)
class Graph:
    symbols: tuple[str, ...]
    edges: frozenset[tuple[str, str]]
    complete: bool
    source: str

    def __post_init__(self):
        if (not isinstance(self.symbols, tuple) or
            any(not isinstance(s, str) or not s for s in self.symbols) or
            len(set(self.symbols)) != len(self.symbols)):
            raise ValueError("symbols must be unique nonempty strings in a tuple")
        if not isinstance(self.edges, frozenset) or any(
            not isinstance(e, tuple) or len(e) != 2 or
            e[0] not in self.symbols or e[1] not in self.symbols
            for e in self.edges
        ):
            raise ValueError("edges must be pairs of declared symbols")
        if type(self.complete) is not bool or not isinstance(self.source, str) or not self.source:
            raise ValueError("complete must be boolean and source must be nonempty")


def generate_graph(seed: int, nodes: int, edge_probability: float) -> Graph:
    """Generate a reproducible complete directed graph, including possible loops."""
    if type(seed) is not int or type(nodes) is not int or nodes < 0:
        raise ValueError("seed and nonnegative nodes must be integers")
    if isinstance(edge_probability, bool) or not isinstance(edge_probability, (int, float)) or not 0 <= edge_probability <= 1:
        raise ValueError("edge_probability must be between 0 and 1")
    rng = random.Random(seed)
    symbols = tuple(f"n{i}" for i in range(nodes))
    edges = frozenset((a, b) for a in symbols for b in symbols if rng.random() < edge_probability)
    return Graph(symbols, edges, True, f"{GENERATOR_VERSION}:seed={seed},nodes={nodes},p={edge_probability}")


def valid_query(graph, query):
    return (isinstance(query, dict) and set(query) == {"property", "subject", "object"}
            and all(isinstance(query[k], str) for k in query)
            and query["property"] in {"calls", "reachable"}
            and query["subject"] in graph.symbols and query["object"] in graph.symbols)


def _check_query(graph, query, *, claim, expected_query=None, assumptions=()):
    """Check a query and emit evidence. expected_query is trusted alignment ground truth.

    Without expected_query the result concerns only the emitted structured query;
    it must not be counted as verification of the original natural-language claim.
    """
    base = {"checker_version": CHECKER_VERSION, "claim": claim, "query": query,
            "observed_facts": sorted(graph.edges), "symbols": list(graph.symbols),
            "analysis_scope": graph.source, "extraction_complete": graph.complete,
            "assumptions": list(assumptions), "source": graph.source,
            "claim_aligned": expected_query == query if expected_query is not None else None}
    if not valid_query(graph, query):
        return {**base, "result": "Unknown", "certificate": None, "reason": "invalid or unsupported query"}
    if expected_query is not None and (not valid_query(graph, expected_query) or query != expected_query):
        return {**base, "result": "Unknown", "certificate": None, "reason": "query does not match claim ground truth"}
    a, b, prop = query["subject"], query["object"], query["property"]
    if prop == "calls":
        if (a, b) in graph.edges:
            return {**base, "result": "Proven", "certificate": {"kind": "path", "nodes": [a, b]}, "reason": None}
        if graph.complete:
            return {**base, "result": "Refuted", "certificate": {"kind": "absent_edge", "edge": [a, b]}, "reason": None}
    else:
        adj = {s: [] for s in graph.symbols}
        for x, y in sorted(graph.edges):
            adj[x].append(y)
        queue = deque([(a, [a])])
        seen = {a}
        while queue:
            x, path = queue.popleft()
            for y in adj[x]:
                if y == b:
                    return {**base, "result": "Proven", "certificate": {"kind": "path", "nodes": path + [y]}, "reason": None}
                if y not in seen:
                    seen.add(y)
                    queue.append((y, path + [y]))
        if graph.complete:
            return {**base, "result": "Refuted", "certificate": {"kind": "closed_set", "nodes": sorted(seen)}, "reason": None}
    return {**base, "result": "Unknown", "certificate": None, "reason": "graph is incomplete"}


def replay(graph, query, result, certificate):
    """Validate a certificate against the full observed graph, not supplied premises."""
    if not valid_query(graph, query) or not isinstance(certificate, dict):
        return False
    a, b, prop = query["subject"], query["object"], query["property"]
    kind = certificate.get("kind")
    if result == "Proven" and kind == "path" and set(certificate) == {"kind", "nodes"}:
        path = certificate["nodes"]
        return (isinstance(path, list) and len(path) >= 2 and path[0] == a and path[-1] == b
                and (prop != "calls" or len(path) == 2)
                and all(isinstance(x, str) and x in graph.symbols for x in path)
                and all((x, y) in graph.edges for x, y in zip(path, path[1:])))
    if result != "Refuted" or not graph.complete:
        return False
    if prop == "calls" and kind == "absent_edge" and set(certificate) == {"kind", "edge"}:
        return certificate["edge"] == [a, b] and (a, b) not in graph.edges
    if prop == "reachable" and kind == "closed_set" and set(certificate) == {"kind", "nodes"}:
        nodes = certificate["nodes"]
        if not isinstance(nodes, list) or any(not isinstance(x, str) or x not in graph.symbols for x in nodes):
            return False
        closure = set(nodes)
        if len(closure) != len(nodes) or a not in closure:
            return False
        # For self-reachability, a is in the set initially; no edge may return to it.
        return all(y in closure and y != b for x, y in graph.edges if x in closure)
    return False


def check(graph, query, *, claim, expected_query=None, assumptions=()):
    """Return claim status and retain query-only evidence separately when unaligned."""
    answer = _check_query(graph, query, claim=claim, expected_query=expected_query, assumptions=assumptions)
    answer["query_result"] = answer["result"]
    if expected_query is None and answer["result"] in {"Proven", "Refuted"}:
        answer["result"] = "Unknown"
        answer["reason"] = "claim/query alignment not established; certificate is query-only"
    return answer
