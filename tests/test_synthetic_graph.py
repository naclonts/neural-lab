"""Exhaustive independent adjacency-matrix oracle for NL-001 semantics."""
import itertools
import unittest

from neural_lab.synthetic_graph import Graph, check, generate_graph, replay


def matrix_oracle(n, edges):
    # Boolean Warshall closure initialized only with edges, never the identity.
    matrix = [[(i, j) in edges for j in range(n)] for i in range(n)]
    for k in range(n):
        for i in range(n):
            for j in range(n):
                matrix[i][j] = matrix[i][j] or (matrix[i][k] and matrix[k][j])
    return matrix


class SyntheticGraphTests(unittest.TestCase):
    def test_all_directed_graphs_up_to_three_nodes(self):
        for n in range(1, 4):
            symbols = tuple(str(i) for i in range(n))
            pairs = list(itertools.product(range(n), repeat=2))
            for mask in range(1 << len(pairs)):
                edges = {edge for bit, edge in enumerate(pairs) if mask & (1 << bit)}
                graph = Graph(symbols, frozenset((str(i), str(j)) for i, j in edges), True, "exhaustive")
                oracle = matrix_oracle(n, edges)
                for i, j in pairs:
                    for prop, truth in (("calls", (i, j) in edges), ("reachable", oracle[i][j])):
                        query = {"property": prop, "subject": str(i), "object": str(j)}
                        answer = check(graph, query, claim="test", expected_query=query)
                        self.assertEqual(answer["result"], "Proven" if truth else "Refuted", (n, mask, query))
                        self.assertTrue(replay(graph, query, answer["result"], answer["certificate"]))

    def test_incomplete_invalid_and_translation(self):
        graph = Graph(("a", "b"), frozenset({("a", "b")}), False, "fixture")
        q = {"property": "reachable", "subject": "a", "object": "b"}
        self.assertEqual(check(graph, q, claim="x")["query_result"], "Proven")
        absent = {**q, "subject": "b", "object": "a"}
        self.assertEqual(check(graph, absent, claim="x")["result"], "Unknown")
        for bad in ({**q, "property": "eventually"}, {**q, "object": "missing"}, {}, None, {**q, "extra": 1}):
            self.assertEqual(check(graph, bad, claim="x")["result"], "Unknown")
        self.assertEqual(check(graph, q, claim="x", expected_query=absent)["result"], "Unknown")
        self.assertIsNone(check(graph, q, claim="x")["claim_aligned"])
        self.assertEqual(check(graph, q, claim="x")["result"], "Unknown")

    def test_corrupt_certificates_and_missing_facts(self):
        graph = Graph(("a", "b", "c"), frozenset({("a", "b"), ("b", "c")}), True, "fixture")
        q = {"property": "reachable", "subject": "a", "object": "c"}
        cert = check(graph, q, claim="x", expected_query=q)["certificate"]
        self.assertFalse(replay(graph, q, "Proven", {"kind": "path", "nodes": ["a", "c"]}))
        incomplete = Graph(graph.symbols, frozenset({("a", "b")}), False, "missing")
        self.assertFalse(replay(incomplete, q, "Proven", cert))
        self.assertFalse(replay(incomplete, q, "Refuted", {"kind": "closed_set", "nodes": ["a", "b"]}))
        self.assertFalse(replay(graph, q, "Refuted", {"kind": "closed_set", "nodes": ["a", "b"]}))
        self.assertFalse(replay(graph, q, "Proven", {"kind": "path", "nodes": ["a", "b", "c"], "extra": 1}))

    def test_generator_reproducible_and_validation(self):
        self.assertEqual(generate_graph(7, 4, 0.4), generate_graph(7, 4, 0.4))
        with self.assertRaises(ValueError):
            Graph(("a",), frozenset({("a", "b")}), True, "bad")
        with self.assertRaises(ValueError):
            generate_graph(1, 4, float("nan"))


if __name__ == "__main__":
    unittest.main()
