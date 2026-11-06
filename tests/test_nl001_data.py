import copy
import unittest
from unittest.mock import patch

from neural_lab.nl001_data import digest, isomorphic, make_splits, model_input
from neural_lab.synthetic_graph import Graph, replay


def graph(names, edges):
    return Graph(tuple(names), frozenset(edges), True, "test")


def oracle(edges, a, b, prop):
    if prop == "calls":
        return (a, b) in edges
    # Independent transitive closure, initialized with nonempty edges only.
    closure = set(edges)
    vertices = {a, b} | {v for edge in edges for v in edge}
    for k in vertices:
        for i in vertices:
            for j in vertices:
                if (i, k) in closure and (k, j) in closure:
                    closure.add((i, j))
    return (a, b) in closure


class DataTests(unittest.TestCase):
    def test_exact_isomorphism_and_budget(self):
        a = graph("abc", [("a", "b"), ("b", "c"), ("c", "a")])
        b = graph("xyz", [("x", "z"), ("z", "y"), ("y", "x")])
        c = graph("xyz", [("x", "x"), ("z", "y"), ("y", "z")])
        self.assertTrue(isomorphic(a, b))
        self.assertFalse(isomorphic(a, c))
        with self.assertRaises(RuntimeError):
            isomorphic(a, b, max_states=1)

    def test_cross_split_isomorphic_permutations_rejected(self):
        # Seed 1 creates an asymmetric directed path with a loop; seed 2 emits
        # the same graph under a reversal of the vertex numbering. Different
        # program and split names must not hide the structural duplicate.
        class PermutedRng:
            def __init__(self, seed):
                self.seed, self.index = seed, 0

            def random(self):
                i, j = divmod(self.index, 10)
                self.index += 1
                if self.seed == 2:
                    i, j = 9 - i, 9 - j
                return 0.0 if (i, j) in {(0, 0), (0, 1), (1, 2), (2, 3)} else 1.0

            def shuffle(self, values):
                pass

        with patch("neural_lab.nl001_data.random.Random", PermutedRng):
            with self.assertRaisesRegex(ValueError, "isomorphic programs"):
                make_splits({"train": [1], "validation": [2], "test": [3]})

    def test_deterministic_split_and_independent_label_oracle(self):
        seeds = {"train": [1], "validation": [2], "test": [3]}
        payload = make_splits(seeds)
        self.assertEqual(payload, make_splits(seeds))
        unhashed = copy.deepcopy(payload)
        expected = unhashed.pop("sha256")
        self.assertEqual(expected, digest(unhashed))
        all_symbols, all_seeds, prior_graphs = set(), set(), []
        for split, programs in payload["splits"].items():
            self.assertEqual(payload["split_sha256"][split], digest(programs))
            for program in programs:
                symbols = set(program["symbols"])
                self.assertFalse(all_symbols & symbols)
                all_symbols.update(symbols)
                self.assertNotIn(program["seed"], all_seeds)
                all_seeds.add(program["seed"])
                self.assertIn(program["seed"], seeds[split])
                full = set(map(tuple, program["full_edges"]))
                incomplete = set(map(tuple, program["observed_incomplete_edges"]))
                this_graph = graph(program["symbols"], full)
                for earlier in prior_graphs:
                    self.assertFalse(isomorphic(this_graph, earlier))
                prior_graphs.append(this_graph)
                # Every view belongs to this same program, with each supported
                # ordered pair appearing exactly once for each property.
                keys = [(q["view"], q["query"]["property"], q["query"]["subject"], q["query"]["object"]) for q in program["questions"]]
                self.assertEqual(len(keys), len(set(keys)))
                for view in ("complete", "incomplete"):
                    for prop in ("calls", "reachable"):
                        self.assertEqual({(a, b) for v, p, a, b in keys if v == view and p == prop},
                                         {(a, b) for a in symbols for b in symbols})
                self.assertTrue(incomplete < full)
                for item in program["questions"]:
                    safe = model_input(program, item)
                    self.assertEqual(set(safe), {"claim", "symbols", "observed_edges", "complete"})
                    self.assertNotIn("query", safe)
                    complete = item["view"] == "complete"
                    edges = full if complete else incomplete
                    q = item["query"]
                    self.assertIn(q["subject"], symbols)
                    self.assertIn(q["object"], symbols)
                    self.assertTrue(all(a in symbols and b in symbols for a, b in edges))
                    prop = q["property"]
                    found = prop != "unsupported" and oracle(edges, q["subject"], q["object"], prop)
                    expected = "Proven" if found else "Refuted" if complete and prop != "unsupported" else "Unknown"
                    self.assertEqual(item["label"], expected)
                    if item["certificate"]:
                        g = Graph(tuple(program["symbols"]), frozenset(edges), complete, program["source"])
                        self.assertTrue(replay(g, q, expected, item["certificate"]))
                self.assertTrue(all(s.startswith(split[0]) for s in program["symbols"]))
        self.assertNotEqual(payload["templates"]["train"], payload["templates"]["test"])
        with self.assertRaises(ValueError):
            make_splits({"train": [1], "validation": [1], "test": [2]})


if __name__ == "__main__":
    unittest.main()
