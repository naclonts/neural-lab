import unittest

from neural_lab.nl001_evaluation import evaluate
from neural_lab.synthetic_graph import Graph, check


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.graph = Graph(("a", "b", "c"), frozenset({("a", "b")}), True, "fixture")
        self.yes = {"property": "calls", "subject": "a", "object": "b"}
        self.no = {"property": "calls", "subject": "a", "object": "c"}

    def row(self, expected, label, emitted=None):
        query = emitted or expected
        pred = check(self.graph, query, claim="claim", expected_query=query)
        return {"graph": self.graph, "expected_query": expected, "label": label, "prediction": pred}

    def test_alignment_replay_and_abstentions(self):
        good = self.row(self.yes, "Proven")
        false = self.row(self.no, "Refuted", self.yes)
        invalid = self.row(self.yes, "Proven")
        invalid["prediction"]["certificate"] = {"kind": "path", "nodes": ["a", "c"]}
        unknown = {"graph": self.graph, "expected_query": self.no, "label": "Unknown", "prediction": None}
        supported = {"graph": self.graph, "expected_query": self.no, "label": "Unknown", "prediction": {"query": self.no, "result": "Supported"}}
        r = evaluate([good, false, invalid, unknown, supported])
        self.assertEqual(r["counts"]["false_proven"], 1)
        self.assertEqual(r["counts"]["proven"], 2)
        self.assertEqual(r["counts"]["replay_rejected"], 1)
        self.assertEqual(r["answer_accuracy"], 0.2)
        self.assertEqual(r["proof_coverage"], 0.2)
        self.assertEqual(r["false_verification_rate"], 0.5)
        self.assertEqual(r["unknown_rate"], 0.6)
        self.assertEqual(r["supported_rate"], 0)
        self.assertEqual(r["counts"]["reported_supported"], 1)

    def test_zero_denominators_and_zero_errors_bound(self):
        empty = evaluate([])
        self.assertIsNone(empty["answer_accuracy"])
        self.assertIsNone(empty["false_verification_rate"])
        self.assertIsNone(empty["false_verification_upper_bound"])
        r = evaluate([self.row(self.yes, "Proven")])
        self.assertEqual(r["false_verification_rate"], 0)
        self.assertGreater(r["false_verification_upper_bound"], 0)
        self.assertLess(r["false_verification_upper_bound"], 1)

    def test_malformed_and_refutations(self):
        for record in ({}, {"graph": self.graph, "expected_query": self.no, "label": []}):
            with self.assertRaises(ValueError):
                evaluate([record])
        with self.assertRaises(ValueError):
            evaluate([], confidence=float("nan"))
        row = self.row(self.no, "Refuted")
        self.assertEqual(evaluate([row])["answer_accuracy"], 1)
        row["prediction"] = {"result": [], "query": None}
        self.assertEqual(evaluate([row])["unknown_rate"], 1)


if __name__ == "__main__":
    unittest.main()
