"""Independent adversarial checks for the controlled-template parser."""
import inspect
import unittest

from neural_lab.nl001_data import TEMPLATES
from neural_lab.nl001_parser import parse_claim


class ParserReviewTests(unittest.TestCase):
    def test_interface_excludes_gold_queries(self):
        self.assertEqual(list(inspect.signature(parse_claim).parameters), ["claim", "symbols"])

    def test_unsupported_collision_must_abstain(self):
        symbols = ["x", "y eventually execute z", "x call y", "z"]
        claim = TEMPLATES["train"]["unsupported"].format(a="x call y", b="z")
        self.assertEqual(claim, TEMPLATES["train"]["calls"].format(a="x", b="y eventually execute z"))
        self.assertIsNone(parse_claim(claim, symbols))

    def test_literal_symbols_and_malformed_claims(self):
        symbols = ["a.*", "{b}", "q?", "space name", "K"]
        for templates in TEMPLATES.values():
            for prop in ("calls", "reachable"):
                query = {"property": prop, "subject": "a.*", "object": "{b}"}
                claim = templates[prop].format(a=query["subject"], b=query["object"])
                self.assertEqual(parse_claim(claim, symbols), query)
                for corrupted in (claim + "\n", " " + claim, claim[:-1], claim + "?", claim.replace("{b}", "b")):
                    self.assertIsNone(parse_claim(corrupted, symbols))
        self.assertIsNone(parse_claim("Does K call {b}?", symbols))


if __name__ == "__main__":
    unittest.main()
