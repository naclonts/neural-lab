import unittest

from neural_lab.nl001_data import TEMPLATES
from neural_lab.nl001_parser import parse_claim


class ParserTests(unittest.TestCase):
    def test_all_templates_and_exact_symbols(self):
        symbols = ["a", "a.b", "x+y", "n 0"]
        for templates in TEMPLATES.values():
            for prop in ("calls", "reachable"):
                for a in symbols:
                    for b in symbols:
                        claim = templates[prop].format(a=a, b=b)
                        self.assertEqual(parse_claim(claim, symbols),
                                         {"property": prop, "subject": a, "object": b})

    def test_fail_closed(self):
        symbols = ["a", "b"]
        for templates in TEMPLATES.values():
            self.assertIsNone(parse_claim(templates["unsupported"].format(a="a", b="b"), symbols))
        for claim in ("Does a call c?", "does a call b?", "Does a call b? ", "Does a call b", "", None):
            self.assertIsNone(parse_claim(claim, symbols))
        for bad in (None, ["a", "a"], ["a", ""], ["a", 1], {"a", "b"}):
            self.assertIsNone(parse_claim("Does a call b?", bad))

    def test_supported_unsupported_collision_abstains(self):
        symbols = ["x", "y eventually execute z", "x call y", "z"]
        self.assertIsNone(parse_claim("Does x call y eventually execute z?", symbols))

    def test_ambiguous_literal_symbols_abstain(self):
        symbols = ["a", "b call c", "a call b", "c"]
        self.assertIsNone(parse_claim("Does a call b call c?", symbols))


if __name__ == "__main__":
    unittest.main()
