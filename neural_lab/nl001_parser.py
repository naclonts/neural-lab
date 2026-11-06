"""Fail-closed parser for the exact NL-001 controlled templates.

Parsing receives only a claim and the declared symbol table; gold queries are
never inputs. Symbols are matched as complete literal substrings, including any
punctuation or regex metacharacters; ambiguous parses abstain.
"""
from .nl001_data import TEMPLATES

PARSER_VERSION = "nl001-template-parser-v1"


def parse_claim(claim, symbols):
    """Return a structured query, or None for unsupported/ambiguous input."""
    if (not isinstance(claim, str) or not isinstance(symbols, (list, tuple)) or
        any(not isinstance(s, str) or not s for s in symbols) or
        len(set(symbols)) != len(symbols)):
        return None
    # Match entire generated strings. Enumerating pairs avoids regex ambiguity,
    # escaping mistakes, substring resolution and untrusted format strings.
    matches = set()
    for templates in TEMPLATES.values():
        for prop in ("calls", "reachable", "unsupported"):
            template = templates[prop]
            for a in symbols:
                for b in symbols:
                    if template.format(a=a, b=b) == claim:
                        matches.add((prop, a, b))
    if len(matches) != 1:
        return None
    prop, a, b = matches.pop()
    if prop == "unsupported":
        return None
    return {"property": prop, "subject": a, "object": b}
