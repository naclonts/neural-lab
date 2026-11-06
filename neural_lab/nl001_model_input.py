"""Leakage-free numeric representation of NL-001 controlled inputs.

Only ``model_input``'s inference-time fields are accepted. Symbol mentions are
encoded as positional pointers into the supplied symbol table; their spelling is
never a vocabulary item. Unrecognized words map to UNK, including unseen test
phrases. No graph edges are added or inferred.
"""
import re


VERSION = "nl001-input-v1"
MAX_NODES = 50
MAX_TOKENS = 32
# This vocabulary is fixed prospectively from train templates only.
WORDS = ("does", "call", "can", "reach", "by", "calls", "eventually", "execute")
VOCAB = {word: i + 4 for i, word in enumerate(WORDS)}
PAD, UNK, SYMBOL, PUNCT = 0, 1, 2, 3


def encode_input(item):
    if not isinstance(item, dict) or set(item) != {"claim", "symbols", "observed_edges", "complete"}:
        raise ValueError("input must contain only inference-time fields")
    claim, symbols, edges, complete = (item[k] for k in ("claim", "symbols", "observed_edges", "complete"))
    if (not isinstance(claim, str) or not isinstance(symbols, list) or not 0 < len(symbols) <= MAX_NODES
            or any(not isinstance(s, str) or not s for s in symbols) or len(set(symbols)) != len(symbols)
            or not isinstance(edges, list) or type(complete) is not bool):
        raise ValueError("invalid input")
    # Literal matching takes precedence over word segmentation. Reject a symbol
    # whose occurrence is a substring of another word or symbol.
    pattern = re.compile(r"(?<!\w)(?:" + "|".join(re.escape(s) for s in sorted(symbols, key=lambda s: (-len(s), s))) + r")(?!\w)|[A-Za-z]+|[^\s]")
    index = {s: i for i, s in enumerate(symbols)}
    tokens, pointers = [], []
    for match in pattern.finditer(claim):
        text = match.group()
        if text in index:
            start, end = match.span()
            if ((start and (claim[start - 1].isalnum() or claim[start - 1] == '_')) or
                    (end < len(claim) and (claim[end].isalnum() or claim[end] == '_'))):
                raise ValueError("ambiguous symbol boundary")
            tokens.append(SYMBOL)
            pointers.append(index[text] + 1)
        else:
            tokens.append(VOCAB.get(text.lower(), UNK) if text.isalpha() else PUNCT)
            pointers.append(0)
    if len(tokens) > MAX_TOKENS:
        raise ValueError("claim exceeds token limit")
    adjacency = [[0] * MAX_NODES for _ in range(MAX_NODES)]
    for edge in edges:
        if (not isinstance(edge, (list, tuple)) or len(edge) != 2 or
                edge[0] not in index or edge[1] not in index):
            raise ValueError("edge references undeclared symbol")
        adjacency[index[edge[0]]][index[edge[1]]] = 1
    token_array = [0] * MAX_TOKENS
    pointer_array = [0] * MAX_TOKENS
    token_array[:len(tokens)] = tokens
    pointer_array[:len(pointers)] = pointers
    return {"tokens": token_array, "pointers": pointer_array, "adjacency": adjacency,
            "node_count": len(symbols), "complete": int(complete)}
