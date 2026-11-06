"""Deterministic claim-level NL-001 accounting; no model or training dependencies.

Ground-truth queries and labels are trusted inputs. Replay verifies graph facts,
not natural-language interpretation or graph extraction completeness.
"""
from math import sqrt
from statistics import NormalDist

from .synthetic_graph import Graph, replay

LABELS = frozenset({"Proven", "Refuted", "Unknown", "Supported"})


def _query_shape(query):
    return (isinstance(query, dict) and set(query) == {"property", "subject", "object"}
            and all(isinstance(query[key], str) and query[key] for key in query))


def _upper_wilson(errors, total, confidence):
    if not total:
        return None
    z = NormalDist().inv_cdf(confidence)
    p = errors / total
    return min(1.0, (p + z*z/(2*total) + z*sqrt(p*(1-p)/total + z*z/(4*total*total))) / (1 + z*z/total))


def evaluate(records, *, confidence=0.95):
    """Evaluate iterable of {graph, expected_query, label, prediction} mappings.

    Predictions are untrusted checker-like mappings with query, result, and
    certificate. Invalid/missing predictions and failed replay become Unknown.
    A replayed but misaligned Proven still counts as a false certification.
    Accuracy counts every abstention as incorrect, including on Unknown labels.
    The false-verification upper bound is one-sided Wilson (approximate).
    Invalid trusted records raise ValueError rather than silently skew metrics.
    """
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0.5 < confidence < 1:
        raise ValueError("confidence must be strictly between 0.5 and 1")
    if not isinstance(records, (list, tuple)):
        raise ValueError("records must be a list or tuple")
    counts = {k: 0 for k in ("claims", "correct_answers", "answered", "aligned_queries", "proven", "refuted", "unknown", "supported", "replay_attempts", "replay_passed", "replay_rejected", "false_proven", "claim_certified", "reported_supported", "aligned_label_conflicts")}
    details = []
    for item in records:
        if not isinstance(item, dict) or not isinstance(item.get("graph"), Graph) or not _query_shape(item.get("expected_query")) or not isinstance(item.get("label"), str) or item.get("label") not in LABELS:
            raise ValueError("invalid trusted evaluation record")
        graph, expected, label = item["graph"], item["expected_query"], item["label"]
        prediction = item.get("prediction")
        prediction = prediction if isinstance(prediction, dict) else {}
        query, status = prediction.get("query"), prediction.get("result")
        aligned = _query_shape(query) and query == expected
        counts["claims"] += 1
        counts["aligned_queries"] += int(aligned)
        replayed = False
        if status in ("Proven", "Refuted"):
            counts["replay_attempts"] += 1
            try:
                replayed = bool(replay(graph, query, status, prediction.get("certificate")))
            except (TypeError, ValueError, KeyError):
                replayed = False
            counts["replay_passed" if replayed else "replay_rejected"] += 1
            if not replayed:
                status = "Unknown"
        elif status == "Supported":
            # There is no source-evidence validator in this module.
            counts["reported_supported"] += 1
            status = "Unknown"
        elif status != "Unknown":
            status = "Unknown"
        counts[status.lower()] += 1
        if status == "Proven" and (not aligned or label != "Proven"):
            counts["false_proven"] += 1
        correct = status in ("Proven", "Refuted") and aligned and status == label
        counts["correct_answers"] += int(correct)
        counts["answered"] += int(status in ("Proven", "Refuted"))
        counts["claim_certified"] += int(status in ("Proven", "Refuted") and aligned)
        counts["aligned_label_conflicts"] += int(status in ("Proven", "Refuted") and aligned and status != label)
        details.append({"status": status, "query_aligned": aligned, "replayed": replayed, "correct": correct})
    n = counts["claims"]
    def rate(a, b):
        return a / b if b else None
    return {"counts": counts, "answer_accuracy": rate(counts["correct_answers"], n),
            "conditional_answer_accuracy": rate(counts["correct_answers"], counts["answered"]),
            "query_semantic_accuracy": rate(counts["aligned_queries"], n),
            "proof_coverage": rate(counts["claim_certified"], n),
            "unknown_rate": rate(counts["unknown"], n), "supported_rate": rate(counts["supported"], n),
            "replay_rate": rate(counts["replay_passed"], counts["replay_attempts"]),
            "false_verification_rate": rate(counts["false_proven"], counts["proven"]),
            "false_verification_upper_bound": _upper_wilson(counts["false_proven"], counts["proven"], confidence),
            "confidence": confidence, "confidence_method": "one-sided Wilson score (approximate)", "details": details}
