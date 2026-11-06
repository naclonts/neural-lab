"""NL-001 calibration placeholder; deliberately cannot be enqueued as an experiment.

See .neural-lab/notes/nl001-calibration-draft.md for the proposed architecture,
contract, and unresolved implementation and validation work. This file is not a
training implementation and must never produce result.json or claim completion.
"""

raise SystemExit(
    "NL-001 calibration is blocked: model training, leakage-free data loading, "
    "claim-level scoring, independent certificate replay, frozen held-out protocol, "
    "and self-contained queue snapshot have not yet been implemented and reviewed."
)
