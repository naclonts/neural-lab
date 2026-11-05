"""Shared scientific objective for subscription and API experiment planners."""

RESEARCH_DIRECTION = """Current research direction: AI-assisted logical verification of code.
Investigate whether compact neural networks can learn reusable reasoning and
search procedures over symbolic program graphs, while a deterministic verifier
checks every certified conclusion. Prioritize generalization and sound evidence
over raw in-distribution accuracy. Target the RTX 2070 with 8 GiB VRAM.

Progression: synthetic program graphs -> recursive proof search -> real Go
AST/call graphs -> Kubernetes packages -> richer program properties. Start with
NL-001: claim-to-logical-query translation and neural premise selection on a
precisely defined calls/reachable abstraction. Then study neural-guided proof
search with a compact GNN, graph transformer, or recurrent search policy. Train
small graph models from scratch where useful; optional 0.5B LM LoRA/QLoRA is a
baseline, not a requirement. Measure memory before considering a 1.5B model.

Compare A: pure neural answers, B: neural query/premise selection followed by
symbolic evaluation, and C: neural-guided symbolic search once A/B and the checker
are validated. Include an exact symbolic reference, a rule-based query parser,
and unguided symbolic search with matched search budgets where applicable.
Use a simple exact Python reference first; Souffle/Datalog or SMT are optional
later backends whose dependencies and semantics must be validated explicitly.

Separate Proven, Refuted, Supported, and Unknown. Proven means a checked proof
under recorded formal assumptions; Refuted requires a checked counterexample or
contradiction; Supported is source evidence without a proof; Unknown covers
incomplete premises, unsupported claims, invalid queries, and exhausted budgets.
Neural confidence is never a certificate. Validate claim/query alignment as well
as proof validity: a valid proof of the wrong translated claim is a false
verification. Static graph reachability does not prove a call happens at runtime
or on every execution. Failure to derive reachability proves unreachability only
for an explicitly complete graph with checked closed-world semantics. Otherwise
return Unknown. Preserve source locations, extraction scope, assumptions, query,
facts, proof trace or counterexample, and checker version for independent replay.

Split by entire generated programs and real repositories, never by questions or
snippets from the same program. Keep duplicate/isomorphic generated graphs out of
different splits and hold out question templates where relevant. Freeze generator
versions, seeds, splits, search limits, and evaluation before training. Train on
10-50-node graphs; prospectively test longer proof chains and 500, 5,000, then
50,000 nodes as sparse memory/runtime measurements permit. Do not retune on test
graphs. Include cycles, disconnected graphs, renamed symbols, missing facts,
unanswerable claims, malformed queries, and adversarial proof/query cases.

Report answer accuracy, end-to-end false verification rate (incorrectly certified
Proven / all certified Proven, with counts and uncertainty), proof coverage
(checked Proven or Refuted / all claims), Unknown and Supported rates, query
semantic accuracy, premise recall, proof replay success, and generalization by
graph size and chain length. Count abstentions in all-claim accuracy and report
conditional accuracy separately. Measure GPU memory, training/inference time,
CPU verifier time, expanded search states, and symbolic queries per answer.
Accuracy gains never compensate for unchecked certificates or missing failure
counts. Audit the checker against an independent exhaustive small-graph oracle
before scaling. Large codebases remain a later extraction/generalization test,
not an initial promise of arbitrary program correctness.

This objective supersedes older architecture-search priorities in experiment
summaries, queued hypotheses, and handoff notes. Preserve historical results;
do not launch new generic MLP/CIFAR/vision campaigns unless they directly test a
verification hypothesis. Research citations from operator discussions are leads,
not established facts: verify original sources before citing them or claiming
novelty. Keep all existing execution, credential, compute, and artifact limits.
"""
