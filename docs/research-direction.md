# AI-assisted logical verification

Neural Lab now studies compact neural networks that construct queries, retrieve
premises, and guide proof search over program graphs. A deterministic checker
establishes the logical guarantees. The central question is whether the learned
procedure transfers from small training graphs to much larger unfamiliar
programs, with independently replayable evidence and few false certifications.

This is the active agenda for future experiments. The shared instructions in
`neural_lab/research_direction.py` are included in the headless scientist prompt
and both the subscription and API planner prompts. Older private handoffs and
experiment notes are historical evidence, not instructions to continue an older
campaign. `DemoPlanner` and the manually invoked starter/candidate scripts remain
infrastructure demonstrations.

## First campaign: NL-001

Hypothesis: a compact neural model that constructs symbolic queries and selects
premises generalizes better to unseen programs and longer dependency chains than
a comparable model that predicts answers directly, while producing independently
checkable evidence. This is a hypothesis to test, not an expected result.

The scientist should implement and independently review the following work before
launching substantive training:

1. Define a versioned synthetic program-graph generator, query schema, and exact
   reference checker. Start with `calls(A,B)` and nonempty-path transitive
   `reachable(A,B)`. Explicitly define cycles, self-reachability, symbol resolution,
   graph completeness, and unsupported properties. Use an exhaustive small-graph
   oracle to check the implementation independently. Test corrupted certificates,
   omitted facts, invalid symbols, malformed queries, and incorrect translations.
2. Generate whole-program train/validation/test splits with disjoint generator
   seeds and duplicate/isomorphism checks across splits. Start with 10-50 nodes
   and both reachable and unreachable pairs. Save exact graphs, questions, logical
   queries, evidence, split manifests, and hashes. Missing-fact and unsupported
   claims must have explicit Unknown labels. Begin with controlled question
   templates; hold out templates and renamed symbols to measure language transfer.
   Scale toward 10k-50k questions only after correctness and throughput checks.
3. Freeze A (pure neural answer prediction), B (neural claim-to-query translation
   plus premise selection and exact verification), and the exact symbolic oracle
   and rule-based parser controls. Give the neural arms the same observed facts,
   training splits, and declared parameter/compute budgets; report mismatches.
   A's predictions are predictions, never certified proofs. B must account for
   translation failures end to end, even if its proof of the emitted query is valid.
4. Queue a short CUDA calibration, then justified paired-seed substantive runs
   with explicit timeouts and fair training budgets. Models must fit the RTX 2070
   8 GiB. A small graph model trained from scratch is a useful starting point;
   0.5B LoRA/QLoRA is an optional language baseline. A 1.5B model requires measured
   memory headroom. Report negative results and do not pad work to reach a runtime.
5. Evaluate longer proof chains separately from larger graph sizes. Prospectively
   progress through 500, 5,000, and 50,000 nodes using sparse representations as
   runtime and memory allow. Report any size that was not evaluated, with the
   measured resource limit; do not silently truncate graphs or count incomplete
   searches as refutations. Freeze all thresholds before held-out evaluation.

After validating A/B, add C: a GNN, graph transformer, or small recurrent policy
choosing premises, edges, or next proof operations with verifier feedback. Compare
against unguided search, deterministic graph traversal, and random premise ranking
under matched expansion/query budgets. A policy must improve a measured cost,
coverage, or generalization tradeoff over exact traversal to justify its added
complexity. Reachability alone is an initial controlled task, not a novelty claim.

## Verification contract

Every answer records the original claim, structured query, observed fact set,
analysis scope, extraction completeness, assumptions, source/generator locations,
checker version, and a proof trace, counterexample, or reason for abstention.
Replay certificates in an independent evaluation pass without neural assistance.

| Result | Required meaning |
| --- | --- |
| Proven | The stated claim follows from explicit assumptions and a checked proof. |
| Refuted | A checked counterexample or contradiction establishes the claim is false within the declared scope. |
| Supported | Source evidence supports the claim, without a formal proof. |
| Unknown | Facts are incomplete, the claim/query is unsupported or invalid, or the search budget is exhausted. |

Positive paths establish reachability in the declared graph abstraction. They do
not establish that a call happens at runtime, happens on every execution, or that
a system eventually performs an action. Failure to derive a path is a negative
answer only for a complete graph under explicit closed-world semantics. For an
incomplete extracted graph, return Unknown unless a separate sound argument
establishes a negative result. Distinguish a complete synthetic graph from an
extractor's potentially incomplete view of a real program.

A proof of the wrong translated query does not verify the user's claim. Evaluate
claim/query semantic alignment against held-out ground truth and include errors
in the end-to-end false-verification metric. Keep checker correctness, extraction
fidelity, and neural translation/search accuracy as separate measured quantities.

## Evaluation and artifacts

Report paired seeds, counts, and uncertainty alongside aggregate metrics:

- Answer accuracy over all claims, with abstentions counted; conditional accuracy
  on answered claims is a separate metric.
- False verification rate: incorrect claims certified Proven / all claims
  certified Proven. Report numerator and denominator. Zero certifications means
  undefined, not zero risk; zero observed errors needs a confidence bound.
- Proof coverage: checked Proven or Refuted answers / all claims, with separate
  Unknown and Supported rates. Supported is not proof coverage.
- Query semantic accuracy, premise recall, and independently replayed proof rate.
- Generalization by graph size, topology, and proof length, including cycles,
  disconnected nodes, symbol renaming, missing facts, and unanswerable claims.
- Training and inference time, peak GPU memory, CPU verifier time, expanded states,
  and symbolic queries per answer. Record CPU preparation separately from training.

Use the existing artifact writer for data provenance and weights. Save the frozen
protocol, generator/configuration, exact split payloads and hashes, query schemas,
checker code/version, per-claim predictions, and proof/counterexample payloads
alongside `artifacts.json`. Keep scalar metrics in the queue's existing
`result.json` contract; link richer evidence from the summary and research notes.
Only test untouched held-out graphs after model/checkpoint selection. Audit a
checker change before comparing its new results with earlier runs.

## Later stages

Move to real Go AST/call graphs only after synthetic results and certificate replay
are reproducible. Record parser/extractor versions, repository commit, file and
line locations, build tags, and handling of interfaces, dynamic dispatch,
reflection, generated code, and external calls. Split real evaluation by entire
repositories. Kubernetes packages are a later transfer target; richer control flow,
data flow, SMT obligations, and temporal properties require new semantics and new
checker validation. A matching toleration removing a taint restriction, for
example, does not guarantee eventual scheduling.

Start with an exact Python reference requiring no new packages. Souffle/Datalog,
Tree-sitter, SMT, or CodeQL can be evaluated later if their dependencies, licenses,
and assumptions suit a bounded experiment. Do not assume these tools are already
installed or that extracted facts describe all runtime behavior. Verify original
research sources before citing them; the conversation's suggested papers and
dates are unverified leads.

## Adopting the direction on an existing laptop

Pull the updated `main` branch in the laptop checkout and restart the scientist:

```bash
bash scripts/research.sh pause-research
git pull --ff-only
bash scripts/research.sh resume-research
```

The next session loads the new agenda and writes a transition note before
planning. Startup no longer adds generic MLP starter/candidate jobs. Existing
queued/running jobs and historical artifacts are preserved; they can finish in
their existing FIFO order. The scientist must inspect that backlog and record its
effect on the new campaign rather than silently cancelling jobs or changing their
frozen protocols. A live session must be restarted to load the changed prompt.
This repository update does not itself restart the laptop or launch experiments.
