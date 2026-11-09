"""UNREVIEWED DRAFT — NL-001 substantive paired A/B CUDA candidate. DO NOT ENQUEUE before independent review.

Session-44 packaging repair (provenance): derived from the failed queued
seed-41001 script .neural-lab/queue-inputs/d52f8c215290475baed813eeeeeffa3a/train.py
(sha256 076f28176f915a45b4aecf1bce793f0262ef91a4162b5a9365a399ad84619e2c).
All four queued substantive jobs (seeds 41001-41004) finished training and then
failed in Phase 5 with ModuleNotFoundError: neural_lab.nl001_inference because
PACKAGE_SHA256 omitted nl001_inference.py; the local smoke passed spuriously
because the repo package was importable and the custody check was skipped in
smoke mode. Changes vs the failed script, and ONLY these: (1) nl001_inference.py
added to PACKAGE_SHA256 (digest verified against the repo file); (2) fail-fast
verify_snapshot_custody() called immediately after snapshot_package, importing
every packaged module and asserting every loaded neural_lab module resolves
inside the snapshot, in BOTH smoke and non-smoke modes (snapshot_package
therefore now also runs in smoke mode); (3) the end-of-run sys.modules custody
check is no longer gated on 'if not smoke' (the snapshot digest re-check stays
gated, unchanged); (4) this header. Frozen protocol bytes unchanged: SEED 41001,
corpus recipe and digest pins, 200,000 updates/arm budgets, LR schedule,
evaluation, timeout policy. This file:
jobs/drafts/nl001_substantive_session44_fixed_packaging.py; its own sha256 is
recorded in .neural-lab/notes/nl001-session44-implementation.md.

Campaign: arm A (pure neural answer prediction) vs arm B (neural claim-to-query
translation + premise/edge selection + exact symbolic verification via the
versioned checker), trained on the session-40 whole-program corpus (448 programs,
10–50 nodes, 28,067 claims; generator nl001-sampled-crossed-v1 over
nl001-whole-program-v1, checker nl001-checker-v1). The corpus was generated and
audited OFFLINE (regeneration match, 720,896 cross-split observed-view
isomorphism comparisons, independent Warshall oracle over every claim,
recipe-only identity manifest + literal-form/positional identity audit). This
job REGENERATES the corpus deterministically and fails closed unless the
canonical digest matches the independently retained pin below; it re-runs the
corpus audit (regeneration + observed-view isomorphism) in-job. Offline audit
report digests are embedded as provenance anchors only.

Recommended queue timeout: --timeout-seconds 25200 (7 h; cap 86400).
  Budget basis: calibration job 450069380ee1482f86404864bf567b38 measured
  ~43.7 updates/s (100 updates in 2.287/2.331 s, batch 32, RTX 2070, torch
  2.8.0+cu128). Model input shape is fixed at MAX_NODES=50 padding, so GPU
  compute per update is the same at 10–50 nodes; only host-side batch assembly
  varies slightly with edge count. Planned 200,000 updates/arm:
    - at measured 43.7/s: ~76 min/arm, ~2.6 h both arms (expected case)
    - at a conservative 25/s:   ~133 min/arm, ~4.4 h both arms (slow case)
  Per-arm optimization budget therefore lands in the required 1–4 h range under
  the measured-to-conservative spread. Add corpus regeneration + in-job audit
  (~0.2 h), selection passes (~0.1 h), test evaluation incl. parser/oracle
  controls and replay (~0.5 h), artifact writes (~0.2 h): expected total
  ~3.5–4 h, slow case ~5.5 h; 7 h leaves ~1.5 h headroom. No sleeps, no
  padding: if hardware is faster the job simply finishes earlier.
  Operational semantics: success REQUIRES the completion marker
  nl001-substantive-complete.txt plus result.json plus a complete inventory.
  Timeout, OOM, CUDA absence, digest mismatch, audit failure, or any exception
  means the job FAILED: result.json is removed/absent, a failure marker and
  partial inventory are preserved when catchable, and there is no silent retry
  and no CPU fallback. SIGKILL/timeout leave partial artifacts only.

Frozen protocol (deviations from the 100-update calibration freeze are stated):
  - Same observed facts, same split, same declared budget for both arms:
    200,000 AdamW updates each, batch 32, paired seed 41001. Equal updates do
    not imply equal compute; B has extra output heads (parameters reported).
  - AdamW lr peak 1e-3, betas (0.9, 0.999), eps 1e-8, weight_decay 0.01 —
    unchanged from the freeze. DEVIATION (justified): a learning-rate schedule,
    linear warmup over the first 500 updates then cosine decay to 0.1x peak at
    the final update. The freeze ran 100 updates where any schedule is
    meaningless; at 200,000 updates, decay to a stable minimum is standard and
    keeps the training-CE checkpoint selection meaningful. Schedule is frozen
    here, identical for both arms, and independent of any observed result.
  - Checkpoint selection: BestCheckpoint-style minimum full-training-set cross
    entropy, evaluation mode, observed every 2,000 updates and at the final
    update (101 observations), earliest tie; B's selection CE sums
    property/subject/object CE and excludes witness BCE (frozen semantics via
    nl001_training.full_training_cross_entropy). DEVIATION: observation
    interval 2,000 instead of 100 — at 100-update granularity the 2,000 full
    training passes would cost more than training itself; the selection metric,
    scope and tie rule are unchanged. Test data is never used for selection;
    the test split is first touched in the evaluation phase after both
    checkpoints are frozen. Final (non-selected) weights are saved separately.
  - Edge threshold 0.5 (frozen). Deterministic algorithms enabled;
    CUBLAS_WORKSPACE_CONFIG set before torch import.
  - Controls on the SAME 17,240 test claims (four paired presentations per test
    program, never independent): rule-based parser (nl001_parser) and exact
    symbolic oracle (gold queries). Validation parser/oracle controls run first
    as an integrity gate (any aligned label conflict aborts before test).
  - Evaluation (nl001_evaluation.evaluate for B): answer accuracy with
    abstentions counted, conditional accuracy, false-verification rate with
    numerator/denominator and one-sided Wilson bound, proof coverage,
    Unknown/Supported rates, query semantic accuracy vs held-out ground truth,
    certificate replay success, premise recall vs the gold witness path (one
    deterministic witness, not all sufficient premises — limitation), and
    breakdowns by graph-size bucket, gold proof-chain length, view, and
    presentation. A's outputs are direct predictions, never certifications.
  - Per-claim prediction dumps (incl. B certificates and selected edges) are
    saved for independent replay; exact split tensors via
    ArtifactWriter.save_dataset; checkpoints and datasets linked in
    artifacts.json with generator version, seeds, split sizes, and hashes.

Limitations: package pins below are self-embedded anchors, not independent
custody; the reviewer must re-verify every pin and the corpus/audit digests
against the independently retained session-40 note. Corpus regeneration in-job
uses the same generator/checker as the offline audit. Training-CE selection can
overfit; that is the frozen protocol and generalization is exactly what the
held-out phase measures. Claim-level Wilson bounds ignore program-level
clustering. Dense model is capped at 50 nodes (MAX_NODES).

Smoke mode (--smoke): CPU-only synthetic mini-corpus, 40 updates/arm, no CUDA
assert — exercises the full pipeline shape in minutes. Since the session-44
repair, smoke also snapshots the package and runs both custody checks, so the
packaging path is exercised end-to-end. The queued job runs with no flags and
asserts CUDA.
"""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import argparse
import hashlib
import importlib.machinery
import json
import math
import platform
import sys
import time
import traceback
from pathlib import Path

# Session-40 source pins; the independent reviewer must re-verify these against
# the retained note before enqueue. Filled from the package at draft time.
PACKAGE_SHA256 = {'__init__.py': 'b1a6036b6786d7e6b2e702799084e6a51bd3e7e4e25f8b801466388c8e4d7393', 'artifacts.py': '4f18284d41d34892b1e0128a05f96e0a0f64162f7ff187893455365258f9e1a7', 'nl001_corpus.py': 'd6def6ee8d904a9cb05e4b46ecad7bca4c9296cd73f6d57cc4ab54d4ea8c27d9', 'nl001_corpus_audit.py': '4b128bf8d812ff3feecdb3fc47acb36ba23f43bc83b6ee5fa27d0fccb415da52', 'nl001_data.py': 'eec8c811b9c7e067848370d6cd553680b2c4afacb0ff3c4ccdc23c48513ffc79', 'nl001_evaluation.py': '87124f56b4b80c6bc2868c599a738ecf983df8fd6f936be0e2a4d716eded2c39', 'nl001_inference.py': '4bf53850815cddaca4a741adc4b15b85a731a10fe1701d7b277591779a6b00e6', 'nl001_manifest.py': 'e287302bcf07cbc4d35ae29a792485fee99e7153123046687034b2975873e661', 'nl001_model_input.py': 'de75724ffeed796c3b537f154e107f3b67ea81795c2fb0e33402955cc5d2d44a', 'nl001_models.py': '4f53bb9ce05e4d3055cf318507d65b57fc219b28e2fb968ad57927842765c195', 'nl001_parser.py': '67db4d93728f52957a6df0406be6af989465ce8ee2a57dd32cc59c18d89786d7', 'nl001_run_evaluation.py': '33ef699edd4e933d0dc738011ad4c3298449bb260cd09fb99b2513d48cdb8009', 'nl001_split_audit.py': '91ab7c8003304fd921645b4aae5b6b63d0d30ced1dc18543fd26862eb74b79f6', 'nl001_training.py': '63849e1d70ac9e47c5067e1b818d3304b9c4695a3e371aeddddf2a9b4383bd79', 'nl001_training_data.py': 'b82d03d23656f80b3ac70972a53e48fffd94c365f329e4314891292f0cc02d2b', 'synthetic_graph.py': 'fed818eeffac595415a4e6b2b93edf6fc804f9dcdcf687e918d2a13dc16e2b6b'}

# --- Frozen corpus recipe (session-40 precommit, plan digest
# --- 7587b6faf88b6c451e1f59200147cee84205d4c48f393713f1e194abf256b6e9) ---
SPLIT_SEEDS = {'train': list(range(100000, 100256)),
               'validation': list(range(200000, 200064)),
               'test': list(range(300000, 300128))}
NODE_SIZES = {seed: 10 + (i % 41) for split in ('train', 'validation', 'test')
              for i, seed in enumerate(SPLIT_SEEDS[split])}
GENERATION = {'nodes': NODE_SIZES, 'edge_probability': 0.08, 'missing_fraction': 0.25,
              'per_stratum': 4, 'sampling_seed': 20261106}
EXPECTED_CORPUS_SHA256 = '5cf328feeef975bb026a5a9a3032ea58d53e2ca1bc7330e7bf8b3708fcc6200d'
EXPECTED_MANIFEST_SHA256 = '5dd5b660b7941520486c73cee9b6390690f55e59aad72cd5e89a3d3b0329ae81'
OFFLINE_AUDIT_ANCHORS = {'seed_plan_sha256': '7587b6faf88b6c451e1f59200147cee84205d4c48f393713f1e194abf256b6e9',
    'identity_manifest_sha256': '0b605c10e22d6c9dfa5c4e6fc872a2b300ef71372fdab0e27f5b73aeb4e8fdfb',
    'corpus_audit_sha256': 'b2ab20689cdf0f114e8286ec4c1237d1d04db7044c9a99bac6ae72ff2d63bf7b',
    'semantic_audit_sha256': '52570ca2cef1b65662db414d142c223b931a3e2463c0834cc39696e6751b645c',
    'semantic_attestation_sha256': '6e42ab0720d3e38254080e5970d4fb32d36bd0e69e4042e8a4e1f5c72ce7f8a3',
    'identity_audit_sha256': '862d3c385f4d3aded0880badbf253b74848e0e5d5522b746198090dfe77928ae',
    'corpus_json_file_sha256': 'fff39f9ae75c789d472476367bb1bd3cc5d26733c347ed87dcce4ac69ebb315b',
    'note': 'offline reports retained in .neural-lab/notes/session40-corpus/; anchors are provenance links, not in-job recomputation'}
AUDIT_BUDGETS = {'max_states': 1000000, 'max_comparisons': 1000000, 'max_total_states': 1000000}

# --- Frozen training protocol ---
SEED = 41001          # paired: identical for both arms
BATCH_SIZE = 32
UPDATES = 200_000     # per arm; see header budget basis
LEARNING_RATE = 0.001
WARMUP_UPDATES = 500
MIN_LR_FRACTION = 0.1
OBSERVE_EVERY = 2000  # full-training-CE selection cadence; final update always observed
EDGE_THRESHOLD = 0.5
EXPECTED_UPDATES_PER_SECOND = 43.7   # calibration-measured, batch 32, RTX 2070
CONSERVATIVE_UPDATES_PER_SECOND = 25.0

# --- Smoke-mode miniature (CPU pipeline check only; never queued) ---
SMOKE = {'split_seeds': {'train': [104001, 104002, 104003], 'validation': [204001], 'test': [304001]},
         'generation': {'nodes': 10, 'edge_probability': 0.08, 'missing_fraction': 0.25,
                        'per_stratum': 1, 'sampling_seed': 20261106},
         'updates': 40, 'warmup': 5, 'observe_every': 20}


def write(path, obj):
    path.write_text(json.dumps(obj, sort_keys=True, indent=2, allow_nan=False))


def snapshot_package(destination):
    """Verify bytes before copying, then import only from this execution-local copy.

    The embedded hashes need independent custody and review; they are not a trust
    anchor by themselves. Third-party packages and the interpreter remain external.
    """
    spec = importlib.machinery.PathFinder.find_spec('neural_lab', sys.path)
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError('cannot locate source package')
    source = Path(next(iter(spec.submodule_search_locations)))
    package = destination / 'neural_lab'
    package.mkdir(parents=True, exist_ok=False)
    for name, expected in PACKAGE_SHA256.items():
        data = (source / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise RuntimeError('package digest mismatch: ' + name)
        (package / name).write_bytes(data)
    os.chdir(destination)
    sys.path.insert(0, str(destination))
    os.environ['PYTHONPATH'] = str(destination)
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    return package


def verify_snapshot_custody(package):
    """Fail fast: every packaged module must import, and only from the snapshot.

    Session-44 repair: called immediately after snapshot_package in BOTH smoke
    and non-smoke modes, so a manifest omission or a non-snapshot import path
    aborts before any training compute is spent.
    """
    package = package.resolve()
    for name in PACKAGE_SHA256:
        module = 'neural_lab' if name == '__init__.py' else 'neural_lab.' + name[:-len('.py')]
        importlib.import_module(module)
    for name, module in tuple(sys.modules.items()):
        if name == 'neural_lab' or name.startswith('neural_lab.'):
            path = Path(module.__file__).resolve()
            if path.parent != package or path.name not in PACKAGE_SHA256:
                raise RuntimeError('package imported outside verified snapshot: ' + name)


def inventory(root, exclude=()):
    result = []
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError('symlink in output')
        if path.is_file() and path.relative_to(root).as_posix() not in exclude:
            result.append({'path': path.relative_to(root).as_posix(),
                           'bytes': path.stat().st_size,
                           'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    return result


def lr_at(step, updates, warmup):
    """Frozen schedule: linear warmup, then cosine decay to MIN_LR_FRACTION x peak."""
    if step <= warmup:
        return LEARNING_RATE * step / warmup
    t = (step - warmup) / max(1, updates - warmup)
    return LEARNING_RATE * (MIN_LR_FRACTION + (1 - MIN_LR_FRACTION) * 0.5 * (1 + math.cos(math.pi * t)))


def items_of(programs):
    return [(p, v, q) for p in programs for v in p['variants'] for q in v['questions']]


def train_arm(*, arm, records, device, writer, updates, warmup, observe_every):
    """Paired-budget training with the frozen LR schedule and training-CE selection.

    Selection: minimum full-training-set CE (evaluation mode) observed every
    observe_every updates and at the final update, earliest tie; B's CE sums
    property/subject/object CE and excludes witness BCE. No held-out data here.
    """
    import torch
    from neural_lab.artifacts import BestCheckpoint
    from neural_lab.nl001_models import CalibrationModel, update
    from neural_lab.nl001_training import batch_indices, full_training_cross_entropy, _batch

    torch.manual_seed(SEED)
    if torch.device(device).type == 'cuda':
        torch.cuda.manual_seed_all(SEED)
        torch.cuda.reset_peak_memory_stats(device)
    torch.use_deterministic_algorithms(True)
    hidden = 192 if arm == 'A' else 128
    model = CalibrationModel(arm, hidden=hidden, embedding=32).to(device)
    optimizer_settings = {'lr': LEARNING_RATE, 'betas': (0.9, 0.999), 'eps': 1e-8,
                          'weight_decay': 0.01, 'amsgrad': False}
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_settings)
    schedule = list(batch_indices(len(records), BATCH_SIZE, updates, SEED))
    scope = ('full training-set CE in fixed order, evaluation mode, after every '
             f'{observe_every} updates and the final update; B sums property, subject, '
             'object CE and excludes witness BCE; earliest tie')
    best = BestCheckpoint(scope=scope)
    observations, losses_tail = [], []
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    selection_seconds = 0.0
    for step, indices in enumerate(schedule, 1):
        for group in optimizer.param_groups:
            group['lr'] = lr_at(step, updates, warmup)
        inputs, targets = _batch(records, indices, device)
        losses_tail.append(update(model, optimizer, inputs, targets))
        if len(losses_tail) > 1000:
            losses_tail.pop(0)
        if step % observe_every == 0 or step == updates:
            if torch.device(device).type == 'cuda':
                torch.cuda.synchronize(device)
            mark = time.perf_counter()
            value = full_training_cross_entropy(model, records, BATCH_SIZE, device)
            if torch.device(device).type == 'cuda':
                torch.cuda.synchronize(device)
            selection_seconds += time.perf_counter() - mark
            observations.append({'step': step, 'training_cross_entropy': value})
            best.observe(model, value, step)
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)
    train_seconds = time.perf_counter() - started
    config = {'version': 'nl001-substantive-session40-v1', 'arm': arm, 'seed': SEED,
              'batch_size': BATCH_SIZE, 'updates': updates, 'hidden': hidden, 'embedding': 32,
              'optimizer': 'AdamW', 'optimizer_settings': optimizer_settings,
              'lr_schedule': {'peak': LEARNING_RATE, 'warmup_updates': warmup,
                              'decay': 'cosine', 'min_lr_fraction': MIN_LR_FRACTION},
              'device': str(device), 'corpus_sha256': EXPECTED_CORPUS_SHA256 if updates == UPDATES else 'smoke'}
    datasets = [EXPECTED_CORPUS_SHA256] if updates == UPDATES else ['smoke-corpus']
    best_entry = best.save(writer, 'best-' + arm, arm=arm, seed=SEED, config=config, datasets=datasets)
    final_entry = writer.save_checkpoint('final-' + arm, model.state_dict(), arm=arm, seed=SEED,
        selection={'metric': 'none', 'mode': 'final update weights, NOT selected',
                   'step': updates, 'scope': 'retained for diagnostics only; never selected on any data'},
        config=config, datasets=datasets)
    return {'config': config, 'examples': len(records),
            'parameters': sum(p.numel() for p in model.parameters()),
            'example_exposures': sum(map(len, schedule)),
            'training_seconds': train_seconds, 'selection_seconds': selection_seconds,
            'training_timing_scope': 'batch tensor construction, optimizer updates, selection passes, device synchronization',
            'updates_per_second': updates / train_seconds,
            'expected_updates_per_second': EXPECTED_UPDATES_PER_SECOND,
            'conservative_updates_per_second': CONSERVATIVE_UPDATES_PER_SECOND,
            'selection_observations': observations,
            'best_checkpoint_sha256': best_entry['sha256'], 'best_selection': best_entry['selection'],
            'final_checkpoint_sha256': final_entry['sha256'],
            'final_losses_tail_mean': sum(losses_tail) / len(losses_tail),
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(device) if torch.device(device).type == 'cuda' else None,
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(device) if torch.device(device).type == 'cuda' else None,
            'torch_version': torch.__version__}


def infer_split(*, arm, model, programs, device):
    """Batched inference in program/presentation/question order; B keeps selected edges."""
    import torch
    from neural_lab.nl001_data import model_input
    from neural_lab.nl001_model_input import encode_input
    from neural_lab.nl001_training_data import PROPERTIES, STATUSES, verify_selected

    items = items_of(programs)
    predictions, selected_edges = [], []
    model.eval()
    with torch.inference_mode():
        for offset in range(0, len(items), 256):
            chunk = items[offset:offset + 256]
            raws = [model_input(v, q) for _, v, q in chunk]
            encoded = [encode_input(r) for r in raws]
            inputs = {k: torch.tensor([e[k] for e in encoded], device=device) for k in encoded[0]}
            out = model(inputs)
            if arm == 'A':
                statuses = out['status'].argmax(dim=1).tolist()
                predictions.extend(STATUSES[int(s)] for s in statuses)
                selected_edges.extend([None] * len(chunk))
            else:
                props = out['property'].argmax(dim=1).tolist()
                subs = out['subject'].argmax(dim=1).tolist()
                objs = out['object'].argmax(dim=1).tolist()
                scores = torch.sigmoid(out['edges']).cpu()
                for i, raw in enumerate(raws):
                    symbols = raw['symbols']
                    query = {'property': PROPERTIES[int(props[i])],
                             'subject': symbols[int(subs[i])], 'object': symbols[int(objs[i])]}
                    chosen = [list(e) for e in raw['observed_edges']
                              if scores[i, symbols.index(e[0]), symbols.index(e[1])].item() >= EDGE_THRESHOLD]
                    predictions.append(verify_selected(raw, query, chosen))
                    selected_edges.append(sorted(chosen))
    return items, predictions, selected_edges


def bucket_of(item):
    program, variant, question = item
    size = len(variant['symbols'])
    size_bucket = f"{10 * (size // 10)}-{10 * (size // 10) + 9}" if size < 50 else '50'
    pres = variant['presentation']
    presentation = ('heldout_template+renamed' if pres['heldout_template'] and pres['renamed']
                    else 'heldout_template' if pres['heldout_template']
                    else 'renamed' if pres['renamed'] else 'original')
    cert = question.get('certificate')
    if question['label'] == 'Proven' and isinstance(cert, dict) and cert.get('kind') == 'path':
        length = len(cert['nodes']) - 1
        chain = str(length) if length <= 3 else '4-6' if length <= 6 else '7+'
    else:
        chain = 'no-gold-path'
    return size_bucket, question['view'], presentation, chain


def breakdown(items, flags):
    """Aggregate correct/certified counts over size, view, presentation, chain strata."""
    axes = {}
    for item, flag in zip(items, flags):
        for axis, key in zip(('graph_size', 'view', 'presentation', 'proof_chain'), bucket_of(item)):
            bucket = axes.setdefault(axis, {}).setdefault(key, {'claims': 0, 'correct': 0, 'certified': 0})
            bucket['claims'] += 1
            bucket['correct'] += int(flag[0])
            bucket['certified'] += int(flag[1])
    for axis in axes.values():
        for bucket in axis.values():
            bucket['accuracy'] = bucket['correct'] / bucket['claims']
            bucket['coverage'] = bucket['certified'] / bucket['claims']
    return axes


def premise_dump(items, predictions, selected):
    """Per-claim premise recall against the gold witness path (B only)."""
    rows, recalls = [], []
    for (program, variant, question), prediction, chosen in zip(items, predictions, selected):
        cert = question.get('certificate')
        gold = ([list(edge) for edge in zip(cert['nodes'], cert['nodes'][1:])]
                if question['label'] == 'Proven' and isinstance(cert, dict) and cert.get('kind') == 'path' else None)
        recall = None
        if gold:
            hit = sum(1 for a, b in gold if [a, b] in chosen)
            recall = hit / len(gold)
            recalls.append(recall)
        rows.append({'pair_id': question['pair_id'], 'claim': question['claim'],
                     'expected_query': question['query'], 'label': question['label'],
                     'prediction': prediction, 'selected_edges': chosen,
                     'gold_witness_edges': gold, 'premise_recall': recall})
    return rows, (sum(recalls) / len(recalls) if recalls else None), len(recalls)


def run_pipeline(*, output, smoke):
    started_all = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    snapshot = snapshot_package(output / 'nl001-source-snapshot')
    verify_snapshot_custody(snapshot)
    import torch
    from neural_lab.artifacts import ArtifactWriter
    from neural_lab.nl001_corpus import make_sampled_corpus
    from neural_lab.nl001_data import digest, model_input
    from neural_lab.nl001_evaluation import evaluate
    from neural_lab.nl001_manifest import make_manifest, authenticate_split
    from neural_lab.nl001_run_evaluation import evaluate_split
    from neural_lab.nl001_training import _flatten, _batch
    from neural_lab.nl001_training_data import example
    from neural_lab.synthetic_graph import Graph

    device = 'cpu'
    if not smoke:
        if not torch.cuda.is_available() or '2070' not in torch.cuda.get_device_name(0):
            raise RuntimeError('substantive campaign requires the target RTX 2070 CUDA device; no fallback')
        device = 'cuda:0'
    work = output / 'nl001-substantive'
    work.mkdir(exist_ok=False)

    # Phase 1: deterministic corpus regeneration pinned to the offline-audited digest.
    mark = time.perf_counter()
    if smoke:
        corpus = make_sampled_corpus(SMOKE['split_seeds'], **SMOKE['generation'])
        audit = None
    else:
        corpus = make_sampled_corpus(SPLIT_SEEDS, **GENERATION)
        if corpus['sha256'] != EXPECTED_CORPUS_SHA256:
            raise RuntimeError('regenerated corpus does not match the independently retained digest')
        from neural_lab.nl001_corpus_audit import audit_corpus
        audit = audit_corpus(corpus, EXPECTED_CORPUS_SHA256, **AUDIT_BUDGETS)
        if audit['manifest_sha256'] != EXPECTED_MANIFEST_SHA256:
            raise RuntimeError('split manifest does not match the independently retained digest')
    manifest = make_manifest(corpus)
    corpus_seconds = time.perf_counter() - mark
    write(work / 'corpus.json', corpus)
    write(work / 'manifest.json', manifest)
    writer = ArtifactWriter(output)
    for name in ('corpus.json', 'manifest.json'):
        writer.register_file('datasets', work / name, name, format='json')
    if audit is not None:
        write(work / 'corpus-audit.json', audit)
        writer.register_file('datasets', work / 'corpus-audit.json', 'corpus-audit', format='json',
                             canonical_sha256=digest(audit))
    questions = {split: sum(len(v['questions']) for p in corpus['splits'][split] for v in p['variants'])
                 for split in ('train', 'validation', 'test')}

    # Phase 2: exact split tensors (targets for validation/test are archival only).
    split_records = {}
    for split in ('train', 'validation', 'test'):
        programs = corpus['splits'][split]
        records = [example(v, q) for p in programs for v in p['variants'] for q in v['questions']]
        split_records[split] = records
        inputs, targets = _batch(records, range(len(records)), 'cpu')
        writer.save_dataset('substantive-' + split + '-tensors',
            {**{'input_' + k: v for k, v in inputs.items()},
             **{'target_' + k: v for k, v in targets.items()}},
            provenance={'split': split, 'split_canonical_sha256': digest(programs),
                        'corpus_sha256': corpus['sha256'], 'questions': questions[split],
                        'order': 'program, presentation, question in retained corpus JSON',
                        'scope': 'targets for validation/test are archival; test is untouched until phase 5'})

    # Phase 3: paired training (selection never sees validation or test).
    train_records = _flatten(corpus['splits']['train'])
    reports = {}
    for arm in ('A', 'B'):
        reports[arm] = train_arm(arm=arm, records=train_records, device=device, writer=writer,
            updates=SMOKE['updates'] if smoke else UPDATES,
            warmup=SMOKE['warmup'] if smoke else WARMUP_UPDATES,
            observe_every=SMOKE['observe_every'] if smoke else OBSERVE_EVERY)
        write(work / f'training-report-{arm}.json', reports[arm])
        writer.register_file('datasets', work / f'training-report-{arm}.json',
                             'training-report-' + arm, format='json')

    # Phase 4: validation parser/oracle controls as an integrity gate (no fitting).
    for mode in ('parser', 'oracle'):
        report = evaluate_split(programs=corpus['splits']['validation'], split='validation',
            manifest=manifest, manifest_sha256=digest(manifest),
            output=work / 'controls' / ('validation-' + mode), mode=mode)
        if report['counts']['aligned_label_conflicts']:
            raise RuntimeError('validation control found aligned label conflict; aborting before test')

    # Phase 5: held-out test evaluation, once, after both checkpoints are frozen.
    from neural_lab.nl001_inference import _checkpoint_model
    test_programs = corpus['splits']['test']
    manifest_sha256 = digest(manifest)
    authenticate_split(test_programs, 'test', manifest, manifest_sha256)
    evaluation = {'controls': {}, 'arms': {}}
    for mode in ('parser', 'oracle'):
        evaluation['controls'][mode] = evaluate_split(programs=test_programs, split='test',
            manifest=manifest, manifest_sha256=manifest_sha256,
            output=work / 'controls' / ('test-' + mode), mode=mode)
    metrics = {'corpus_regeneration_audit_seconds': corpus_seconds,
               'questions_train': questions['train'], 'questions_validation': questions['validation'],
               'questions_test': questions['test']}
    for arm in ('A', 'B'):
        checkpoint = output / 'checkpoints' / ('best-' + arm + '.pt')
        model = _checkpoint_model(checkpoint, reports[arm]['best_checkpoint_sha256'], arm).to(device)
        mark = time.perf_counter()
        items, predictions, selected = infer_split(arm=arm, model=model, programs=test_programs, device=device)
        inference_seconds = time.perf_counter() - mark
        mode = 'direct' if arm == 'A' else 'checked'
        report = evaluate_split(programs=test_programs, split='test', manifest=manifest,
            manifest_sha256=manifest_sha256, output=work / 'eval' / ('test-' + arm),
            mode=mode, predictions=predictions)
        if arm == 'A':
            flags = [(isinstance(p, str) and p in ('Proven', 'Refuted') and p == q['label'], False)
                     for p, (_, _, q) in zip(predictions, items)]
            extra = {'breakdowns': breakdown(items, flags)}
        else:
            details = report['details']
            flags = [(d['correct'], d['status'] in ('Proven', 'Refuted') and d['query_aligned'])
                     for d in details]
            rows, mean_recall, recall_count = premise_dump(items, predictions, selected)
            write(work / 'B-test-premise-dump.json',
                  {'premise_recall_mean': mean_recall, 'premise_recall_claims': recall_count,
                   'limitation': 'gold witness is one deterministic checker path, not all sufficient premises',
                   'rows': rows})
            writer.register_file('datasets', work / 'B-test-premise-dump.json',
                                 'B-test-premise-dump', format='json')
            extra = {'breakdowns': breakdown(items, flags),
                     'premise_recall_mean': mean_recall, 'premise_recall_claims': recall_count}
        evaluation['arms'][arm] = {'mode': mode, 'inference_seconds': inference_seconds,
                                   'report': {k: v for k, v in report.items() if k != 'details'}, **extra}
        prefix = arm.lower() + '_'
        metrics[prefix + 'training_seconds'] = reports[arm]['training_seconds']
        metrics[prefix + 'updates_per_second'] = reports[arm]['updates_per_second']
        metrics[prefix + 'peak_allocated_bytes'] = reports[arm]['peak_allocated_bytes']
        metrics[prefix + 'parameters'] = reports[arm]['parameters']
        metrics[prefix + 'best_step'] = reports[arm]['best_selection']['step']
        metrics[prefix + 'best_training_ce'] = reports[arm]['best_selection']['value']
        metrics[prefix + 'test_inference_seconds'] = inference_seconds
        if arm == 'A':
            metrics['a_test_answer_accuracy'] = report['answer_accuracy']
            metrics['a_test_conditional_accuracy'] = report['conditional_answer_accuracy']
        else:
            metrics['b_test_answer_accuracy'] = report['answer_accuracy']
            metrics['b_test_conditional_accuracy'] = report['conditional_answer_accuracy']
            metrics['b_test_query_semantic_accuracy'] = report['query_semantic_accuracy']
            metrics['b_test_proof_coverage'] = report['proof_coverage']
            metrics['b_test_unknown_rate'] = report['unknown_rate']
            metrics['b_test_supported_rate'] = report['supported_rate']
            metrics['b_test_replay_rate'] = report['replay_rate']
            metrics['b_test_false_verification_numerator'] = report['counts']['false_proven']
            metrics['b_test_false_verification_denominator'] = report['counts']['proven']
            metrics['b_test_false_verification_rate'] = report['false_verification_rate']
            metrics['b_test_false_verification_upper_bound'] = report['false_verification_upper_bound']
            metrics['b_test_premise_recall_mean'] = extra['premise_recall_mean']
    for mode in ('parser', 'oracle'):
        control = evaluation['controls'][mode]
        prefix = 'control_' + mode + '_'
        metrics[prefix + 'test_answer_accuracy'] = control['answer_accuracy']
        metrics[prefix + 'test_proof_coverage'] = control['proof_coverage']
        metrics[prefix + 'test_false_verification_numerator'] = control['counts']['false_proven']
        metrics[prefix + 'test_false_verification_denominator'] = control['counts']['proven']

    write(work / 'evaluation.json', evaluation)
    writer.register_file('datasets', work / 'evaluation.json', 'evaluation', format='json')
    write(work / 'environment.json', {'python': sys.version, 'platform': platform.platform(),
        'torch': torch.__version__, 'cuda_runtime': torch.version.cuda, 'smoke': smoke,
        'gpu': torch.cuda.get_device_name(0) if not smoke else None,
        'device': device, 'offline_audit_anchors': OFFLINE_AUDIT_ANCHORS if not smoke else None,
        'corpus_sha256': corpus['sha256'], 'manifest_sha256': manifest_sha256,
        'split_questions': questions, 'generator_version': corpus['version'],
        'base_generator_version': corpus['base_version'], 'checker_version': corpus['checker_version'],
        'split_seeds': corpus['config']['split_seeds'], 'package_sha256': PACKAGE_SHA256 if not smoke else None})
    writer.register_file('datasets', work / 'environment.json', 'environment', format='json')
    metrics['wall_seconds'] = time.perf_counter() - started_all
    write(output / 'result.json', {'metrics': metrics, 'summary':
        ('SMOKE pipeline check only; no scientific content.' if smoke else
         'NL-001 substantive paired A/B on the session-40 audited corpus; see '
         'nl001-substantive/evaluation.json, controls/, eval/, training reports and artifacts.json. '
         'A outputs are predictions, never certifications. B false-verification counts include '
         'misaligned-query proofs. Claim-level bounds ignore program-level clustering.')})
    # Detect unexpected package imports before success; since the session-44
    # repair the sys.modules custody check runs in smoke mode too. The snapshot
    # digest re-check stays gated on 'if not smoke', unchanged.
    snapshot = output / 'nl001-source-snapshot' / 'neural_lab'
    if not smoke:
        for name, expected in PACKAGE_SHA256.items():
            if hashlib.sha256((snapshot / name).read_bytes()).hexdigest() != expected:
                raise RuntimeError('snapshot digest changed: ' + name)
    for name, module in tuple(sys.modules.items()):
        if name == 'neural_lab' or name.startswith('neural_lab.'):
            path = Path(module.__file__).resolve()
            if path.parent != snapshot.resolve() or path.name not in PACKAGE_SHA256:
                raise RuntimeError('package imported outside verified snapshot: ' + name)
    write(work / 'inventory.json', inventory(work, exclude={'inventory.json'}))
    write(output / 'output-inventory.json', inventory(output, exclude={'output-inventory.json'}))
    # A result without this last marker may be a killed or interrupted run.
    (output / 'nl001-substantive-complete.txt').write_text(
        hashlib.sha256((output / 'output-inventory.json').read_bytes()).hexdigest() + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true', help='CPU synthetic mini-pipeline; never for queue')
    parser.add_argument('--output', type=Path, default=Path('/output'))
    args = parser.parse_args()
    try:
        run_pipeline(output=args.output, smoke=args.smoke)
    except Exception:
        try:
            args.output.mkdir(parents=True, exist_ok=True)
            (args.output / 'result.json').unlink(missing_ok=True)
            (args.output / 'nl001-substantive-failure.txt').write_text(traceback.format_exc())
            write(args.output / 'failure-inventory.json',
                  inventory(args.output, exclude={'failure-inventory.json'}))
        except Exception:
            traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
