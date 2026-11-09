#!/usr/bin/env python3
"""NL-001 substantive post-exit validation package (CPU-only, stdlib only).

Independent validator for one or more NL-001 substantive seed output dirs
(seeds 41001-41004 campaign). For each output dir it checks:

1. Output-inventory contract, reusing scripts/validate_nl001_output.py
   ``validate()`` verbatim (loaded by explicit path; only the module-level
   MARKER constant is adapted to the substantive completion marker).
2. Independent replay of the saved per-claim prediction dumps
   (nl001-substantive/eval/* and nl001-substantive/controls/*) through a
   from-scratch reimplementation of the frozen checker semantics
   (nl001-checker-v1, neural_lab/synthetic_graph.py). No job code is
   imported; certificate replay and gold-label recomputation are
   reimplemented here. Replay covers ALL claims when the dump is under
   --sample-cap, else a deterministic seeded sample.
3. Honest outcome counts recomputed from the per-claim dump per arm:
   totals, Proven/Refuted/Supported/Unknown, certified-Proven, false-
   verification numerator/denominator, one-sided Wilson upper bound; these
   are compared against the job's own evaluation.json.
4. Checkpoint selection integrity: BestCheckpoint metric/step recomputed
   from the recorded training curve (earliest tie), artifacts.json
   cross-references, and .pt embedded metadata read with a restricted
   unpickler (no torch, no code execution). Final weights must be stored
   separately from best weights.
5. Corpus/manifest digest consistency (canonical-JSON SHA-256 recompute).

Outputs a machine-readable JSON report per seed plus a combined summary,
and a human summary on stdout. Exit codes:
  0 = every seed passed; 1 = at least one check failed;
  2 = no failures, but at least one seed had no eval artifacts
  (e.g. failed packaging runs) and was marked not applicable.

No queue interaction, no network, no GPU, no randomness without a fixed
seed.
"""
import argparse
import hashlib
import importlib.util
import io
import json
import math
import pickle
import random
import re
import statistics
import sys
import zipfile
from collections import deque
from pathlib import Path

CHECKER_VERSION = 'nl001-checker-v1'
LABELS = frozenset({'Proven', 'Refuted', 'Unknown', 'Supported'})
COUNT_KEYS = ('claims', 'correct_answers', 'answered', 'aligned_queries', 'proven',
              'refuted', 'unknown', 'supported', 'replay_attempts', 'replay_passed',
              'replay_rejected', 'false_proven', 'claim_certified',
              'reported_supported', 'aligned_label_conflicts')
RATE_KEYS = ('answer_accuracy', 'conditional_answer_accuracy', 'query_semantic_accuracy',
             'proof_coverage', 'unknown_rate', 'supported_rate', 'replay_rate',
             'false_verification_rate', 'false_verification_upper_bound')
FLOAT_TOL = dict(rel_tol=1e-9, abs_tol=1e-12)
SUBSTANTIVE_MARKER = 'nl001-substantive-complete.txt'
CALIBRATION_MARKER = 'nl001-calibration-complete.txt'


# --------------------------------------------------------------------------
# Generic helpers

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def load_json(path):
    with open(path, encoding='utf-8') as stream:
        return json.load(stream)


class Check:
    def __init__(self, name, status, detail=''):
        self.name, self.status, self.detail = name, status, detail

    def as_dict(self):
        return {'name': self.name, 'status': self.status, 'detail': self.detail}


# --------------------------------------------------------------------------
# Frozen checker semantics, reimplemented from scratch (nl001-checker-v1).
# Source of truth: neural_lab/synthetic_graph.py @ sha256 fed818ee... (this
# module never imports it; the digests are reported for comparison).

def valid_query(symbols, query):
    return (isinstance(query, dict) and set(query) == {'property', 'subject', 'object'}
            and all(isinstance(query[k], str) for k in query)
            and query['property'] in {'calls', 'reachable'}
            and query['subject'] in symbols and query['object'] in symbols)


def replay_certificate(symbols, edges, complete, query, result, certificate):
    """Validate a certificate against the full observed graph (frozen replay())."""
    if not valid_query(symbols, query) or not isinstance(certificate, dict):
        return False
    a, b, prop = query['subject'], query['object'], query['property']
    kind = certificate.get('kind')
    if result == 'Proven' and kind == 'path' and set(certificate) == {'kind', 'nodes'}:
        path = certificate['nodes']
        return (isinstance(path, list) and len(path) >= 2 and path[0] == a and path[-1] == b
                and (prop != 'calls' or len(path) == 2)
                and all(isinstance(x, str) and x in symbols for x in path)
                and all((x, y) in edges for x, y in zip(path, path[1:])))
    if result != 'Refuted' or not complete:
        return False
    if prop == 'calls' and kind == 'absent_edge' and set(certificate) == {'kind', 'edge'}:
        return certificate['edge'] == [a, b] and (a, b) not in edges
    if prop == 'reachable' and kind == 'closed_set' and set(certificate) == {'kind', 'nodes'}:
        nodes = certificate['nodes']
        if not isinstance(nodes, list) or any(not isinstance(x, str) or x not in symbols for x in nodes):
            return False
        closure = set(nodes)
        if len(closure) != len(nodes) or a not in closure:
            return False
        return all(y in closure and y != b for x, y in edges if x in closure)
    return False


def checker_solve(symbols, edges, complete, query):
    """Reimplementation of frozen check() for an already-aligned query.

    Returns (result, certificate) exactly as _check_query would with
    expected_query == query: deterministic BFS over sorted adjacency.
    """
    if not valid_query(symbols, query):
        return 'Unknown', None
    a, b, prop = query['subject'], query['object'], query['property']
    if prop == 'calls':
        if (a, b) in edges:
            return 'Proven', {'kind': 'path', 'nodes': [a, b]}
        if complete:
            return 'Refuted', {'kind': 'absent_edge', 'edge': [a, b]}
        return 'Unknown', None
    adj = {s: [] for s in symbols}
    for x, y in sorted(edges):
        adj[x].append(y)
    queue = deque([(a, [a])])
    seen = {a}
    while queue:
        x, path = queue.popleft()
        for y in adj[x]:
            if y == b:
                return 'Proven', {'kind': 'path', 'nodes': path + [y]}
            if y not in seen:
                seen.add(y)
                queue.append((y, path + [y]))
    if complete:
        return 'Refuted', {'kind': 'closed_set', 'nodes': sorted(seen)}
    return 'Unknown', None


def upper_wilson(errors, total, confidence=0.95):
    """One-sided Wilson score upper bound (matches nl001_evaluation._upper_wilson)."""
    if not total:
        return None
    z = statistics.NormalDist().inv_cdf(confidence)
    p = errors / total
    return min(1.0, (p + z * z / (2 * total) + z * math.sqrt(
        p * (1 - p) / total + z * z / (4 * total * total))) / (1 + z * z / total))


# --------------------------------------------------------------------------
# Restricted .pt metadata reader (zip + pickle, no torch, no code execution).

def _rebuild_tensor_stub(*args):
    # (storage, storage_offset, size, stride, ...) -> plain metadata tuple
    return ('tensor', args[1], tuple(args[2]), tuple(args[3]))


_PT_GLOBALS = {
    ('collections', 'OrderedDict'): dict,
    ('torch._utils', '_rebuild_tensor_v2'): _rebuild_tensor_stub,
    ('torch._utils', '_rebuild_tensor'): _rebuild_tensor_stub,
    ('torch._utils', '_rebuild_parameter'): lambda data, *a: data,
}


class _PtUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in _PT_GLOBALS:
            return _PT_GLOBALS[(module, name)]
        if module == 'torch' and re.fullmatch(r'(U?Int|Float|Double|Half|BFloat16|Bool|Byte|Char|Short|Long)\d*Storage', name):
            return name  # storage classes appear only as persistent-id type names
        raise pickle.UnpicklingError(f'blocked global {module}.{name}')

    def persistent_load(self, pid):
        if isinstance(pid, tuple) and pid and pid[0] == 'storage':
            return ('storage', str(pid[1]), str(pid[2]), str(pid[3]), int(pid[4]))
        raise pickle.UnpicklingError(f'unexpected persistent id: {pid!r}')


def read_pt_metadata(path):
    """Return (metadata, tensor_names) from a torch .pt without executing code."""
    with zipfile.ZipFile(path) as archive:
        entries = [n for n in archive.namelist() if n.endswith('.pkl')]
        if len(entries) != 1:
            raise ValueError(f'{path}: expected exactly one .pkl entry, found {len(entries)}')
        payload = _PtUnpickler(io.BytesIO(archive.read(entries[0]))).load()
    if not isinstance(payload, dict) or payload.get('format') != 'neural-lab-state-dict-pt-v1':
        raise ValueError(f'{path}: unsupported checkpoint format')
    tensors = payload.get('tensors')
    if not isinstance(tensors, dict):
        raise ValueError(f'{path}: tensors must be a mapping')
    metadata = {k: v for k, v in payload.items() if k != 'tensors'}
    return metadata, sorted(tensors)


# --------------------------------------------------------------------------
# Inventory contract (reuse scripts/validate_nl001_output.py by explicit path)

def run_inventory_contract(root, validator_path):
    spec = importlib.util.spec_from_file_location('nl001_inventory_validator', validator_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    marker = SUBSTANTIVE_MARKER if (root / SUBSTANTIVE_MARKER).is_file() else CALIBRATION_MARKER
    original = module.MARKER
    module.MARKER = marker
    try:
        outcome = module.validate(root)
    finally:
        module.MARKER = original
    return {'marker': marker, **outcome}


# --------------------------------------------------------------------------
# Per-claim dump replay

def build_split_index(programs):
    """Index split.json variants by (source, canonical presentation) and questions."""
    variants = {}
    for program in programs:
        for variant in program.get('variants', []):
            key = (variant['source'], canonical(variant['presentation']))
            if key in variants:
                raise ValueError('duplicate variant in split payload: ' + variant['source'])
            questions = {}
            for question in variant['questions']:
                qkey = (question['pair_id'], question['view'])
                if qkey in questions:
                    raise ValueError('duplicate question in variant: ' + variant['source'])
                questions[qkey] = question
            variants[key] = (variant, questions)
    return variants


def graph_of(variant, view):
    symbols = tuple(variant['symbols'])
    raw = variant['full_edges'] if view == 'complete' else variant['observed_incomplete_edges']
    return set(symbols), {(str(a), str(b)) for a, b in (tuple(e) for e in raw)}, view == 'complete'


def select_sample(n, fraction, cap, seed):
    """Deterministic claim sample: ALL claims if n <= cap or fraction >= 1,
    else a seeded deterministic sample of round(n * fraction) claims."""
    if n <= cap or fraction >= 1.0:
        return list(range(n)), 'all'
    k = max(1, min(n, round(n * fraction)))
    rng = random.Random(seed)
    return sorted(rng.sample(range(n), k)), f'seeded-sample k={k}/{n} seed={seed}'


def replay_eval_dir(eval_dir, *, fraction, cap, seed, confidence):
    """Replay one per-claim dump dir (predictions.json + split.json + evaluation.json)."""
    eval_dir = Path(eval_dir)
    records = load_json(eval_dir / 'predictions.json')
    programs = load_json(eval_dir / 'split.json')
    reported = load_json(eval_dir / 'evaluation.json')
    index = build_split_index(programs)
    if not isinstance(records, list):
        raise ValueError('predictions.json must be a list')

    indices, sample_desc = select_sample(len(records), fraction, cap, seed)
    sample_digest = digest(indices)

    mode = None
    counts = {k: 0 for k in COUNT_KEYS}
    raw_outcomes = {label: 0 for label in sorted(LABELS)}
    raw_other = 0
    gold_mismatches, record_errors, rejected_samples = [], [], []

    for position in indices:
        record = records[position]
        try:
            if not isinstance(record, dict):
                raise ValueError('record is not an object')
            key = (record['source'], canonical(record['presentation']))
            variant, questions = index[key]
            question = questions[(record['pair_id'], record['view'])]
            if question['query'] != record['expected_query']:
                raise ValueError('expected_query does not match split payload')
            if question['claim'] != record['claim']:
                raise ValueError('claim text does not match split payload')
            symbols, edges, complete = graph_of(variant, record['view'])
            label = record['label']
            if label not in LABELS or question['label'] != label:
                raise ValueError('label missing or inconsistent with split payload')
            # Independent gold recomputation (frozen checker semantics).
            gold_result, gold_certificate = checker_solve(symbols, edges, complete,
                                                          record['expected_query'])
            if gold_result != label or gold_certificate != question.get('certificate'):
                gold_mismatches.append({
                    'index': position, 'pair_id': record['pair_id'],
                    'dump_label': label, 'recomputed': gold_result,
                    'certificate_match': gold_certificate == question.get('certificate')})
            prediction = record.get('prediction')
            if isinstance(prediction, str):
                this_mode = 'direct'
                if prediction in raw_outcomes:
                    raw_outcomes[prediction] += 1
                else:
                    raw_other += 1
                counts['claims'] += 1
                correct = prediction in ('Proven', 'Refuted') and prediction == label
                counts['correct_answers'] += int(correct)
                counts['answered'] += int(prediction in ('Proven', 'Refuted'))
                counts['aligned_queries'] += 1  # not meaningful for direct; not compared
            elif isinstance(prediction, dict) or prediction is None:
                this_mode = 'checked'
                prediction = prediction if isinstance(prediction, dict) else {}
                query, status = prediction.get('query'), prediction.get('result')
                if status in raw_outcomes:
                    raw_outcomes[status] += 1
                else:
                    raw_other += 1
                aligned = (isinstance(query, dict)
                           and set(query) == {'property', 'subject', 'object'}
                           and all(isinstance(query[k], str) and query[k] for k in query)
                           and query == record['expected_query'])
                counts['claims'] += 1
                counts['aligned_queries'] += int(aligned)
                replayed = False
                if status in ('Proven', 'Refuted'):
                    counts['replay_attempts'] += 1
                    try:
                        replayed = bool(replay_certificate(
                            symbols, edges, complete, query, status,
                            prediction.get('certificate')))
                    except (TypeError, ValueError, KeyError):
                        replayed = False
                    counts['replay_passed' if replayed else 'replay_rejected'] += 1
                    if not replayed:
                        if len(rejected_samples) < 20:
                            rejected_samples.append({'index': position,
                                                     'pair_id': record['pair_id'],
                                                     'result': status})
                        status = 'Unknown'
                elif status == 'Supported':
                    counts['reported_supported'] += 1
                    status = 'Unknown'
                elif status != 'Unknown':
                    status = 'Unknown'
                counts[status.lower()] += 1
                if status == 'Proven' and (not aligned or label != 'Proven'):
                    counts['false_proven'] += 1
                correct = status in ('Proven', 'Refuted') and aligned and status == label
                counts['correct_answers'] += int(correct)
                counts['answered'] += int(status in ('Proven', 'Refuted'))
                counts['claim_certified'] += int(status in ('Proven', 'Refuted') and aligned)
                counts['aligned_label_conflicts'] += int(
                    status in ('Proven', 'Refuted') and aligned and status != label)
            else:
                raise ValueError('prediction has unexpected type')
            if mode is None:
                mode = this_mode
            elif mode != this_mode:
                raise ValueError('mixed direct/checked predictions in one dump')
        except (KeyError, TypeError, ValueError) as exc:
            record_errors.append({'index': position, 'error': str(exc)})
            if len(record_errors) > 50:
                raise ValueError('too many malformed records; aborting dump replay')

    n = counts['claims']

    def rate(a, b):
        return a / b if b else None

    recomputed = {
        'counts': counts,
        'answer_accuracy': rate(counts['correct_answers'], n),
        'conditional_answer_accuracy': rate(counts['correct_answers'], counts['answered']),
        'false_verification_numerator': counts['false_proven'],
        'false_verification_denominator': counts['proven'],
        'false_verification_rate': rate(counts['false_proven'], counts['proven']),
        'false_verification_upper_bound': upper_wilson(counts['false_proven'],
                                                       counts['proven'], confidence),
    }
    if mode == 'checked':
        recomputed.update({
            'query_semantic_accuracy': rate(counts['aligned_queries'], n),
            'proof_coverage': rate(counts['claim_certified'], n),
            'unknown_rate': rate(counts['unknown'], n),
            'supported_rate': rate(counts['supported'], n),
            'replay_rate': rate(counts['replay_passed'], counts['replay_attempts']),
        })

    # Comparison against the job's own report (exact for ints, tolerant for floats).
    comparison = {'status': 'skipped', 'diffs': []}
    full_replay = sample_desc == 'all'
    if full_replay and isinstance(reported, dict):
        diffs = []
        if mode == 'checked':
            for key_name in COUNT_KEYS:
                if reported.get('counts', {}).get(key_name) != counts[key_name]:
                    diffs.append(f'counts.{key_name}: job={reported.get("counts", {}).get(key_name)} replay={counts[key_name]}')
            for key_name in RATE_KEYS:
                job_value = reported.get(key_name)
                my_value = recomputed.get(key_name)
                if job_value is None or my_value is None:
                    if job_value is not my_value:
                        diffs.append(f'{key_name}: job={job_value} replay={my_value}')
                elif not math.isclose(job_value, my_value, **FLOAT_TOL):
                    diffs.append(f'{key_name}: job={job_value} replay={my_value}')
        elif mode == 'direct':
            for key_name in ('claims', 'correct_answers', 'answered'):
                if reported.get(key_name) != counts[key_name]:
                    diffs.append(f'{key_name}: job={reported.get(key_name)} replay={counts[key_name]}')
            for key_name in ('answer_accuracy', 'conditional_answer_accuracy'):
                job_value, my_value = reported.get(key_name), recomputed.get(key_name)
                if job_value is None or my_value is None:
                    if job_value is not my_value:
                        diffs.append(f'{key_name}: job={job_value} replay={my_value}')
                elif not math.isclose(job_value, my_value, **FLOAT_TOL):
                    diffs.append(f'{key_name}: job={job_value} replay={my_value}')
        comparison = {'status': 'match' if not diffs else 'MISMATCH', 'diffs': diffs[:40]}

    return {
        'mode': mode, 'records_total': len(records), 'records_replayed': n,
        'sample': sample_desc, 'sample_indices_sha256': sample_digest,
        'raw_prediction_outcomes': {**raw_outcomes, 'other': raw_other},
        'record_errors': record_errors,
        'gold_recheck_mismatches': gold_mismatches[:50],
        'gold_recheck_mismatch_count': len(gold_mismatches),
        'replay_rejected_examples': rejected_samples,
        'recomputed': recomputed,
        'job_report_comparison': comparison,
    }


# --------------------------------------------------------------------------
# Checkpoint selection integrity

def check_checkpoints(root, checks, details):
    artifacts_path = root / 'artifacts.json'
    if not artifacts_path.is_file():
        checks.append(Check('checkpoints.artifacts_manifest', 'fail', 'artifacts.json missing'))
        return
    artifacts = load_json(artifacts_path)
    entries = {e['label']: e for e in artifacts.get('checkpoints', [])}
    arm_reports = {}
    for arm in ('A', 'B'):
        report_path = root / 'nl001-substantive' / f'training-report-{arm}.json'
        if not report_path.is_file():
            checks.append(Check(f'checkpoints.arm_{arm}', 'fail',
                                f'nl001-substantive/training-report-{arm}.json missing'))
            continue
        report = load_json(report_path)
        arm_reports[arm] = report
        observations = report.get('selection_observations')
        selection = report.get('best_selection')
        problems = []
        if not isinstance(observations, list) or not observations:
            problems.append('selection_observations missing or empty')
        else:
            best_value, best_step = None, None
            for obs in observations:
                value, step = obs['training_cross_entropy'], obs['step']
                if best_value is None or value < best_value:  # strict: earliest tie wins
                    best_value, best_step = value, step
            if not isinstance(selection, dict):
                problems.append('best_selection missing')
            else:
                if selection.get('metric') != 'training_cross_entropy' or selection.get('mode') != 'min':
                    problems.append('best_selection metric/mode unexpected: '
                                    f"{selection.get('metric')}/{selection.get('mode')}")
                if selection.get('step') != best_step:
                    problems.append(f"best step {selection.get('step')} != curve minimum step {best_step}")
                if selection.get('value') != best_value:
                    problems.append(f"best value {selection.get('value')} != curve minimum {best_value}")
                if selection.get('observations') != len(observations):
                    problems.append('observation count mismatch')
                if selection.get('observed_steps') != [o['step'] for o in observations]:
                    problems.append('observed_steps mismatch')
        details[f'arm_{arm}_selection'] = {
            'observations': len(observations) if isinstance(observations, list) else None,
            'recomputed_min_step': best_step if isinstance(observations, list) and observations else None,
            'recomputed_min_value': best_value if isinstance(observations, list) and observations else None,
            'recorded_selection': selection,
        }
        checks.append(Check(f'checkpoints.arm_{arm}.selection_curve', 'fail' if problems else 'pass',
                            '; '.join(problems) if problems else
                            f'selection matches training-curve minimum at step {best_step}'))

        # File-level: best and final checkpoints exist, hash-match artifacts.json,
        # and final weights are stored separately from best weights.
        for kind in ('best', 'final'):
            label = f'{kind}-{arm}'
            entry = entries.get(label)
            path = root / 'checkpoints' / f'{label}.pt'
            file_problems = []
            if entry is None:
                file_problems.append(f'no artifacts.json checkpoint entry {label!r}')
            if not path.is_file():
                file_problems.append(f'{path.relative_to(root)} missing')
            if not file_problems:
                actual = sha256_file(path)
                if entry.get('sha256') != actual:
                    file_problems.append('artifacts.json sha256 mismatch')
                report_key = f'{kind}_checkpoint_sha256'
                if report.get(report_key) != actual:
                    file_problems.append(f'training report {report_key} mismatch')
                try:
                    metadata, tensor_names = read_pt_metadata(path)
                    schema = entry.get('tensors', {})
                    if sorted(schema) != tensor_names:
                        file_problems.append('embedded tensor names differ from artifacts.json schema')
                    if metadata.get('arm') != arm:
                        file_problems.append(f"embedded arm {metadata.get('arm')!r} != {arm!r}")
                    embedded = metadata.get('selection')
                    if kind == 'best':
                        if embedded != selection:
                            file_problems.append('embedded selection != training-report best_selection')
                    else:
                        if not isinstance(embedded, dict) or embedded.get('metric') != 'none':
                            file_problems.append('final checkpoint selection metric is not "none"')
                    details[f'arm_{arm}_{kind}_checkpoint'] = {
                        'sha256': actual, 'tensors': len(tensor_names),
                        'embedded_selection': embedded,
                    }
                except (ValueError, pickle.UnpicklingError, zipfile.BadZipFile, KeyError) as exc:
                    file_problems.append(f'checkpoint metadata unreadable: {exc}')
            checks.append(Check(f'checkpoints.arm_{arm}.{kind}_file',
                                'fail' if file_problems else 'pass',
                                '; '.join(file_problems) if file_problems else 'ok'))
        both = details.get(f'arm_{arm}_best_checkpoint'), details.get(f'arm_{arm}_final_checkpoint')
        if both[0] and both[1]:
            same = both[0]['sha256'] == both[1]['sha256']
            details[f'arm_{arm}_best_equals_final_payload'] = same
            checks.append(Check(f'checkpoints.arm_{arm}.separate_storage', 'pass',
                                'best and final stored as separate artifacts'
                                + (' (payloads identical: minimum occurred at final update)' if same
                                   else ' (payloads differ)')))


# --------------------------------------------------------------------------
# Corpus / manifest digest consistency

def check_corpus(root, checks, details):
    work = root / 'nl001-substantive'
    corpus_path, manifest_path = work / 'corpus.json', work / 'manifest.json'
    if not corpus_path.is_file() or not manifest_path.is_file():
        checks.append(Check('corpus.digests', 'fail', 'corpus.json or manifest.json missing'))
        return None
    corpus = load_json(corpus_path)
    manifest = load_json(manifest_path)
    problems = []
    body = {k: v for k, v in corpus.items() if k != 'sha256'}
    if corpus.get('sha256') != digest(body):
        problems.append('corpus payload digest mismatch')
    for split in ('train', 'validation', 'test'):
        entries = corpus.get('splits', {}).get(split)
        if entries is None or corpus.get('split_sha256', {}).get(split) != digest(entries):
            problems.append(f'split digest mismatch: {split}')
        descriptor = manifest.get('splits', {}).get(split, {})
        if entries is not None and descriptor.get('sha256') != digest(entries):
            problems.append(f'manifest split digest mismatch: {split}')
        if entries is not None and descriptor.get('seeds') != [e['seed'] for e in entries]:
            problems.append(f'manifest split seeds mismatch: {split}')
    if manifest.get('corpus_sha256') != corpus.get('sha256'):
        problems.append('manifest corpus_sha256 mismatch')
    details['corpus_sha256'] = corpus.get('sha256')
    details['manifest_sha256'] = digest(manifest)
    env_path = work / 'environment.json'
    if env_path.is_file():
        env = load_json(env_path)
        if env.get('corpus_sha256') != corpus.get('sha256'):
            problems.append('environment.json corpus_sha256 mismatch')
        if env.get('manifest_sha256') != digest(manifest):
            problems.append('environment.json manifest_sha256 mismatch')
        details['environment_checker_version'] = env.get('checker_version')
    checks.append(Check('corpus.digests', 'fail' if problems else 'pass',
                        '; '.join(problems) if problems else
                        f"corpus {corpus.get('sha256', '')[:12]}... digests consistent"))
    return manifest


# --------------------------------------------------------------------------
# Per-seed driver

def validate_seed(root, args):
    root = Path(root)
    report = {'output_dir': str(root), 'checks': [], 'details': {}, 'eval': {}}
    checks, details = report['checks'], report['details']

    if root.is_symlink() or not root.is_dir():
        report['verdict'] = 'not_applicable'
        report['verdict_reason'] = ('not applicable - output dir does not exist (job never '
                                    'started or wrong id): ' + str(root))
        checks.append(Check('output_dir', 'skip', report['verdict_reason']))
        return report

    # --- applicability ------------------------------------------------------
    # Full substantive validation applies only when the substantive completion
    # marker exists. Runs without eval artifacts (failed packaging runs, the
    # disposable calibration layout) get a clear not-applicable verdict.
    failure_markers = [name for name in ('nl001-substantive-failure.txt', 'failure-inventory.json')
                       if (root / name).is_file()]
    if not (root / SUBSTANTIVE_MARKER).is_file():
        reasons = []
        if (root / CALIBRATION_MARKER).is_file():
            reasons.append('disposable calibration layout (nl001-calibration-complete.txt), '
                           'not the substantive protocol')
        missing = [name for name in ('output-inventory.json', 'result.json')
                   if not (root / name).is_file()]
        if missing:
            reasons.append('missing ' + ', '.join(missing))
        if failure_markers:
            reasons.append('failure markers present: ' + ', '.join(failure_markers))
        report['verdict'] = 'not_applicable'
        report['verdict_reason'] = ('not applicable - no substantive eval artifacts'
                                    + (': ' + '; '.join(reasons) if reasons else ''))
        checks.append(Check('applicability', 'skip', report['verdict_reason']))
        # Still run the inventory contract when its files exist (e.g. calibration).
        if (root / 'output-inventory.json').is_file() and (root / CALIBRATION_MARKER).is_file() \
                and (root / 'result.json').is_file():
            try:
                outcome = run_inventory_contract(root, args.inventory_validator)
                details['inventory'] = outcome
                checks.append(Check('inventory.contract', 'pass',
                                    f"{outcome['files_verified']} files verified (informational)"))
            except Exception as exc:
                checks.append(Check('inventory.contract', 'fail', str(exc)))
                report['verdict'] = 'fail'
                report['verdict_reason'] = 'inventory contract failed: ' + str(exc)
        return report

    # --- 1. inventory contract ---------------------------------------------
    try:
        outcome = run_inventory_contract(root, args.inventory_validator)
        details['inventory'] = outcome
        checks.append(Check('inventory.contract', 'pass',
                            f"{outcome['files_verified']} files, marker {outcome['marker']}, "
                            f"inventory sha256 {outcome['inventory_sha256'][:12]}..."))
    except Exception as exc:  # the contract raises ValueError/OSError/JSONDecodeError
        details['inventory_error'] = str(exc)
        checks.append(Check('inventory.contract', 'fail', str(exc)))
        report['verdict'] = 'fail'
        return report  # inventory failure: nothing else can be trusted

    # --- frozen checker identity (digest reporting only) --------------------
    snapshot_checker = root / 'nl001-source-snapshot' / 'neural_lab' / 'synthetic_graph.py'
    if snapshot_checker.is_file():
        snap_sha = sha256_file(snapshot_checker)
        details['snapshot_checker_sha256'] = snap_sha
        note = f'snapshot synthetic_graph.py sha256 {snap_sha[:12]}...'
        if args.reference_checker and Path(args.reference_checker).is_file():
            ref_sha = sha256_file(args.reference_checker)
            details['reference_checker_sha256'] = ref_sha
            checks.append(Check('checker.identity', 'pass' if ref_sha == snap_sha else 'fail',
                                note + (' == reference' if ref_sha == snap_sha
                                        else f' != reference {ref_sha[:12]}...')))
        else:
            checks.append(Check('checker.identity', 'pass', note + ' (no reference given)'))
    else:
        checks.append(Check('checker.identity', 'fail', 'source snapshot checker missing'))

    # --- 2/3. per-claim replay over every eval/control dump -----------------
    work = root / 'nl001-substantive'
    eval_dirs = []
    for group in ('eval', 'controls'):
        parent = work / group
        if parent.is_dir():
            for child in sorted(parent.iterdir()):
                if all((child / name).is_file()
                       for name in ('predictions.json', 'split.json', 'evaluation.json')):
                    eval_dirs.append((f'{group}/{child.name}', child))
    details['eval_dirs_found'] = [name for name, _ in eval_dirs]
    for name, path in eval_dirs:
        try:
            outcome = replay_eval_dir(path, fraction=args.sample_fraction,
                                      cap=args.sample_cap, seed=args.sample_seed,
                                      confidence=args.confidence)
        except Exception as exc:
            checks.append(Check(f'eval.{name}', 'fail', f'dump replay error: {exc}'))
            continue
        report['eval'][name] = outcome
        problems = []
        if outcome['record_errors']:
            problems.append(f"{len(outcome['record_errors'])} malformed records")
        if outcome['gold_recheck_mismatch_count']:
            problems.append(f"{outcome['gold_recheck_mismatch_count']} gold-label recheck mismatches")
        if outcome['job_report_comparison']['status'] == 'MISMATCH':
            problems.append('recomputed counts differ from job evaluation.json: '
                            + '; '.join(outcome['job_report_comparison']['diffs'][:3]))
        checks.append(Check(f'eval.{name}', 'fail' if problems else 'pass',
                            '; '.join(problems) if problems else
                            f"{outcome['records_replayed']}/{outcome['records_total']} claims "
                            f"({outcome['sample']}), mode {outcome['mode']}, "
                            f"replay {outcome['recomputed']['counts']['replay_passed']}"
                            f"/{outcome['recomputed']['counts']['replay_attempts']} passed, "
                            f"false-verification "
                            f"{outcome['recomputed']['false_verification_numerator']}"
                            f"/{outcome['recomputed']['false_verification_denominator']}"))

    # Arm coverage required by the substantive protocol.
    for arm in ('A', 'B'):
        name = f'eval/test-{arm}'
        if name not in report['eval']:
            checks.append(Check(f'eval.{name}.presence', 'fail',
                                'required per-claim test dump missing'))
        else:
            expected_mode = 'direct' if arm == 'A' else 'checked'
            if report['eval'][name]['mode'] != expected_mode:
                checks.append(Check(f'eval.{name}.mode', 'fail',
                                    f"expected mode {expected_mode}, got {report['eval'][name]['mode']}"))

    # --- 4. checkpoint selection integrity ----------------------------------
    check_checkpoints(root, checks, details)

    # --- 5. corpus/manifest digests ------------------------------------------
    check_corpus(root, checks, details)

    report['verdict'] = 'fail' if any(c.status == 'fail' for c in checks) else 'pass'
    return report


# --------------------------------------------------------------------------

def combined_summary(reports, confidence):
    arms = {}
    for arm in ('A', 'B'):
        totals = {'claims': 0, 'proven': 0, 'refuted': 0, 'unknown': 0, 'supported': 0,
                  'false_proven': 0, 'replay_attempts': 0, 'replay_passed': 0}
        complete = True
        for report in reports:
            outcome = report.get('eval', {}).get(f'eval/test-{arm}')
            if report.get('verdict') != 'pass' or not outcome or outcome['sample'] != 'all':
                complete = False
                break
            for key in totals:
                totals[key] += outcome['recomputed']['counts'][key]
        entry = {'complete_full_replay': complete, 'totals': totals if complete else None}
        if complete:
            entry['false_verification_rate'] = (totals['false_proven'] / totals['proven']
                                                if totals['proven'] else None)
            entry['false_verification_upper_bound'] = upper_wilson(
                totals['false_proven'], totals['proven'], confidence)
            if totals['proven'] == 0:
                entry['false_verification_note'] = ('zero certifications: false-verification '
                                                    'risk is undefined, not zero')
        arms[arm] = entry
    return arms


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('outputs', nargs='+', help='output dirs (or IDs under --outputs-root)')
    parser.add_argument('--outputs-root', type=Path,
                        default=Path('/repo/.neural-lab/queue-outputs'),
                        help='resolve bare job IDs under this root')
    parser.add_argument('--out', type=Path, default=Path('.'),
                        help='directory for JSON reports (default: CWD)')
    parser.add_argument('--inventory-validator', type=Path,
                        default=Path(__file__).resolve().with_name('validate_nl001_output.py'))
    parser.add_argument('--reference-checker', type=Path,
                        default=Path('/repo/neural_lab/synthetic_graph.py'),
                        help='frozen checker reference for digest comparison (missing = skip)')
    parser.add_argument('--sample-fraction', type=float, default=1.0,
                        help='claim replay fraction when a dump exceeds --sample-cap')
    parser.add_argument('--sample-cap', type=int, default=100000,
                        help='replay ALL claims at or below this size (default 100000)')
    parser.add_argument('--sample-seed', type=int, default=20261107)
    parser.add_argument('--confidence', type=float, default=0.95)
    args = parser.parse_args()

    if not args.inventory_validator.is_file():
        parser.exit(1, f'inventory validator not found: {args.inventory_validator}\n')

    roots = []
    for item in args.outputs:
        path = Path(item)
        if not path.is_dir() and args.outputs_root:
            path = args.outputs_root / item
        roots.append(path)

    args.out.mkdir(parents=True, exist_ok=True)
    reports = []
    for root in roots:
        report = validate_seed(root, args)
        reports.append(report)
        name = f'nl001-postexit-{Path(root).name}.json'
        (args.out / name).write_text(json.dumps(
            report, indent=2, sort_keys=True,
            default=lambda o: o.as_dict() if isinstance(o, Check) else repr(o)) + '\n')

    summary = {
        'seeds': [{'output_dir': r['output_dir'], 'verdict': r['verdict'],
                   'failed_checks': [c.name for c in r['checks'] if c.status == 'fail']}
                  for r in reports],
        'combined_arms': combined_summary(reports, args.confidence),
        'sample_policy': {'fraction': args.sample_fraction, 'cap': args.sample_cap,
                          'seed': args.sample_seed},
        'confidence': args.confidence,
    }
    (args.out / 'nl001-postexit-summary.json').write_text(
        json.dumps(summary, indent=2, sort_keys=True) + '\n')

    any_fail = any(r['verdict'] == 'fail' for r in reports)
    any_na = any(r['verdict'] == 'not_applicable' for r in reports)
    print('NL-001 post-exit validation')
    for report, root in zip(reports, roots):
        print(f"  {Path(root).name}: {report['verdict'].upper()}")
        if report['verdict'] == 'not_applicable':
            print(f"    {report['verdict_reason']}")
        for check in report['checks']:
            if check.status != 'pass':
                print(f'    [{check.status}] {check.name}: {check.detail}')
        for arm in ('A', 'B'):
            outcome = report.get('eval', {}).get(f'eval/test-{arm}')
            if outcome:
                rec = outcome['recomputed']
                bound = rec['false_verification_upper_bound']
                print(f"    arm {arm}: {outcome['records_replayed']} claims, "
                      f"proven={rec['counts']['proven']} refuted={rec['counts']['refuted']} "
                      f"unknown={rec['counts']['unknown']} supported={rec['counts']['supported']}, "
                      f"false-verification {rec['false_verification_numerator']}"
                      f"/{rec['false_verification_denominator']}"
                      + (f' (Wilson {args.confidence:.0%} upper {bound:.6f})'
                         if bound is not None else ' (undefined: no certifications)'))
    print(f'  reports written to {args.out.resolve()}')
    if any_fail:
        print('OVERALL: FAIL')
        return 1
    if any_na:
        print('OVERALL: NOT APPLICABLE (no eval artifacts for at least one seed; no failures)')
        return 2
    print('OVERALL: PASS')
    return 0


if __name__ == '__main__':
    sys.exit(main())
