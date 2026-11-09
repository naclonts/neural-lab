#!/usr/bin/env python3
"""Tests and live dry runs for scripts/validate_nl001_postexit.py.

CPU-only, stdlib-only, no queue interaction, no writes outside /tmp and the
report dirs. Uses the pinned source snapshot of the running seed-41001 job
(read-only) to generate a synthetic fixture in the SUBSTANTIVE output schema,
then exercises the validator on:
  1. the fixture (expect PASS, exit 0);
  2. tampered copies (certificate, gold label, false verification, selection,
     inventory - expect FAIL, exit 1);
  3. a no-eval dir (expect NOT APPLICABLE, exit 2);
  4. the real disposable calibration output 45006938... (exit 2, inventory ok);
  5. the four real failed substantive outputs (exit 2 each);
  6. real per-claim dumps of failed run d52f8c21... via direct module calls
     (2,172-claim control dumps, counts compared against the job's report);
  7. real checkpoints of d52f8c21... via the restricted .pt reader.
"""
import copy
import hashlib
import importlib.util
import io
import json
import pickle
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = Path('/repo')
VALIDATOR = REPO / 'scripts' / 'validate_nl001_postexit.py'
SNAPSHOT = (REPO / '.neural-lab/queue-outputs/c6e34e33f19149689a2913f8a1a563ea'
            / 'nl001-source-snapshot')
OUTPUTS = REPO / '.neural-lab/queue-outputs'
TMP = Path('/tmp/nl001-postexit-tests')
FIXTURE = TMP / 'fixture-ok'

FAILED_IDS = ['d52f8c215290475baed813eeeeeffa3a', '7354aa35dbaa43c18600ddf5c0dfdc55',
              'adbfdea333ef466d879b7ee52b800adf', '8c177abd01b343ac90f8a30642728e96']
CALIBRATION_ID = '450069380ee1482f86404864bf567b38'


def load_validator():
    spec = importlib.util.spec_from_file_location('nl001_postexit', VALIDATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_snapshot_modules():
    sys.path.insert(0, str(SNAPSHOT))
    for name in ('nl001_run_evaluation', 'nl001_manifest', 'nl001_data',
                 'nl001_corpus', 'synthetic_graph', 'artifacts'):
        __import__('neural_lab.' + name)
    corpus_mod = sys.modules['neural_lab.nl001_corpus']
    assert str(SNAPSHOT) in corpus_mod.__file__, corpus_mod.__file__
    return corpus_mod


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False, indent=2))


def write_pt(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as archive:
        archive.writestr('fixture/data.pkl', pickle.dumps(payload, protocol=2))
    path.write_bytes(buffer.getvalue())


def refresh_inventory(root):
    """Rewrite output-inventory.json + completion marker after a deliberate mutation."""
    root = Path(root)
    entries = []
    for path in sorted(root.rglob('*')):
        if path.is_file() and path.name not in ('output-inventory.json',
                                                'nl001-substantive-complete.txt'):
            entries.append({'path': path.relative_to(root).as_posix(),
                            'bytes': path.stat().st_size,
                            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    (root / 'output-inventory.json').write_text(json.dumps(entries, indent=1))
    digest = hashlib.sha256((root / 'output-inventory.json').read_bytes()).hexdigest()
    (root / 'nl001-substantive-complete.txt').write_text(digest + '\n')


def build_fixture():
    corpus_mod = load_snapshot_modules()
    from neural_lab import nl001_manifest
    from neural_lab.nl001_data import digest
    from neural_lab.nl001_run_evaluation import evaluate_split

    if FIXTURE.exists():
        shutil.rmtree(FIXTURE)
    FIXTURE.mkdir(parents=True)

    # Small deterministic corpus; find seeds that generate at least one edge.
    def valid_seeds(count, start):
        import random
        found, seed = [], start
        while len(found) < count:
            rng = random.Random(seed)
            edges = [(a, b) for a in range(10) for b in range(10) if rng.random() < 0.08]
            if edges:
                found.append(seed)
            seed += 1
        return found

    split_seeds = {'train': valid_seeds(2, 510000), 'validation': valid_seeds(2, 520000),
                   'test': valid_seeds(2, 530000)}
    corpus = corpus_mod.make_sampled_corpus(split_seeds, nodes=10, per_stratum=2)
    manifest = nl001_manifest.make_manifest(corpus)
    manifest_sha = digest(manifest)

    work = FIXTURE / 'nl001-substantive'
    write_json(work / 'corpus.json', corpus)
    write_json(work / 'manifest.json', manifest)
    write_json(work / 'environment.json',
               {'corpus_sha256': corpus['sha256'], 'manifest_sha256': manifest_sha,
                'checker_version': corpus['checker_version'], 'smoke': False})

    def items_of(programs):
        return [(v, q) for p in programs for v in p['variants'] for q in v['questions']]

    # Arm B (checked): perfect gold predictions; Arm A (direct): gold strings.
    for arm, mode in (('A', 'direct'), ('B', 'checked')):
        for split in ('test',):
            programs = corpus['splits'][split]
            if mode == 'checked':
                predictions = [{'query': q['query'], 'result': q['label'],
                                'certificate': q['certificate']}
                               for _, q in items_of(programs)]
            else:
                predictions = [q['label'] if q['label'] in ('Proven', 'Refuted') else 'Unknown'
                               for _, q in items_of(programs)]
            evaluate_split(programs=programs, split=split, manifest=manifest,
                           manifest_sha256=manifest_sha,
                           output=work / 'eval' / f'{split}-{arm}', mode=mode,
                           predictions=predictions)
    for split in ('validation', 'test'):
        for mode in ('parser', 'oracle'):
            evaluate_split(programs=corpus['splits'][split], split=split, manifest=manifest,
                           manifest_sha256=manifest_sha,
                           output=work / 'controls' / f'{split}-{mode}', mode=mode)

    # Training reports + fabricated but format-faithful .pt checkpoints.
    steps = [2000, 4000, 6000, 8000, 10000]
    artifacts = {'version': 1, 'datasets': [], 'checkpoints': []}
    for arm in ('A', 'B'):
        ces = [0.9, 0.7, 0.5, 0.4, 0.6] if arm == 'A' else [1.0, 0.8, 0.55, 0.65, 0.7]
        observations = [{'step': s, 'training_cross_entropy': c} for s, c in zip(steps, ces)]
        best_value, best_step = min(((c, s) for s, c in zip(steps, ces)), key=lambda t: (t[0], t[1]))
        selection = {'metric': 'training_cross_entropy', 'mode': 'min', 'value': best_value,
                     'step': best_step, 'observations': len(observations),
                     'observed_steps': steps, 'scope': 'fixture scope'}
        config = {'version': 'nl001-substantive-session40-v1', 'arm': arm, 'seed': 41001,
                  'updates': 10000}
        final_selection = {'metric': 'none', 'mode': 'final update weights, NOT selected',
                           'step': 10000, 'scope': 'retained for diagnostics only'}
        report = {'config': config, 'selection_observations': observations,
                  'best_selection': selection}
        for kind, sel in (('best', selection), ('final', final_selection)):
            path = FIXTURE / 'checkpoints' / f'{kind}-{arm}.pt'
            write_pt(path, {'format': 'neural-lab-state-dict-pt-v1', 'tensors': {},
                            'arm': arm, 'seed': 41001, 'selection': sel, 'config': config})
            artifacts['checkpoints'].append({
                'label': f'{kind}-{arm}', 'path': f'checkpoints/{kind}-{arm}.pt',
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'bytes': path.stat().st_size, 'format': 'torch-pt', 'arm': arm, 'seed': 41001,
                'selection': sel, 'config': config, 'datasets': [], 'tensors': {}})
            report[f'{kind}_checkpoint_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        write_json(work / f'training-report-{arm}.json', report)
    write_json(FIXTURE / 'artifacts.json', artifacts)

    (FIXTURE / 'nl001-source-snapshot' / 'neural_lab').mkdir(parents=True)
    shutil.copy(REPO / 'neural_lab' / 'synthetic_graph.py',
                FIXTURE / 'nl001-source-snapshot' / 'neural_lab' / 'synthetic_graph.py')
    write_json(FIXTURE / 'result.json',
               {'metrics': {'wall_seconds': 1.0, 'b_test_answer_accuracy': 1.0},
                'summary': 'synthetic fixture in the NL-001 substantive output schema'})
    refresh_inventory(FIXTURE)
    return corpus


def run_validator(*dirs, out):
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    proc = subprocess.run([sys.executable, str(VALIDATOR), *map(str, dirs), '--out', str(out)],
                          capture_output=True, text=True)
    return proc


def load_report(out, root):
    return json.loads((Path(out) / f'nl001-postexit-{Path(root).name}.json').read_text())


def fresh_copy(src, dst):
    if Path(dst).exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def main():
    val = load_validator()
    results = []

    def record(name, ok, detail=''):
        results.append((name, bool(ok), detail))
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")

    print('== building synthetic substantive fixture in', FIXTURE)
    corpus = build_fixture()
    n_test = sum(len(v['questions']) for p in corpus['splits']['test'] for v in p['variants'])
    print(f'   fixture test claims (4 presentations): {n_test}')

    # 1. happy path ------------------------------------------------------------
    proc = run_validator(FIXTURE, out=TMP / 'out-ok')
    report = load_report(TMP / 'out-ok', FIXTURE)
    record('fixture passes end to end', proc.returncode == 0 and report['verdict'] == 'pass',
           f'exit={proc.returncode}')
    if proc.returncode != 0:
        print(proc.stdout, proc.stderr)
    b = report['eval']['eval/test-B']
    record('fixture B replay: zero false verifications, all replays pass',
           b['recomputed']['false_verification_numerator'] == 0
           and b['recomputed']['counts']['replay_rejected'] == 0
           and b['job_report_comparison']['status'] == 'match',
           f"counts={b['recomputed']['counts']}")

    # 2. determinism -----------------------------------------------------------
    proc2 = run_validator(FIXTURE, out=TMP / 'out-ok-2')
    same = (Path(TMP / 'out-ok' / f'nl001-postexit-{FIXTURE.name}.json').read_bytes()
            == Path(TMP / 'out-ok-2' / f'nl001-postexit-{FIXTURE.name}.json').read_bytes())
    record('deterministic reruns produce byte-identical reports',
           proc2.returncode == 0 and same)

    # 2b. deterministic seeded sampling when a dump exceeds the cap ----------
    proc = run_validator(FIXTURE, '--sample-cap', '10', '--sample-fraction', '0.1',
                         out=TMP / 'out-sample')
    report = load_report(TMP / 'out-sample', FIXTURE)
    b = report['eval']['eval/test-B']
    proc2 = run_validator(FIXTURE, '--sample-cap', '10', '--sample-fraction', '0.1',
                          out=TMP / 'out-sample-2')
    report2 = load_report(TMP / 'out-sample-2', FIXTURE)
    record('seeded sample mode: 14 of 144 claims, comparison skipped, reproducible',
           proc.returncode == 0 and b['records_replayed'] == 14
           and b['sample'].startswith('seeded-sample')
           and b['job_report_comparison']['status'] == 'skipped'
           and b['sample_indices_sha256']
           == report2['eval']['eval/test-B']['sample_indices_sha256'],
           f"exit={proc.returncode} sample={b['sample']}")

    # 3. tamper: certificate path endpoint (fresh inventory -> replay must catch)
    tam = TMP / 'tamper-cert'
    fresh_copy(FIXTURE, tam)
    preds_path = tam / 'nl001-substantive' / 'eval' / 'test-B' / 'predictions.json'
    preds = json.loads(preds_path.read_text())
    target = next(r for r in preds if isinstance(r['prediction'], dict)
                  and r['prediction'].get('result') == 'Proven'
                  and r['prediction']['certificate']['kind'] == 'path')
    victim = target['prediction']['certificate']['nodes'][-1]
    other = next(s for s in json.loads((tam / 'nl001-substantive' / 'eval' / 'test-B'
                                        / 'split.json').read_text())[0]['variants'][0]['symbols']
                 if s != victim)
    target['prediction']['certificate']['nodes'][-1] = other
    preds_path.write_text(json.dumps(preds))
    refresh_inventory(tam)
    proc = run_validator(tam, out=TMP / 'out-tamper-cert')
    report = load_report(TMP / 'out-tamper-cert', tam)
    b = report['eval']['eval/test-B']
    record('tampered certificate rejected by replay',
           proc.returncode == 1 and b['recomputed']['counts']['replay_rejected'] == 1
           and b['recomputed']['counts']['proven'] == 64 - 1
           and b['job_report_comparison']['status'] == 'MISMATCH',
           f"exit={proc.returncode} rejected={b['recomputed']['counts']['replay_rejected']} "
           f"comparison={b['job_report_comparison']['status']}")

    # 4. tamper: gold label (gold recheck must catch)
    tam = TMP / 'tamper-label'
    fresh_copy(FIXTURE, tam)
    preds_path = tam / 'nl001-substantive' / 'eval' / 'test-B' / 'predictions.json'
    preds = json.loads(preds_path.read_text())
    target = next(r for r in preds if r['label'] == 'Proven')
    target['label'] = 'Refuted'
    preds_path.write_text(json.dumps(preds))
    refresh_inventory(tam)
    proc = run_validator(tam, out=TMP / 'out-tamper-label')
    report = load_report(TMP / 'out-tamper-label', tam)
    b = report['eval']['eval/test-B']
    record('tampered gold label detected',
           proc.returncode == 1 and (b['record_errors'] or b['gold_recheck_mismatch_count']),
           f"exit={proc.returncode} errors={len(b['record_errors'])} "
           f"gold_mismatch={b['gold_recheck_mismatch_count']}")

    # 5. tamper: false verification (misaligned but valid certificate)
    tam = TMP / 'tamper-falsever'
    fresh_copy(FIXTURE, tam)
    preds_path = tam / 'nl001-substantive' / 'eval' / 'test-B' / 'predictions.json'
    preds = json.loads(preds_path.read_text())
    by_variant = {}
    for r in preds:
        if isinstance(r['prediction'], dict) and r['prediction'].get('result') == 'Proven':
            by_variant.setdefault((r['source'], json.dumps(r['presentation'], sort_keys=True),
                                   r['view']), []).append(r)
    pair = next(v for v in by_variant.values() if len(v) >= 2)
    pair[0]['prediction'] = copy.deepcopy(pair[1]['prediction'])  # valid cert, wrong claim
    preds_path.write_text(json.dumps(preds))
    refresh_inventory(tam)
    proc = run_validator(tam, out=TMP / 'out-tamper-falsever')
    report = load_report(TMP / 'out-tamper-falsever', tam)
    b = report['eval']['eval/test-B']
    record('false verification counted (numerator becomes 1)',
           proc.returncode == 1 and b['recomputed']['false_verification_numerator'] == 1
           and b['recomputed']['counts']['replay_rejected'] == 0,
           f"exit={proc.returncode} fvr={b['recomputed']['false_verification_numerator']}"
           f"/{b['recomputed']['false_verification_denominator']} "
           f"upper={b['recomputed']['false_verification_upper_bound']}")

    # 6. tamper: checkpoint selection step (inventory refreshed -> selection check)
    tam = TMP / 'tamper-selection'
    fresh_copy(FIXTURE, tam)
    report_path = tam / 'nl001-substantive' / 'training-report-A.json'
    training = json.loads(report_path.read_text())
    training['best_selection']['step'] += 2000
    report_path.write_text(json.dumps(training))
    refresh_inventory(tam)
    proc = run_validator(tam, out=TMP / 'out-tamper-selection')
    report = load_report(TMP / 'out-tamper-selection', tam)
    failed = [c['name'] for c in report['checks'] if c['status'] == 'fail']
    record('tampered best-checkpoint step detected',
           proc.returncode == 1 and 'checkpoints.arm_A.selection_curve' in failed,
           f'exit={proc.returncode} failed={failed}')

    # 7. tamper: stale inventory (append a byte, do not refresh)
    tam = TMP / 'tamper-inventory'
    fresh_copy(FIXTURE, tam)
    with open(tam / 'nl001-substantive' / 'corpus.json', 'ab') as stream:
        stream.write(b' ')
    proc = run_validator(tam, out=TMP / 'out-tamper-inventory')
    report = load_report(TMP / 'out-tamper-inventory', tam)
    record('stale inventory detected by output-inventory contract',
           proc.returncode == 1
           and report['checks'][0]['name'] == 'inventory.contract'
           and report['checks'][0]['status'] == 'fail',
           f"exit={proc.returncode} detail={report['checks'][0]['detail'][:80]}")

    # 8. no-eval dir -> not applicable
    barren = TMP / 'barren'
    if barren.exists():
        shutil.rmtree(barren)
    (barren / 'datasets').mkdir(parents=True)
    (barren / 'datasets' / 'placeholder.txt').write_text('nothing')
    proc = run_validator(barren, out=TMP / 'out-barren')
    report = load_report(TMP / 'out-barren', barren)
    record('dir without eval artifacts -> not applicable (exit 2)',
           proc.returncode == 2 and report['verdict'] == 'not_applicable',
           f"exit={proc.returncode} reason={report.get('verdict_reason', '')[:70]}")

    # 9. real disposable calibration output -----------------------------------
    proc = run_validator(CALIBRATION_ID, out=TMP / 'out-calibration')
    report = load_report(TMP / 'out-calibration', CALIBRATION_ID)
    inv = next((c for c in report['checks'] if c['name'] == 'inventory.contract'), None)
    record('calibration 45006938: NA verdict + inventory contract passes',
           proc.returncode == 2 and report['verdict'] == 'not_applicable'
           and inv and inv['status'] == 'pass',
           f"exit={proc.returncode} inventory={inv and inv['status']}")

    # 10. real failed substantive outputs --------------------------------------
    for job_id in FAILED_IDS:
        proc = run_validator(job_id, out=TMP / f'out-failed-{job_id[:8]}')
        report = load_report(TMP / f'out-failed-{job_id[:8]}', job_id)
        record(f'failed run {job_id[:8]}: graceful NA (exit 2)',
               proc.returncode == 2 and report['verdict'] == 'not_applicable'
               and 'failure' in report.get('verdict_reason', ''),
               f"exit={proc.returncode}")

    # 11. real per-claim dumps of failed run d52f8c21 (substantive schema) -----
    base = OUTPUTS / FAILED_IDS[0] / 'nl001-substantive' / 'controls'
    for name in ('validation-oracle', 'validation-parser'):
        outcome = val.replay_eval_dir(base / name, fraction=1.0, cap=100000,
                                      seed=20261107, confidence=0.95)
        record(f'real dump replay d52f8c21 {name}: counts match job report',
               outcome['records_replayed'] == outcome['records_total'] == 2172
               and outcome['job_report_comparison']['status'] == 'match'
               and not outcome['record_errors'] and not outcome['gold_recheck_mismatch_count'],
               f"claims={outcome['records_replayed']} "
               f"replay={outcome['recomputed']['counts']['replay_passed']}"
               f"/{outcome['recomputed']['counts']['replay_attempts']} "
               f"false_proven={outcome['recomputed']['false_verification_numerator']}")

    # 12. real checkpoints of d52f8c21 via restricted .pt reader ----------------
    checks, details = [], {}
    val.check_checkpoints(OUTPUTS / FAILED_IDS[0], checks, details)
    failed = [c for c in checks if c.status == 'fail']
    record('real checkpoint selection integrity d52f8c21 (all arms)',
           checks and not failed,
           f'{len(checks)} checks; best step A='
           f"{details.get('arm_A_selection', {}).get('recomputed_min_step')}")

    print()
    failed_tests = [r for r in results if not r[1]]
    print(f'{len(results) - len(failed_tests)}/{len(results)} test groups passed')
    for name, ok, detail in failed_tests:
        print('FAILED:', name, detail)
    return 1 if failed_tests else 0


if __name__ == '__main__':
    sys.exit(main())
