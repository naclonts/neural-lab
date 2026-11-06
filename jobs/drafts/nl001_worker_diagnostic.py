"""Unreviewed no-training NL-001 worker diagnostic; do not enqueue before review.

Uses disposable generated data and CPU inference. A successful result is an
integration check, not evidence of CUDA readiness or scientific performance.
"""
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

PACKAGE_SHA256 = {'__init__.py': 'b1a6036b6786d7e6b2e702799084e6a51bd3e7e4e25f8b801466388c8e4d7393', 'artifacts.py': '6ef5863486d3f9cfa4fac8b06823ff399986c78e898130e979a9035c6f062baf', 'synthetic_graph.py': 'fed818eeffac595415a4e6b2b93edf6fc804f9dcdcf687e918d2a13dc16e2b6b', 'nl001_corpus.py': 'd6def6ee8d904a9cb05e4b46ecad7bca4c9296cd73f6d57cc4ab54d4ea8c27d9', 'nl001_corpus_audit.py': '00ca8f3b8cd8ff5a1e9dd396592aa2e2a7b5e9e3a3e55f728d03c53451cb67cc', 'nl001_data.py': 'c6d84b43220069b168ff48e5902444d36bc29adcaea5df55786b0cdb76ea72dc', 'nl001_evaluation.py': '87124f56b4b80c6bc2868c599a738ecf983df8fd6f936be0e2a4d716eded2c39', 'nl001_inference.py': '6086b17c2efbfd63f8a41c570dff04a1c3591baa64f1c2fe9169daabd176e5c0', 'nl001_manifest.py': 'e287302bcf07cbc4d35ae29a792485fee99e7153123046687034b2975873e661', 'nl001_model_input.py': 'de75724ffeed796c3b537f154e107f3b67ea81795c2fb0e33402955cc5d2d44a', 'nl001_models.py': '4f53bb9ce05e4d3055cf318507d65b57fc219b28e2fb968ad57927842765c195', 'nl001_parser.py': '67db4d93728f52957a6df0406be6af989465ce8ee2a57dd32cc59c18d89786d7', 'nl001_run_evaluation.py': '33ef699edd4e933d0dc738011ad4c3298449bb260cd09fb99b2513d48cdb8009', 'nl001_selection_runner.py': 'f267a8c598da3070f577947059cb8beeda2e598304a5009718434fe267fea422', 'nl001_split_audit.py': 'cc40c17d70d568a4721402d1a3ba0c5d7849143083ac54a67ebb726344d0879e', 'nl001_training.py': '63849e1d70ac9e47c5067e1b818d3304b9c4695a3e371aeddddf2a9b4383bd79', 'nl001_training_data.py': 'b82d03d23656f80b3ac70972a53e48fffd94c365f329e4314891292f0cc02d2b'}

def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)


def main(output=Path('/output')):
    start = time.perf_counter()
    if not output.is_dir() or any(output.iterdir()):
        raise ValueError('diagnostic requires an existing empty output directory')
    import neural_lab
    root = Path(neural_lab.__file__).resolve().parent
    observed = {}
    for name, expected in PACKAGE_SHA256.items():
        observed[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if observed[name] != expected:
            raise ValueError('package digest mismatch: ' + name)
    import torch
    from neural_lab.artifacts import ArtifactWriter
    from neural_lab.nl001_corpus import make_sampled_corpus
    from neural_lab.nl001_data import digest
    from neural_lab.nl001_inference import infer_checkpoint, replay_saved
    from neural_lab.nl001_manifest import make_manifest
    from neural_lab.nl001_models import CalibrationModel
    torch.set_num_threads(1)
    corpus = make_sampled_corpus({'train': [11], 'validation': [12], 'test': [13]}, per_stratum=1)
    programs = corpus['splits']['validation']
    manifest = make_manifest(corpus)
    manifest_pin = digest(manifest)
    work = output / 'diagnostic'
    work.mkdir()
    write(work / 'corpus.json', corpus)
    write(work / 'manifest.json', manifest)
    writer = ArtifactWriter(work)
    writer.register_file('datasets', work / 'corpus.json', 'synthetic-corpus', format='json')
    writer.register_file('datasets', work / 'manifest.json', 'manifest', format='json')
    reports = {}
    for arm in ('A', 'B'):
        torch.manual_seed(711)
        entry = writer.save_checkpoint('untrained-' + arm, CalibrationModel(arm), arm=arm,
                                       seed=711, selection={'scope': 'untrained diagnostic'})
        predictions = work / (arm + '-predictions.json')
        args = dict(programs=programs, split='validation', manifest=manifest,
                    manifest_sha256=manifest_pin, checkpoint_sha256=entry['sha256'], arm=arm)
        prediction_pin = infer_checkpoint(**args, checkpoint=work / entry['path'], output=predictions)
        request = {**args, 'predictions_path': str(predictions.resolve()),
                   'predictions_sha256': prediction_pin, 'edge_threshold': 0.5}
        request_path = work / (arm + '-request.json')
        report_path = work / (arm + '-report.json')
        write(request_path, request)
        command = [sys.executable, '-m', 'neural_lab.nl001_selection_runner', '--replay',
                   str(request_path), '--replay-output', str(report_path)]
        child = subprocess.run(command, text=True, capture_output=True, timeout=120)
        if child.returncode != 0:
            raise RuntimeError('replay subprocess failed: ' + child.stderr[-4000:])
        report = json.loads(report_path.read_text())
        if report != replay_saved(**request):
            raise ValueError('fresh-process replay differs')
        bad_request = {**request, 'predictions_sha256': '0' * 64}
        bad_path = work / (arm + '-bad-request.json')
        bad_report = work / (arm + '-bad-report.json')
        write(bad_path, bad_request)
        bad = subprocess.run([sys.executable, '-m', 'neural_lab.nl001_selection_runner', '--replay',
                              str(bad_path), '--replay-output', str(bad_report)],
                             text=True, capture_output=True, timeout=120)
        if bad.returncode == 0 or bad_report.exists():
            raise ValueError('bad digest did not fail closed')
        for path in (predictions, request_path, report_path, bad_path):
            writer.register_file('datasets', path, path.stem, format='json')
        reports[arm] = {'checkpoint_sha256': entry['sha256'], 'prediction_sha256': prediction_pin,
                        'replay_exit': child.returncode, 'bad_pin_exit': bad.returncode,
                        'bad_pin_stderr': bad.stderr[-4000:]}
    provenance = {'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'package_sha256_at_start': observed, 'python': sys.version,
                  'platform': platform.platform(), 'torch': torch.__version__,
                  'corpus_sha256': corpus['sha256'], 'manifest_sha256': manifest_pin,
                  'reports': reports, 'training': False, 'device': 'cpu',
                  'limitations': 'Live toolkit hashes are checked at startup, not immutable custody. Replay uses the same implementation. No GPU, optimizer, queue timeout or scientific result is tested.'}
    write(work / 'provenance.json', provenance)
    writer.register_file('datasets', work / 'provenance.json', 'provenance', format='json')
    write(output / 'result.json', {'metrics': {'arms': len(reports), 'wall_seconds': time.perf_counter() - start},
          'summary': 'No-training synthetic CPU integration diagnostic; see diagnostic/provenance.json and artifacts.json. Same-code replay does not independently validate semantics or CUDA.'})


if __name__ == '__main__':
    try:
        main()
    except Exception:
        # Never replace an existing result or hide the original failure.
        try:
            with (Path('/output') / 'diagnostic-failure.txt').open('x') as stream:
                stream.write(traceback.format_exc())
        except OSError:
            pass
        raise
