"""Unreviewed disposable NL-001 queue candidate. Do not enqueue before review.

The test split is a disposable audit fixture generated in the same process, not a
sealed substantive test set. This candidate cannot establish held-out isolation.
"""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import hashlib
import json
import platform
import sys
import time
import traceback
from pathlib import Path

# Filled from the reviewed package snapshot when this candidate is frozen.
PACKAGE_SHA256 = {'__init__.py': 'b1a6036b6786d7e6b2e702799084e6a51bd3e7e4e25f8b801466388c8e4d7393', 'artifacts.py': '6ef5863486d3f9cfa4fac8b06823ff399986c78e898130e979a9035c6f062baf', 'synthetic_graph.py': 'fed818eeffac595415a4e6b2b93edf6fc804f9dcdcf687e918d2a13dc16e2b6b', 'nl001_corpus.py': 'd6def6ee8d904a9cb05e4b46ecad7bca4c9296cd73f6d57cc4ab54d4ea8c27d9', 'nl001_corpus_audit.py': '00ca8f3b8cd8ff5a1e9dd396592aa2e2a7b5e9e3a3e55f728d03c53451cb67cc', 'nl001_data.py': '327c79be554b859fd636f2c1bed9f3e894019ca89b93e657c618fc2afd3ada35', 'nl001_evaluation.py': '87124f56b4b80c6bc2868c599a738ecf983df8fd6f936be0e2a4d716eded2c39', 'nl001_inference.py': 'fe90da5c2dc32145999ef82aa0d5f5ae8832f851fdee793fc2d9129bffc37fc8', 'nl001_manifest.py': 'e287302bcf07cbc4d35ae29a792485fee99e7153123046687034b2975873e661', 'nl001_model_input.py': 'de75724ffeed796c3b537f154e107f3b67ea81795c2fb0e33402955cc5d2d44a', 'nl001_models.py': '4f53bb9ce05e4d3055cf318507d65b57fc219b28e2fb968ad57927842765c195', 'nl001_parser.py': '67db4d93728f52957a6df0406be6af989465ce8ee2a57dd32cc59c18d89786d7', 'nl001_run_evaluation.py': '33ef699edd4e933d0dc738011ad4c3298449bb260cd09fb99b2513d48cdb8009', 'nl001_selection_runner.py': '74682abb92645308be98c1d07420507add5a27cff331db52d6d416ff0d69456d', 'nl001_split_audit.py': 'f806a590828daeb94f179723af5d569aec70ac779fbe3c6c16de5a2c1824f921', 'nl001_training.py': '33f6336c67e01e9fb3b16477d133ed36fcf70c265fe57b82fa80cbc03f111451', 'nl001_training_data.py': 'b82d03d23656f80b3ac70972a53e48fffd94c365f329e4314891292f0cc02d2b'}
OUTPUT = Path('/output')
CONFIG = {'split_seeds': {'train': list(range(10001, 10033)),
                          'validation': list(range(20001, 20009)),
                          'test': [30001]},
          'nodes': 10, 'per_stratum': 2, 'sampling_seed': 20261106}


def write(path, obj):
    path.write_text(json.dumps(obj, sort_keys=True, indent=2, allow_nan=False))


def package_check():
    import neural_lab
    root = Path(neural_lab.__file__).resolve().parent
    if not PACKAGE_SHA256:
        raise RuntimeError('candidate requires independently reviewed package digest freeze')
    for name, expected in PACKAGE_SHA256.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError('package digest mismatch: ' + name)


def inventory(root):
    result = []
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError('symlink in output')
        if path.is_file() and path.name != 'inventory.json':
            result.append({'path': path.relative_to(root).as_posix(),
                           'bytes': path.stat().st_size,
                           'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    return result


def main():
    raise RuntimeError('Calibration is blocked pending independent custody, semantic review, integration and resource gates')
    package_check()
    import torch
    from neural_lab.nl001_corpus import make_sampled_corpus
    from neural_lab.nl001_corpus_audit import audit_corpus
    from neural_lab.nl001_data import digest
    from neural_lab.nl001_selection_runner import run
    if not torch.cuda.is_available() or '2070' not in torch.cuda.get_device_name(0):
        raise RuntimeError('calibration requires target RTX 2070 CUDA device')
    start = time.perf_counter()
    corpus = make_sampled_corpus(**CONFIG)
    audit = audit_corpus(corpus, corpus['sha256'])
    work = OUTPUT / 'nl001-calibration'
    work.mkdir(exist_ok=False)
    # Audit fixture is retained for reproducibility. It is not a sealed test.
    write(work / 'disposable-corpus.json', corpus)
    write(work / 'train.json', corpus['splits']['train'])
    write(work / 'validation.json', corpus['splits']['validation'])
    write(work / 'manifest.json', audit['manifest'])
    write(work / 'audit.json', audit)
    write(work / 'environment.json', {'python': sys.version, 'platform': platform.platform(),
        'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(0), 'package_sha256': PACKAGE_SHA256,
        'corpus_generation_audit_seconds': time.perf_counter() - start})
    summary = run(train_path=work / 'train.json', validation_path=work / 'validation.json',
        manifest_path=work / 'manifest.json', manifest_sha256=audit['manifest_sha256'],
        output=work / 'selection', seed=41001, batch_size=32, updates=100,
        learning_rate=0.001, device='cuda:0', audit_path=work / 'audit.json',
        audit_sha256=digest(audit), edge_threshold=0.5)
    metrics = {'wall_seconds': time.perf_counter() - start}
    for arm in ('A', 'B'):
        training = summary['reports'][arm]['training']
        metrics[arm + '_training_seconds'] = training['training_seconds']
        metrics[arm + '_peak_allocated_bytes'] = training['peak_allocated_bytes']
        metrics[arm + '_parameters'] = training['parameters']
    write(work / 'inventory.json', inventory(work))
    write(OUTPUT / 'result.json', {'metrics': metrics, 'summary':
        'Disposable NL-001 calibration; see nl001-calibration/selection/selection-summary.json and inventory.json. Test fixture is not isolated; same-code replay and audit are not independent semantic validation.'})


if __name__ == '__main__':
    try:
        main()
    except Exception:
        # Preserve failure details while exiting nonzero; never publish success.
        (OUTPUT / 'nl001-calibration-failure.txt').write_text(traceback.format_exc())
        raise
