"""Unreviewed disposable NL-001 queue candidate. Do not enqueue before review.

The test split is a disposable audit fixture generated in the same process, not a
sealed substantive test set. This candidate cannot establish held-out isolation.
"""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import hashlib
import importlib.machinery
import json
import platform
import sys
import time
import traceback
from pathlib import Path

# Filled from the reviewed package snapshot when this candidate is frozen.
PACKAGE_SHA256 = {'nl001_child_attestation.py': '1dd350525de8261ff94f1fbdd51cccb3ff568ced43d4fe1a160666c62cc8a9b6', '__init__.py': 'b1a6036b6786d7e6b2e702799084e6a51bd3e7e4e25f8b801466388c8e4d7393', 'artifacts.py': '4f18284d41d34892b1e0128a05f96e0a0f64162f7ff187893455365258f9e1a7', 'synthetic_graph.py': 'fed818eeffac595415a4e6b2b93edf6fc804f9dcdcf687e918d2a13dc16e2b6b', 'nl001_corpus.py': 'd6def6ee8d904a9cb05e4b46ecad7bca4c9296cd73f6d57cc4ab54d4ea8c27d9', 'nl001_corpus_audit.py': '00ca8f3b8cd8ff5a1e9dd396592aa2e2a7b5e9e3a3e55f728d03c53451cb67cc', 'nl001_data.py': 'c6d84b43220069b168ff48e5902444d36bc29adcaea5df55786b0cdb76ea72dc', 'nl001_evaluation.py': '87124f56b4b80c6bc2868c599a738ecf983df8fd6f936be0e2a4d716eded2c39', 'nl001_independent_oracle.py': 'dfe157a2b0e6564fb403ec04838080f35fc6a3ff4018551d365822b86f7c39f1', 'nl001_inference.py': '4bf53850815cddaca4a741adc4b15b85a731a10fe1701d7b277591779a6b00e6', 'nl001_manifest.py': 'e287302bcf07cbc4d35ae29a792485fee99e7153123046687034b2975873e661', 'nl001_model_input.py': 'de75724ffeed796c3b537f154e107f3b67ea81795c2fb0e33402955cc5d2d44a', 'nl001_models.py': '4f53bb9ce05e4d3055cf318507d65b57fc219b28e2fb968ad57927842765c195', 'nl001_parser.py': '67db4d93728f52957a6df0406be6af989465ce8ee2a57dd32cc59c18d89786d7', 'nl001_prediction_audit.py': 'f13bbcde337ac06f7f9065737113db76e8e810a5dc4a9e7bb130946871e5aa5b', 'nl001_run_evaluation.py': '33ef699edd4e933d0dc738011ad4c3298449bb260cd09fb99b2513d48cdb8009', 'nl001_seed_selection.py': 'd71fcbf5eb37ac8d9d32273019b77309dbcddd2f1716d1660f31a11d8ceb90ec', 'nl001_selection_runner.py': 'aa0d032a972bfa6c652ed3ecfe5f62e1021d9ddcbcca569a0093141b3b28a42e', 'nl001_split_audit.py': 'cc40c17d70d568a4721402d1a3ba0c5d7849143083ac54a67ebb726344d0879e', 'nl001_training.py': '63849e1d70ac9e47c5067e1b818d3304b9c4695a3e371aeddddf2a9b4383bd79', 'nl001_training_data.py': 'b82d03d23656f80b3ac70972a53e48fffd94c365f329e4314891292f0cc02d2b'}
OUTPUT = Path('/output')
CONFIG = {'edge_probability': 0.08, 'missing_fraction': 0.25, 'nodes': 10, 'per_stratum': 2, 'sampling_seed': 20261106, 'split_seeds': {'test': [304001], 'train': [104001, 104002, 104003, 104004, 104005, 104006, 104007, 104008, 104009, 104010, 104011, 104012, 104013, 104014, 104015, 104016, 104017, 104018, 104019, 104020, 104021, 104022, 104023, 104024, 104025, 104026, 104027, 104028, 104029, 104030, 104031, 104032], 'validation': [204001, 204002, 204003, 204004, 204005, 204006, 204007, 204008]}}
EXPECTED_CORPUS_SHA256 = 'bd7e01bba43a01416e9727846d57145b29b98be05526119cc2406ae1efa2bf84'


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
    # -m subprocesses resolve the current directory before PYTHONPATH.
    os.chdir(destination)
    sys.path.insert(0, str(destination))
    os.environ['PYTHONPATH'] = str(destination)
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    return package


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



def save_split_artifacts(writer, corpus, manifest_sha256):
    """Persist ordered encoded examples for the exposed disposable splits."""
    from neural_lab.nl001_data import digest
    from neural_lab.nl001_training import _batch
    from neural_lab.nl001_training_data import example

    for split in ('train', 'validation', 'test'):
        programs = corpus['splits'][split]
        records = [example(variant, question) for program in programs
                   for variant in program['variants'] for question in variant['questions']]
        if not records:
            raise ValueError('empty disposable split: ' + split)
        inputs, targets = _batch(records, range(len(records)), 'cpu')
        writer.save_dataset('disposable-' + split + '-tensors',
            {**{'input_' + key: value for key, value in inputs.items()},
             **{'target_' + key: value for key, value in targets.items()}},
            provenance={'split': split, 'split_canonical_sha256': digest(programs),
                        'manifest_sha256': manifest_sha256,
                        'order': 'program, presentation, question in retained split JSON',
                        'scope': 'disposable exposed fixture; targets for validation/test are archival'})


def main():
    raise RuntimeError('Calibration is blocked pending independent custody, semantic review, integration and resource gates')
    snapshot = snapshot_package(OUTPUT / 'nl001-source-snapshot')
    manifest_path = OUTPUT / 'nl001-child-source-manifest.json'
    write(manifest_path, {'package': str(snapshot.resolve()), 'sha256': PACKAGE_SHA256})
    os.environ['NL001_CHILD_ATTESTATION_MANIFEST'] = str(manifest_path)
    import torch
    from neural_lab.artifacts import ArtifactWriter
    from neural_lab.nl001_corpus import make_sampled_corpus
    from neural_lab.nl001_corpus_audit import audit_corpus
    from neural_lab.nl001_data import digest
    from neural_lab.nl001_selection_runner import run
    if not torch.cuda.is_available() or '2070' not in torch.cuda.get_device_name(0):
        raise RuntimeError('calibration requires target RTX 2070 CUDA device')
    start = time.perf_counter()
    corpus = make_sampled_corpus(**CONFIG)
    audit = audit_corpus(corpus, EXPECTED_CORPUS_SHA256)
    work = OUTPUT / 'nl001-calibration'
    work.mkdir(exist_ok=False)
    # Audit fixture is retained for reproducibility. It is not a sealed test.
    write(work / 'disposable-corpus.json', corpus)
    write(work / 'train.json', corpus['splits']['train'])
    write(work / 'validation.json', corpus['splits']['validation'])
    write(work / 'manifest.json', audit['manifest'])
    write(work / 'audit.json', audit)
    writer = ArtifactWriter(OUTPUT)
    for name in ('disposable-corpus.json', 'train.json', 'validation.json', 'manifest.json', 'audit.json'):
        writer.register_file('datasets', work / name, name, format='json')
    save_split_artifacts(writer, corpus, audit['manifest_sha256'])
    write(work / 'environment.json', {'python': sys.version, 'platform': platform.platform(),
        'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(0), 'package_sha256': PACKAGE_SHA256,
        'corpus_generation_audit_seconds': time.perf_counter() - start})
    summary = run(train_path=work / 'train.json', validation_path=work / 'validation.json',
        manifest_path=work / 'manifest.json', manifest_sha256=audit['manifest_sha256'],
        output=work / 'selection', seed=41001, batch_size=32, updates=100,
        learning_rate=0.001, device='cuda:0', audit_path=work / 'audit.json',
        audit_sha256=digest(audit), edge_threshold=0.5)
    for arm in ('A', 'B'):
        checkpoint = work / 'selection' / arm / 'checkpoints' / ('best-' + arm + '.json')
        entry = writer.register_file('checkpoints', checkpoint, 'best-' + arm, format='json-state-dict', arm=arm)
        if entry['sha256'] != summary['reports'][arm]['training']['checkpoint_sha256']:
            raise RuntimeError('root checkpoint digest mismatch: ' + arm)
    metrics = {'wall_seconds': time.perf_counter() - start}
    for arm in ('A', 'B'):
        training = summary['reports'][arm]['training']
        metrics[arm + '_training_seconds'] = training['training_seconds']
        metrics[arm + '_peak_allocated_bytes'] = training['peak_allocated_bytes']
        metrics[arm + '_peak_reserved_bytes'] = training['peak_reserved_bytes']
        metrics[arm + '_parameters'] = training['parameters']
        metrics[arm + '_inference_seconds'] = summary['reports'][arm]['inference_seconds']
        metrics[arm + '_fresh_inference_seconds'] = summary['reports'][arm]['fresh_process_inference_seconds']
        metrics[arm + '_replay_seconds'] = summary['reports'][arm]['fresh_process_replay_seconds']
        metrics[arm + '_independent_audit_seconds'] = summary['reports'][arm]['independent_audit_seconds']
    # Detect persistent changes and unexpected package imports before success.
    for name, expected in PACKAGE_SHA256.items():
        if hashlib.sha256((snapshot / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError('snapshot digest changed: ' + name)
    imported = {}
    for name, module in tuple(sys.modules.items()):
        if name == 'neural_lab' or name.startswith('neural_lab.'):
            path = Path(module.__file__).resolve()
            if path.parent != snapshot.resolve() or path.name not in PACKAGE_SHA256:
                raise RuntimeError('package imported outside verified snapshot: ' + name)
            imported[name] = str(path)
    write(work / 'source-attestation.json', {'imported_modules': imported,
        'post_run_sha256': PACKAGE_SHA256, 'limitation': 'point-in-time check; writable snapshot and runtime are not immutable'})
    write(work / 'inventory.json', inventory(work, exclude={'inventory.json'}))
    write(OUTPUT / 'result.json', {'metrics': metrics, 'summary':
        'Disposable NL-001 calibration; see nl001-calibration/selection/selection-summary.json and inventory.json. Test fixture is not isolated; same-code replay and audit are not independent semantic validation.'})
    write(OUTPUT / 'output-inventory.json', inventory(OUTPUT, exclude={'output-inventory.json'}))
    # A result without this last marker may be a killed or interrupted run.
    (OUTPUT / 'nl001-calibration-complete.txt').write_text(
        hashlib.sha256((OUTPUT / 'output-inventory.json').read_bytes()).hexdigest() + '\n')


def guarded_entry():
    """Best-effort ordinary-exception diagnostics; SIGKILL/OOM are not catchable."""
    try:
        main()
    except Exception:
        try:
            OUTPUT.mkdir(parents=True, exist_ok=True)
            (OUTPUT / 'result.json').unlink(missing_ok=True)
            (OUTPUT / 'nl001-calibration-failure.txt').write_text(traceback.format_exc())
            write(OUTPUT / 'failure-inventory.json', inventory(OUTPUT, exclude={'failure-inventory.json'}))
        except Exception:
            traceback.print_exc()
        raise


if __name__ == '__main__':
    guarded_entry()
