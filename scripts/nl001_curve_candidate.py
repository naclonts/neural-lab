#!/usr/bin/env python3
"""Unreviewed validation-only candidate; do not queue without an independent freeze.

Run with --config and an independently retained raw SHA-256 --config-sha256.
Inputs are train, validation, manifest and corpus-audit JSON only. Staging is
owner-private, not a security boundary against the same user or administrator.
"""
import argparse
import ctypes
import errno
import hashlib
import json
import os
import platform
import traceback
import sys
import tempfile
import time
from pathlib import Path


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False))


def checked(path, expected):
    data = Path(path).read_bytes()
    if sha(data) != expected:
        raise ValueError('digest mismatch: ' + str(path))
    return data


def rename_noreplace(source, destination):
    """Linux atomic no-clobber rename; fail closed if unavailable."""
    libc = ctypes.CDLL(None, use_errno=True)
    rename = getattr(libc, 'renameat2', None)
    if rename is None:
        raise OSError(errno.ENOSYS, 'renameat2 unavailable')
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(source), -100, os.fsencode(destination), 1):
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))


def portable_requests(stage):
    """Make saved replay requests relative to the published directory (cwd)."""
    prefix = str(stage.resolve()) + os.sep
    def convert(value):
        if isinstance(value, str) and value.startswith(prefix):
            return value[len(prefix):]
        if isinstance(value, list):
            return [convert(item) for item in value]
        if isinstance(value, dict):
            return {key: convert(item) for key, item in value.items()}
        return value
    for path in [stage / 'child-source-manifest.json', *stage.rglob('infer-request-*.json'),
                 *stage.rglob('audit-request-*.json'), *stage.rglob('infer-imports-*.json'),
                 *stage.rglob('audit-imports-*.json')]:
        write(path, convert(json.loads(path.read_text())))


def run(config_path, config_sha256, output, *, worker_output=False):
    raise RuntimeError('legacy curve candidate disabled: mandatory selection audit gates are not integrated')
    raw = checked(config_path, config_sha256)
    config = json.loads(raw)
    if set(config) != {'version', 'source_dir', 'source_sha256', 'inputs', 'manifest_sha256',
                       'audit_sha256', 'curve'} or config['version'] != 'nl001-curve-candidate-v1':
        raise ValueError('invalid candidate configuration')
    if set(config['inputs']) != {'train', 'validation', 'manifest', 'audit'}:
        raise ValueError('only train, validation, manifest, and audit inputs are accepted')
    pins = config['source_sha256']
    if not isinstance(pins, dict) or not pins or '__init__.py' not in pins:
        raise ValueError('invalid source pins')
    for name, pin in pins.items():
        if Path(name).name != name or not name.endswith('.py') or not isinstance(pin, str) or len(pin) != 64:
            raise ValueError('invalid source pin')
    payloads = {}
    for name, item in config['inputs'].items():
        if set(item) != {'path', 'sha256'}:
            raise ValueError('invalid input pin')
        payloads[name] = checked(item['path'], item['sha256'])
    source = Path(config['source_dir'])
    source_bytes = {name: checked(source / name, pin) for name, pin in pins.items()}
    output = Path(output).absolute()
    if worker_output:
        if output.is_symlink() or not output.is_dir() or any(output.iterdir()):
            raise ValueError('worker output must be an empty real directory')
        final = output / 'completed'
    else:
        final = output
    if final.exists() or final.is_symlink():
        raise ValueError('output already exists')
    final.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.nl001-curve-', dir=final.parent))
    try:
        package = stage / 'source' / 'neural_lab'
        package.mkdir(parents=True)
        for name, data in source_bytes.items():
            (package / name).write_bytes(data)
        if any(name == 'neural_lab' or name.startswith('neural_lab.') for name in sys.modules):
            raise RuntimeError('package was imported before source verification')
        os.chdir(stage / 'source')
        sys.path.insert(0, str(stage / 'source'))
        os.environ['PYTHONPATH'] = str(stage / 'source')
        os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
        sys.dont_write_bytecode = True
        child_manifest = stage / 'child-source-manifest.json'
        write(child_manifest, {'package': str(package), 'sha256': pins})
        os.environ['NL001_CHILD_ATTESTATION_MANIFEST'] = str(child_manifest)
        from neural_lab.artifacts import ArtifactWriter
        from neural_lab.nl001_data import digest
        from neural_lab.nl001_learning_curve import run_curve
        from neural_lab.nl001_manifest import authenticate_split
        data = {name: json.loads(value) for name, value in payloads.items()}
        manifest, audit = data['manifest'], data['audit']
        if (digest(audit) != config['audit_sha256'] or audit.get('version') != 'nl001-corpus-audit-v2'
            or audit.get('regeneration') != 'exact match using current generator and checker'
            or audit.get('manifest') != manifest or audit.get('manifest_sha256') != config['manifest_sha256']
            or audit.get('corpus_sha256') != manifest.get('corpus_sha256')):
            raise ValueError('corpus audit does not match pinned manifest')
        comparison = audit.get('observed_view_audit')
        if (not isinstance(comparison, dict) or type(comparison.get('comparisons')) is not int
            or comparison['comparisons'] <= 0 or type(comparison.get('max_states_per_comparison')) is not int
            or comparison['max_states_per_comparison'] <= 0):
            raise ValueError('invalid corpus audit comparisons')
        for split in ('train', 'validation'):
            authenticate_split(data[split], split, manifest, config['manifest_sha256'])
        (stage / 'config.json').write_bytes(raw)
        for name, value in payloads.items():
            (stage / (name + '.json')).write_bytes(value)
        start = time.perf_counter()
        summary = run_curve(train=data['train'], validation=data['validation'], manifest=manifest,
                            manifest_sha256=config['manifest_sha256'], output=stage / 'curve', **config['curve'])
        writer = ArtifactWriter(stage)
        for name in ('config', 'train', 'validation', 'manifest', 'audit'):
            writer.register_file('datasets', stage / (name + '.json'), name, format='json')
        for seed, arms in summary['fits'].items():
            for arm, fit in arms.items():
                for step, entry in fit['training']['checkpoints'].items():
                    path = stage / 'curve' / (seed + '-' + arm) / entry['path']
                    saved = writer.register_file('checkpoints', path, seed + '-' + arm + '-' + step,
                                                 format='torch-pt', arm=arm, seed=int(seed), step=int(step))
                    if saved['sha256'] != entry['sha256']:
                        raise ValueError('checkpoint changed')
        imported = {}
        for name, module in tuple(sys.modules.items()):
            if name == 'neural_lab' or name.startswith('neural_lab.'):
                path = Path(module.__file__).resolve(strict=True)
                if path.parent != package.resolve() or path.name not in pins or sha(path.read_bytes()) != pins[path.name]:
                    raise ValueError('source import or digest mismatch: ' + name)
                imported[name] = path.name
        for name, pin in pins.items():
            checked(package / name, pin)
        write(stage / 'environment.json', {'python': sys.version, 'platform': platform.platform(),
              'imported_modules': imported, 'limitation': 'point-in-time self-report; runtime and source remain writable'})
        write(stage / 'result.json', {'metrics': {'wall_seconds': time.perf_counter() - start},
              'summary': 'Validation-only curve; see curve/summary.json. CPU inference; no held-out test evaluation.'})
        portable_requests(stage)
        inventory = []
        for path in sorted(stage.rglob('*')):
            if path.is_symlink():
                raise ValueError('symlink in output')
            if path.is_file():
                content = path.read_bytes()
                inventory.append({'path': path.relative_to(stage).as_posix(), 'bytes': len(content), 'sha256': sha(content)})
        write(stage / 'output-inventory.json', inventory)
        (stage / '.nl001-calibration-marker-pending').write_text(sha((stage / 'output-inventory.json').read_bytes()) + '\n')
        if final.exists() or final.is_symlink():
            raise ValueError('output appeared during execution')
        rename_noreplace(stage, final)
        if worker_output:
            # Publish the worker result only after the complete directory.
            os.link(final / 'result.json', output / 'result.json', follow_symlinks=False)
        # The marker is the last publication step, after the worker result.
        os.replace(final / '.nl001-calibration-marker-pending', final / 'nl001-calibration-complete.txt')
    except BaseException:
        # Retain partial evidence without a completion marker for diagnosis.
        try:
            # After rename, diagnostics live at final; never touch a competing output.
            diagnostic = final if not stage.exists() else stage
            (diagnostic / 'nl001-calibration-complete.txt').unlink(missing_ok=True)
            (diagnostic / 'failure.txt').write_text(traceback.format_exc())
        except OSError:
            pass
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--worker-output', action='store_true', help='publish into an existing empty worker output directory')
    args = parser.parse_args()
    run(args.config, args.config_sha256, args.output, worker_output=args.worker_output)


if __name__ == '__main__':
    main()
