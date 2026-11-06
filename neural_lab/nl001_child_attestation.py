"""Run an NL-001 child module and record point-in-time package import origins.

This is diagnostic evidence, not a security boundary or remote attestation.
"""
import hashlib
import json
import os
import runpy
import sys
from pathlib import Path


def main():
    manifest_path, report_path, target, *args = sys.argv[1:]
    manifest = json.loads(Path(manifest_path).read_text())
    root = Path(manifest['package']).resolve(strict=True)
    pins = manifest['sha256']
    if target not in ('neural_lab.nl001_selection_runner', 'neural_lab.nl001_prediction_audit'):
        raise ValueError('unexpected child target')
    sys.argv = [target, *args]
    runpy.run_module(target, run_name='__main__', alter_sys=True)
    imported = {}
    # runpy removes __main__ on return; independently check its target file too.
    names = {name: module for name, module in tuple(sys.modules.items())
             if name == 'neural_lab' or name.startswith('neural_lab.')}
    for name, module in names.items():
        path = Path(module.__file__).resolve(strict=True)
        if path.parent != root or path.name not in pins:
            raise ValueError('child import outside pinned package: ' + name)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != pins[path.name]:
            raise ValueError('child import digest mismatch: ' + name)
        imported[name] = {'path': str(path), 'sha256': actual}
    target_path = root / (target.rsplit('.', 1)[1] + '.py')
    target_hash = hashlib.sha256(target_path.read_bytes()).hexdigest()
    if target_hash != pins[target_path.name]:
        raise ValueError('child target digest mismatch')
    report = {'target': target, 'target_path': str(target_path), 'target_sha256': target_hash,
              'imported_modules': imported, 'pid': os.getpid(), 'python': sys.executable,
              'limitation': 'point-in-time self-report; writable files and runtime are not trusted'}
    with Path(report_path).open('x') as stream:
        json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
