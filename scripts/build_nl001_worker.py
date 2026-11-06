#!/usr/bin/env python3
"""Build a no-argument NL-001 worker script from independently pinned inputs.

The resulting file contains source and train/validation data; protect it as data.
Its own SHA-256 must be retained outside the worker. This builder does not queue
or execute the script, and does not establish the authenticity of supplied pins.
"""
import argparse
import base64
import lzma
import hashlib
import json
from pathlib import Path

MAX_SCRIPT_BYTES = 65_536  # Keep in sync with neural_lab.queue.MAX_SCRIPT_BYTES.


def sha(data):
    return hashlib.sha256(data).hexdigest()


def build(config_path, config_pin, candidate_path, output):
    raise RuntimeError('legacy curve worker builder disabled: mandatory selection audit gates are not integrated')
    raw = Path(config_path).read_bytes()
    if sha(raw) != config_pin:
        raise ValueError('configuration digest mismatch')
    config = json.loads(raw)
    regen = config.get('regeneration')
    required = {'version', 'source_dir', 'source_sha256', 'inputs', 'manifest_sha256', 'audit_sha256', 'curve'}
    if set(config) != required | ({'regeneration'} if regen is not None else set()) or config['version'] != 'nl001-curve-candidate-v1':
        raise ValueError('invalid candidate configuration')
    if set(config['inputs']) != {'train', 'validation', 'manifest', 'audit'}:
        raise ValueError('invalid input set')
    if regen is not None and (not isinstance(regen, dict) or set(regen) != {'parameters', 'corpus_sha256'} or not isinstance(regen['parameters'], dict)):
        raise ValueError('invalid regeneration specification')
    pins = config['source_sha256']
    if not isinstance(pins, dict) or '__init__.py' not in pins:
        raise ValueError('invalid source pins')
    files = {}
    for name, pin in pins.items():
        if not isinstance(name, str) or Path(name).name != name or not name.endswith('.py'):
            raise ValueError('invalid source name')
        data = (Path(config['source_dir']) / name).read_bytes()
        if sha(data) != pin:
            raise ValueError('source digest mismatch: ' + name)
        files['source/' + name] = data
    for name, item in config['inputs'].items():
        if not isinstance(item, dict) or set(item) != ({'sha256'} if regen is not None and name in ('train', 'validation') else {'path', 'sha256'}):
            raise ValueError('invalid input pin')
        if regen is not None and name in ('train', 'validation'):
            continue
        data = Path(item['path']).read_bytes()
        if sha(data) != item['sha256']:
            raise ValueError('input digest mismatch: ' + name)
        files['inputs/' + name + '.json'] = data
    candidate = Path(candidate_path).read_bytes()
    # Length-delimited bytes avoid base64 expansion inside the compressed stream.
    members = {**files, 'candidate': candidate}
    metadata = {'config': config, 'original_config_sha256': config_pin,
                'lengths': {name: len(data) for name, data in sorted(members.items())},
                'candidate_sha256': sha(candidate)}
    header = json.dumps(metadata, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    archive = len(header).to_bytes(8, 'big') + header + b''.join(members[name] for name in sorted(members))
    packed = lzma.compress(archive, format=lzma.FORMAT_XZ, preset=9)
    payload = base64.b85encode(packed).decode('ascii')
    script = '''"""Generated NL-001 worker; retain its externally verified SHA-256."""
import base64, hashlib, json, lzma, os, subprocess, sys, tempfile
from pathlib import Path
PAYLOAD = %r
PAYLOAD_SHA256 = %r

def main():
    packed = base64.b85decode(PAYLOAD)
    if hashlib.sha256(packed).hexdigest() != PAYLOAD_SHA256: raise ValueError('bundle mismatch')
    archive = lzma.decompress(packed)
    header_length = int.from_bytes(archive[:8], 'big')
    bundle = json.loads(archive[8:8 + header_length])
    offset = 8 + header_length
    members = {}
    for name, length in sorted(bundle['lengths'].items()):
        members[name] = archive[offset:offset + length]
        if len(members[name]) != length: raise ValueError('truncated bundle')
        offset += length
    if offset != len(archive): raise ValueError('invalid bundle length')
    with tempfile.TemporaryDirectory(prefix='nl001-input-', dir='/tmp') as tmp:
        root = Path(tmp)
        for name, data in members.items():
            if name == 'candidate': continue
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        config = bundle['config']
        regen = config.pop('regeneration', None)
        if regen is not None:
            # Isolate imports from the candidate's independently verified package.
            code = r"""import hashlib,json,sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from neural_lab.nl001_corpus import make_sampled_corpus
spec=json.loads(Path(sys.argv[2]).read_bytes())
corpus=make_sampled_corpus(**spec['parameters'])
if corpus['sha256'] != spec['corpus_sha256']: raise ValueError('regenerated corpus digest mismatch')
for name in ('train','validation'):
 data=json.dumps(corpus['splits'][name],sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()
 if hashlib.sha256(data).hexdigest()!=spec['input_sha256'][name]: raise ValueError('regenerated split digest mismatch: '+name)
 (Path(sys.argv[3])/(name+'.json')).write_bytes(data)
"""
            package_root = root / 'package'; package_root.mkdir()
            (package_root / 'neural_lab').symlink_to(root / 'source', target_is_directory=True)
            spec = dict(regen, input_sha256={n: config['inputs'][n]['sha256'] for n in ('train', 'validation')})
            spec_path = root / 'regeneration.json'; spec_path.write_text(json.dumps(spec))
            subprocess.run([sys.executable, '-I', '-B', '-c', code, str(package_root), str(spec_path), str(root / 'inputs')], check=True)
        config['source_dir'] = str(root / 'source')
        for name, item in config['inputs'].items(): item['path'] = str(root / 'inputs' / (name + '.json'))
        raw = json.dumps(config, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        config_path = root / 'config.json'; config_path.write_bytes(raw)
        candidate = members['candidate']
        if hashlib.sha256(candidate).hexdigest() != bundle['candidate_sha256']: raise ValueError('candidate mismatch')
        namespace = {'__name__': 'nl001_embedded_candidate', '__file__': '<embedded-candidate>'}
        exec(compile(candidate, '<embedded-candidate>', 'exec'), namespace)
        namespace['run'](config_path, hashlib.sha256(raw).hexdigest(), Path(os.environ.get('NL001_WORKER_OUTPUT', '/output')), worker_output=True)
if __name__ == '__main__': main()
''' % (payload, sha(packed))
    if len(script.encode('utf-8')) > MAX_SCRIPT_BYTES:
        raise ValueError('worker exceeds queue limit: %d > %d bytes' % (len(script.encode('utf-8')), MAX_SCRIPT_BYTES))
    with Path(output).open('x', encoding='utf-8') as stream:
        stream.write(script)
    return {'worker_sha256': sha(script.encode()), 'candidate_sha256': sha(candidate), 'original_config_sha256': config_pin}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--candidate', default=str(Path(__file__).with_name('nl001_curve_candidate.py')))
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.config, args.config_sha256, args.candidate, args.output), sort_keys=True))


if __name__ == '__main__':
    main()
