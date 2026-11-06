#!/usr/bin/env python3
"""Build the self-contained, mock-only no-argument worker diagnostic."""
import base64
import hashlib
from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = (root / 'scripts/nl001_curve_candidate.py').read_bytes()
encoded = base64.b64encode(source).decode('ascii')
script = '''"""Mock-only worker publication diagnostic. No training, downloads, or queue action."""
import base64, hashlib, json, os, tempfile
from pathlib import Path

CANDIDATE = base64.b64decode(%r)
CANDIDATE_SHA256 = %r

def sha(data): return hashlib.sha256(data).hexdigest()
def canonical(value): return sha(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode())
def main():
    if sha(CANDIDATE) != CANDIDATE_SHA256: raise ValueError('embedded candidate mismatch')
    output = Path(os.environ.get('NL001_DIAGNOSTIC_OUTPUT', '/output'))
    modules = {
        '__init__.py': '',
        'nl001_data.py': 'import hashlib,json\\ndef digest(x): return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode()).hexdigest()\\n',
        'nl001_manifest.py': 'def authenticate_split(*args): pass\\n',
        'nl001_learning_curve.py': 'from pathlib import Path\\ndef run_curve(*,output,**kwargs):\\n p=Path(output); p.mkdir(); (p/"summary.json").write_text("{\\\\"fixture\\\\":true}")\\n return {"fits": {}}\\n',
        'artifacts.py': 'import json\\nfrom pathlib import Path\\nclass ArtifactWriter:\\n def __init__(self, output): self.output=Path(output); self.items=[]\\n def register_file(self, category,path,label,**kw):\\n  self.items.append(label); (self.output/"artifacts.json").write_text(json.dumps({"fixture_labels":self.items})); return {}\\n',
    }
    with tempfile.TemporaryDirectory(prefix='nl001-worker-fixture-', dir='/tmp') as temp:
        root = Path(temp); source = root / 'source'; source.mkdir()
        pins = {}
        for name, text in modules.items():
            data = text.encode(); (source / name).write_bytes(data); pins[name] = sha(data)
        manifest = {'corpus_sha256': 'mock-fixture'}; manifest_pin = canonical(manifest)
        audit = {'version': 'nl001-corpus-audit-v2', 'regeneration': 'exact match using current generator and checker', 'manifest': manifest, 'manifest_sha256': manifest_pin, 'corpus_sha256': 'mock-fixture', 'observed_view_audit': {'comparisons': 1, 'max_states_per_comparison': 1}}
        inputs = {}
        for name, value in {'train': [], 'validation': [], 'manifest': manifest, 'audit': audit}.items():
            data = json.dumps(value).encode(); path = root / (name + '.json'); path.write_bytes(data)
            inputs[name] = {'path': str(path), 'sha256': sha(data)}
        config = {'version': 'nl001-curve-candidate-v1', 'source_dir': str(source), 'source_sha256': pins, 'inputs': inputs, 'manifest_sha256': manifest_pin, 'audit_sha256': canonical(audit), 'curve': {}}
        data = json.dumps(config).encode(); config_path = root / 'config.json'; config_path.write_bytes(data)
        namespace = {'__name__': 'embedded_candidate', '__file__': '<embedded-candidate>'}
        exec(compile(CANDIDATE, '<embedded-candidate>', 'exec'), namespace)
        original_write = namespace['write']
        def diagnostic_write(path, value):
            if path.name == 'result.json':
                value = {'metrics': {'fixture_completed': 1}, 'summary': 'Mock-only publication diagnostic: synthetic inputs and stubbed authentication, curve and artifact writer; no training, replay, CUDA or scientific result.'}
            original_write(path, value)
        namespace['write'] = diagnostic_write
        namespace['run'](config_path, sha(data), output, worker_output=True)
if __name__ == '__main__': main()
''' % (encoded, hashlib.sha256(source).hexdigest())
(root / 'jobs/drafts/nl001_curve_worker_diagnostic.py').write_text(script)
