"""Fetch small pinned voice assets and reuse verified codec files locally.

Main weights are downloaded and verified separately. This never loads a model.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import urllib.request

REVISION = '9ef3ef71176182ce8570f970c5c592fd1bbf3e9b'
ROOT = Path('models/personaplex-edge')
SOURCE = Path('vendor/personaplex-edge-source')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    manifest = json.loads((SOURCE / 'manifest.json').read_text())
    ROOT.mkdir(parents=True, exist_ok=True)
    for name in ('config.json', 'manifest.json'):
        target = ROOT / name
        content = (SOURCE / name).read_bytes()
        if target.exists() and target.read_bytes() != content:
            raise ValueError(f'Preserve differing metadata: {target}')
        target.write_bytes(content)
    assert digest(ROOT / 'config.json') == manifest['files']['config.json']['sha256']
    for name in ('tokenizer-e351c8d8-checkpoint125.safetensors', 'tokenizer_spm_32k_3.model'):
        source, target = Path('models/moshi') / name, ROOT / name
        assert digest(source) == manifest['files'][name]['sha256']
        if not target.exists():
            os.link(source, target)
        assert digest(target) == manifest['files'][name]['sha256']

    def fetch(item):
        name, record = item
        path = Path(name)
        assert not path.is_absolute() and '..' not in path.parts
        target = ROOT / path
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            assert digest(target) == record['sha256']
            return name
        url = f'https://huggingface.co/Mayank022/personaplex-edge-q4/resolve/{REVISION}/{name}'
        with urllib.request.urlopen(url, timeout=30) as response:
            data = response.read(record['bytes'] + 1)
        assert len(data) == record['bytes']
        assert hashlib.sha256(data).hexdigest() == record['sha256']
        target.write_bytes(data)
        return name

    voices = [(name, row) for name, row in manifest['files'].items() if name.startswith('voices/')]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for name in pool.map(fetch, voices):
            print('Verified', name, flush=True)
    report = {'repo': 'Mayank022/personaplex-edge-q4', 'revision': REVISION,
              'voice_files_verified': len(voices), 'codec_reuse': 'Verified same-filesystem hard links',
              'main_weight_status': 'Separate range download; full SHA verification still required before loading',
              'status': 'Small assets verified; no model inference performed'}
    Path('results/personaplex-small-assets.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
