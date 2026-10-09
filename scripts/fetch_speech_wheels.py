"""Fetch resolver-pinned public PyPI wheels with bounded ranges and hash checks.

Works around very slow streaming wheel downloads on this connection. Resolution
remains uv's job; this does not guess versions or execute downloaded source.
"""
import concurrent.futures as cf
import hashlib
import json
import os
from pathlib import Path
import subprocess
import urllib.request
from packaging.utils import parse_wheel_filename

dest = Path('vendor/wheels'); dest.mkdir(parents=True, exist_ok=True)
tags = json.loads(subprocess.check_output(['.venv-speech/bin/python','-c',
    'import json;from pip._vendor.packaging.tags import sys_tags;print(json.dumps([str(t) for t in sys_tags()]))']))
priority = {t:i for i,t in enumerate(tags)}
packages = [line.strip().split('==') for line in Path('requirements-speech.txt').read_text().splitlines()
            if line and not line[0].isspace() and not line.startswith('#')]


def metadata(pair):
    name, version = pair
    with urllib.request.urlopen(f'https://pypi.org/pypi/{name}/{version}/json', timeout=30) as r:
        meta = json.load(r)
    options = []
    for item in meta['urls']:
        if item['filename'].endswith('.whl'):
            *_, wheel_tags = parse_wheel_filename(item['filename'])
            scores = [priority[str(t)] for t in wheel_tags if str(t) in priority]
            if scores:
                options.append((min(scores), item))
    if not options:
        raise RuntimeError(f'No binary wheel for {name}=={version}')
    return min(options, key=lambda x:x[0])[1]


with cf.ThreadPoolExecutor(max_workers=8) as pool:
    assets = list(pool.map(metadata, packages))
jobs = []
handles = {}
for asset in assets:
    path = dest/asset['filename']
    if path.exists():
        with path.open('rb') as f:
            if hashlib.file_digest(f,'sha256').hexdigest() == asset['digests']['sha256']:
                continue
    fd = os.open(str(path)+'.partial', os.O_RDWR|os.O_CREAT|os.O_TRUNC, 0o644)
    handles[asset['filename']] = fd
    # Moderate chunks avoid long stalled streams while limiting request overhead.
    size = asset['size']; chunk = 256 * 1024
    jobs.extend((asset, lo, min(lo+chunk-1,size-1)) for lo in range(0,size,chunk))


def fetch(job):
    asset, lo, hi = job
    for retry in range(3):
        try:
            req = urllib.request.Request(asset['url'], headers={'Range':f'bytes={lo}-{hi}'})
            with urllib.request.urlopen(req, timeout=30) as r:
                assert r.status == 206
                assert r.headers['Content-Range'] == f"bytes {lo}-{hi}/{asset['size']}"
                data = r.read()
            assert len(data) == hi-lo+1
            assert os.pwrite(handles[asset['filename']], data, lo) == len(data)
            return
        except Exception:
            if retry == 2:
                raise


try:
    with cf.ThreadPoolExecutor(max_workers=16) as pool:
        for i,_ in enumerate(pool.map(fetch, jobs)):
            if i % 32 == 0:
                print('ranges',i+1,'/',len(jobs),flush=True)
finally:
    for fd in handles.values():
        os.close(fd)
for asset in assets:
    path = dest/asset['filename']
    if asset['filename'] in handles:
        partial = Path(str(path)+'.partial')
        with partial.open('rb') as f:
            assert hashlib.file_digest(f,'sha256').hexdigest() == asset['digests']['sha256']
        partial.replace(path)
print('Verified',len(assets),'pinned wheels',flush=True)
