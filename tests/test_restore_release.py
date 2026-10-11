"""Restore the flat HF upload layout without downloading or loading a model."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile

import pytest

spec = importlib.util.spec_from_file_location('restore_release', Path(__file__).resolve().parents[1]/'scripts/restore_release.py')
restore = importlib.util.module_from_spec(spec)
spec.loader.exec_module(restore)


def test_restore_zipped_metadata(monkeypatch, tmp_path):
    published = tmp_path/'published'
    published.mkdir()
    base_weight = tmp_path/'base-weight'
    base_weight.write_bytes(b'example base weights')
    provenance = {'repo': 'fixture/base', 'revision': 'fixed', 'weight_sha256_verified': restore.digest(base_weight)}
    with zipfile.ZipFile(published/'base-metadata.zip', 'w') as archive:
        archive.writestr('base4b_provenance.json', json.dumps(provenance))
        archive.writestr('config.json', '{"restored": true}')
    (published/'adapters.safetensors').write_bytes(b'example adapter')
    (published/'adapter_config.json').write_text('{}')
    manifest = {'files': {p.name: restore.digest(p) for p in published.iterdir()}}
    (published/'release-manifest.json').write_text(json.dumps(manifest))
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(snapshot_download=lambda *a, **k: str(published), hf_hub_download=lambda *a, **k: str(base_weight)))
    monkeypatch.setattr(restore, 'ROOT', tmp_path/'restored')
    monkeypatch.setattr(sys, 'argv', ['restore_release', '--revision', 'fixture'])
    restore.main()
    assert (restore.ROOT/'models/base4b/config.json').read_text() == '{"restored": true}'
    assert (restore.ROOT/'adapters/run26-step0128/adapters.safetensors').read_bytes() == b'example adapter'
    (restore.ROOT/'models/base4b/config.json').write_text('local changes')
    with pytest.raises(ValueError, match='Refusing to overwrite'):
        restore.main()
