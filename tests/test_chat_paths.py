import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from chat import resolve_inputs


def fixture_paths(tmp_path):
    base = tmp_path / 'models' / 'base'
    adapter = tmp_path / 'adapters' / 'candidate'
    base.mkdir(parents=True)
    adapter.mkdir(parents=True)
    (adapter / 'adapter_config.json').write_text(json.dumps({'model': 'models/base'}))
    (adapter / 'adapters.safetensors').touch()
    return base, adapter


def test_uses_recorded_base(tmp_path):
    base, adapter = fixture_paths(tmp_path)
    assert resolve_inputs('adapters/candidate', root=tmp_path) == (base, adapter)


def test_rejects_same_architecture_wrong_base_path(tmp_path):
    fixture_paths(tmp_path)
    (tmp_path / 'models' / 'instruct').mkdir()
    with pytest.raises(ValueError, match='trained on'):
        resolve_inputs('adapters/candidate', 'models/instruct', root=tmp_path)


def test_accepts_alias_to_the_same_base(tmp_path):
    base, adapter = fixture_paths(tmp_path)
    alias = tmp_path / 'alias'
    alias.symlink_to(base, target_is_directory=True)
    assert resolve_inputs(adapter, alias, root=tmp_path) == (base, adapter)


def test_requires_actual_adapter_weights(tmp_path):
    _, adapter = fixture_paths(tmp_path)
    (adapter / 'adapters.safetensors').unlink()
    with pytest.raises(ValueError, match='weights'):
        resolve_inputs(adapter, root=tmp_path)
