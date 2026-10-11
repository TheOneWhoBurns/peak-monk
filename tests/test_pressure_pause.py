"""Pressure recovery cannot permit compute while warning or bypass other guards."""
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import bounded_text_runtime as runtime


def policy(monkeypatch, tmp_path, pressures, wait=1, swap=0, elapsed=0):
    clock = [elapsed]
    monkeypatch.setattr(runtime.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(runtime.time, 'sleep', lambda s: clock.__setitem__(0, clock[0]+s))
    pressure = iter(pressures)
    last = [pressures[-1]]
    def read(*a, **k):
        last[0] = next(pressure, last[0])
        return str(last[0])
    monkeypatch.setattr(runtime.subprocess, 'check_output', read)
    monkeypatch.setattr(runtime, 'swap_bytes', lambda: swap)
    monkeypatch.setattr(runtime.resource, 'getrusage', lambda _: SimpleNamespace(ru_maxrss=100))
    p = runtime.BoundedTextRuntime(tmp_path, max_seconds=10, pressure_wait_seconds=wait)
    p.started = p.initial_swap = 0
    calls = []
    p.mx = SimpleNamespace(synchronize=lambda *a: calls.append('sync'),
                          get_peak_memory=lambda: 100, get_active_memory=lambda: 100)
    p.generate = SimpleNamespace(generation_stream=None)
    p.original_clear = lambda: calls.append('clear')
    return p, calls


def test_warning_must_recover_before_return(monkeypatch, tmp_path):
    p, calls = policy(monkeypatch, tmp_path, [2, 2, 1])
    assert p.check()['memory_pressure_level'] == 1
    assert calls == ['sync', 'sync', 'clear']


@pytest.mark.parametrize('pressures,swap,elapsed', [([2],0,0), ([2,4],0,0),
    ([4],0,0), ([2],600*1024**2,0), ([2],0,11)])
def test_warning_timeout_critical_swap_and_deadline_stop(monkeypatch, tmp_path, pressures, swap, elapsed):
    p, _ = policy(monkeypatch, tmp_path, pressures, swap=swap, elapsed=elapsed)
    with pytest.raises(RuntimeError, match='Bounded runtime'):
        p.check()


def test_default_policy_still_stops_immediately(monkeypatch, tmp_path):
    p, calls = policy(monkeypatch, tmp_path, [2,1], wait=0)
    with pytest.raises(RuntimeError):
        p.check()
    assert calls == []
