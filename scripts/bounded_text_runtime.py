"""Process-local conservative MLX inference policy after the GPU panic.

This is a mitigation experiment, not a guarantee against kernel bugs. Never
changes site-packages or calls set_wired_limit (including to restore a limit).
"""
import contextlib
import fcntl
import importlib
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import time


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def swap_bytes():
    value = subprocess.check_output(['sysctl', '-n', 'vm.swapusage'], text=True)
    match = re.search(r'used = ([\d.]+)([MG])', value)
    if match is None:
        raise RuntimeError('Unrecognized swap usage; cannot apply resource guard')
    return int(float(match[1]) * (1024 ** (2 if match[2] == 'M' else 3)))


class BoundedTextRuntime:
    def __init__(self, output, memory_limit=3_800_000_000, max_seconds=600, pressure_wait_seconds=0):
        self.output = Path(output)
        self.memory_limit = memory_limit
        self.max_seconds = max_seconds
        self.pressure_wait_seconds = pressure_wait_seconds
        if not 0 <= pressure_wait_seconds <= 10:
            raise ValueError('Pressure wait must be between zero and ten seconds')

    def event(self, phase, **details):
        with (self.output / 'runtime.jsonl').open('a') as f:
            f.write(json.dumps(dict(time=time.time(), phase=phase, **details))+'\n')
            f.flush()
            os.fsync(f.fileno())

    def check(self):
        elapsed = time.monotonic() - self.started
        # On macOS ru_maxrss is bytes. Peak, rather than current, is deliberate.
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        swap = swap_bytes()
        pressure = int(subprocess.check_output(
            ['sysctl', '-n', 'kern.memorystatus_vm_pressure_level'], text=True).strip())
        if (pressure == 2 and self.pressure_wait_seconds and elapsed <= self.max_seconds
                and rss <= 4_600_000_000 and swap-self.initial_swap <= 512*1024**2):
            self.event('pressure_pause_begin', memory_pressure_level=pressure)
            self.mx.synchronize(self.generate.generation_stream)
            self.mx.synchronize()
            self.original_clear()
            deadline = time.monotonic()+self.pressure_wait_seconds
            while True:
                elapsed = time.monotonic()-self.started
                rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                swap = swap_bytes()
                pressure = int(subprocess.check_output(
                    ['sysctl', '-n', 'kern.memorystatus_vm_pressure_level'], text=True))
                if (pressure != 2 or time.monotonic() >= deadline or elapsed > self.max_seconds
                        or rss > 4_600_000_000 or swap-self.initial_swap > 512*1024**2):
                    break
                time.sleep(.2)
            self.event('pressure_pause_end', memory_pressure_level=pressure,
                       resumed=pressure == 1, elapsed_seconds=elapsed)
        metrics = dict(elapsed_seconds=elapsed, peak_rss_bytes=rss, swap_bytes=swap,
                       memory_pressure_level=pressure, mlx_peak_bytes=self.mx.get_peak_memory(),
                       mlx_active_bytes=self.mx.get_active_memory())
        reason = None
        if elapsed > self.max_seconds:
            reason = 'elapsed-time limit'
        elif rss > 4_600_000_000:
            reason = 'process-RSS limit'
        elif pressure != 1:
            reason = 'system memory pressure is not normal'
        elif swap - self.initial_swap > 512 * 1024**2:
            reason = 'swap-growth limit'
        if reason:
            self.event('resource_guard_stop', reason=reason, **metrics)
            raise RuntimeError('Bounded runtime '+reason)
        return metrics

    def __enter__(self):
        self.lock = open(Path(__file__).resolve().parents[1] / 'results' / '.text-model.lock', 'a')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.started = time.monotonic()
        self.initial_swap = swap_bytes()
        self.event('import_begin', initial_swap_bytes=self.initial_swap)
        self.mx = importlib.import_module('mlx.core')
        self.generate = importlib.import_module('mlx_lm.generate')
        self.original_wired = self.generate.wired_limit
        self.original_set_wired = self.mx.set_wired_limit
        self.original_clear = self.mx.clear_cache

        def reject_wiring(*args, **kwargs):
            raise RuntimeError('Unexpected set_wired_limit in unwired experiment')

        @contextlib.contextmanager
        def unwired(model, streams=None):
            try:
                yield
            finally:
                for stream in streams or [self.mx.default_stream(self.mx.default_device())]:
                    self.mx.synchronize(stream)

        self.generate.wired_limit = unwired
        self.mx.set_wired_limit = reject_wiring
        # Avoid library cache eviction during asynchronous generation. Perform
        # one explicit synchronized eviction at a completed response boundary.
        self.mx.clear_cache = lambda: None
        self.mx.set_memory_limit(self.memory_limit)
        self.mx.set_cache_limit(32_000_000)
        self.event('runtime_ready', memory_limit=self.memory_limit,
                   policy='no wiring; deferred cache eviction; one process; bounded requests')
        return self

    def response_done(self):
        self.mx.synchronize(self.generate.generation_stream)
        self.mx.synchronize()
        self.original_clear()
        metrics = self.check()
        metrics['mlx_peak_bytes'] = self.mx.get_peak_memory()
        return metrics

    def settle_after_load(self, seconds=10):
        """Pause at a synchronized load boundary; never compute under warning.

        Short-lived warning pressure during weight loading can clear without
        further work. Persistent warning, critical pressure, RSS or swap limits
        still stop the run. Training/generation checks remain immediate.
        """
        self.mx.synchronize(self.generate.generation_stream)
        self.mx.synchronize()
        self.original_clear()
        deadline = time.monotonic() + seconds
        while True:
            pressure = int(subprocess.check_output(
                ['sysctl', '-n', 'kern.memorystatus_vm_pressure_level'], text=True))
            if (pressure != 2 or time.monotonic() >= deadline
                    or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > 4_600_000_000
                    or swap_bytes()-self.initial_swap > 512*1024**2
                    or time.monotonic()-self.started > self.max_seconds):
                return self.check()
            self.event('load_settle_wait', memory_pressure_level=pressure)
            time.sleep(1)

    def __exit__(self, kind, error, traceback):
        try:
            self.mx.synchronize(self.generate.generation_stream)
            self.mx.synchronize()
            self.event('runtime_exit', error=None if error is None else str(error))
        finally:
            self.generate.wired_limit = self.original_wired
            self.mx.set_wired_limit = self.original_set_wired
            self.mx.clear_cache = self.original_clear
            self.lock.close()
