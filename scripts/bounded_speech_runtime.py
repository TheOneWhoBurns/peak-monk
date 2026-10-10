"""Serialized, unwired native-training probes with checks at each small phase."""
import fcntl,importlib,time
from pathlib import Path
from bounded_text_runtime import BoundedTextRuntime,swap_bytes

class BoundedSpeechRuntime(BoundedTextRuntime):
    def __enter__(self):
        self.lock=open(Path(__file__).resolve().parents[1]/'results'/'.text-model.lock','a')
        fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.started=time.monotonic();self.initial_swap=swap_bytes()
        self.mx=importlib.import_module('mlx.core')
        self.original_set_wired=self.mx.set_wired_limit;self.original_clear=self.mx.clear_cache
        def reject(*args,**kwargs):raise RuntimeError('Wiring prohibited in streamed speech probe')
        self.mx.set_wired_limit=reject;self.mx.clear_cache=lambda:None
        self.mx.set_memory_limit(self.memory_limit);self.mx.set_cache_limit(32_000_000)
        self.event('runtime_ready',initial_swap_bytes=self.initial_swap,memory_guideline_bytes=self.memory_limit,
                   policy='One model part at a time; no wiring; synchronized eviction and guards at phase boundaries')
        self.check();return self
    def response_done(self):
        self.mx.synchronize();self.original_clear()
        metrics=self.check()
        if metrics['mlx_peak_bytes']>3_200_000_000:
            self.event('resource_guard_stop',reason='streamed speech peak exceeds 3.2GB',**metrics)
            raise RuntimeError('Streamed speech peak exceeded resource plan')
        return metrics
    def __exit__(self,kind,error,traceback):
        try:
            self.mx.synchronize();self.event('runtime_exit',error=None if error is None else str(error))
        finally:
            self.mx.set_wired_limit=self.original_set_wired;self.mx.clear_cache=self.original_clear;self.lock.close()
