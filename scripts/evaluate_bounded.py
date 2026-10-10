"""Run the unchanged character loss and generation protocol with resource checks."""
import argparse
import json
from pathlib import Path
import sys

from bounded_text_runtime import BoundedTextRuntime, atomic_json
from sample_dialogue_paths import sha


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--runtime-log', type=Path, required=True)
    p.add_argument('--max-seconds', type=int, default=600)
    a, remaining = p.parse_known_args()
    if '--output' not in remaining:
        raise ValueError('Evaluation output required')
    result = Path(remaining[remaining.index('--output')+1])
    if result.exists() or a.runtime_log.exists():
        raise ValueError('Preserve previous captures')
    adapter = Path(remaining[remaining.index('--adapter')+1]) if '--adapter' in remaining else None
    adapter_sha = sha(adapter/'adapters.safetensors') if adapter else None
    a.runtime_log.mkdir(parents=True)
    atomic_json(a.runtime_log/'config.json', dict(arguments=remaining, adapter_sha256=adapter_sha,
        source_sha256={n:sha(Path(__file__).with_name(n)) for n in
                       ['evaluate_bounded.py','evaluate.py','assistant_objective.py','bounded_text_runtime.py']},
        scope='Same loss, tokenization, sampling and prefill defaults; only resource lifecycle and checks change'))
    with BoundedTextRuntime(a.runtime_log, memory_limit=3_400_000_000, max_seconds=a.max_seconds) as runtime:
        import mlx.core as mx
        import evaluate as implementation
        original_loss = implementation.assistant_loss
        original_stream = implementation.stream_generate
        original_load = implementation.load

        def guarded_load(*args, **kwargs):
            loaded = original_load(*args, **kwargs)
            runtime.event('load_settled', **runtime.settle_after_load())
            return loaded

        def guarded_loss(*args, **kwargs):
            runtime.check()
            value = original_loss(*args, **kwargs)
            mx.eval(value)
            runtime.response_done()
            return value

        def guarded_stream(*args, **kwargs):
            runtime.check()
            generator = original_stream(*args, **kwargs)
            try:
                for chunk in generator:
                    if chunk.generation_tokens % 16 == 0:
                        runtime.check()
                    yield chunk
            finally:
                generator.close()
                runtime.event('response_end', **runtime.response_done())

        implementation.assistant_loss = guarded_loss
        implementation.stream_generate = guarded_stream
        implementation.load = guarded_load
        sys.argv = ['evaluate.py', *remaining]
        implementation.main()
    if adapter and sha(adapter/'adapters.safetensors') != adapter_sha:
        raise RuntimeError('Adapter changed during capture')
    captured = json.loads(result.read_text())
    captured.update(adapter_sha256=adapter_sha, runtime_log=str(a.runtime_log))
    atomic_json(result, captured)
    atomic_json(a.runtime_log/'completion.json', dict(status='capture complete', result=str(result),
                                                     result_sha256=sha(result)))


if __name__ == '__main__':
    main()
