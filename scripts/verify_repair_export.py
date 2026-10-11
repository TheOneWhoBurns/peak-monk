"""Compare a cached-suffix export reference against the reloaded full backbone."""
import argparse
import json
from pathlib import Path

from bounded_text_runtime import BoundedTextRuntime, atomic_json
from sample_dialogue_paths import sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--adapter', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--streamed', action='store_true')
    p.add_argument('--resident-layers', type=int, default=8)
    p.add_argument('--query-chunk-size', type=int, default=0)
    p.add_argument('--pressure-wait-seconds', type=float, default=0)
    a = p.parse_args()
    ref = json.loads(a.reference.read_text())
    if sha(a.adapter/'adapters.safetensors') != ref['adapter_sha256']:
        raise ValueError('Export differs from captured reference')
    if sha(ref['data']) != ref['data_sha256']:
        raise ValueError('Reference data changed')
    if a.output.exists():
        raise ValueError('Preserve earlier verification')
    a.output.mkdir(parents=True)
    row = [json.loads(s) for s in Path(ref['data']).read_text().splitlines()][ref['row_index']]
    with BoundedTextRuntime(a.output, memory_limit=1_800_000_000 if a.streamed else 3_400_000_000, max_seconds=300, pressure_wait_seconds=a.pressure_wait_seconds) as runtime:
        from mlx_lm import load
        from conversation_objective import encode_conversation
        from repair_export_score import selected_score
        model, tok = load(ref['model'], adapter_path=str(a.adapter), lazy=a.streamed)
        model.eval()
        if a.query_chunk_size:
            from qwen_chunked_attention import chunk_qwen_attention
            chunk_qwen_attention(model, a.query_chunk_size, include_eval=True)
        if a.streamed:
            from qwen_streamed_inference import install_streaming
            def boundary(index):
                metrics=runtime.check()
                if metrics['mlx_peak_bytes']>2_000_000_000:
                    raise RuntimeError('Streamed verification exceeded 2GB peak plan')
            install_streaming(model,ref['model'],resident_layers=a.resident_layers,boundary=boundary)
            import mlx.core as mx
            mx.eval(model.parameters())
            runtime.event('load_settled', **runtime.settle_after_load())
        tokens, mask = encode_conversation(row, tok)
        runtime.event('full_model_score_begin', tokens=len(tokens))
        actual = selected_score(model, tokens, mask, prefill_chunk=0 if a.streamed else 64,
                                layerwise=a.streamed, boundary=runtime.response_done)
        reference = ref['score']
        if actual['target_tokens'] != reference['target_tokens']:
            raise ValueError('Target alignment differs')
        difference = max(abs(a-b) for a,b in zip(actual['target_log_probs'],reference['target_log_probs'], strict=True))
        passed = difference < .02 and abs(actual['mean_loss']-reference['mean_loss']) < .005
        report = dict(status='pass' if passed else 'fail', maximum_logp_difference=difference,
            mean_loss_difference=abs(actual['mean_loss']-reference['mean_loss']), actual=actual, reference=ref,
            metrics=runtime.response_done(), streamed_weights=a.streamed, resident_layers=a.resident_layers if a.streamed else None, query_chunk_size=a.query_chunk_size,
            source_sha256={name:sha(Path(__file__).with_name(name)) for name in ['verify_repair_export.py','qwen_streamed_inference.py','repair_export_score.py','bounded_text_runtime.py']},
            scope='Actual saved standard adapter reloaded on full36-layer base versus cached frozen-prefix suffix scoring; BF16 tolerance. Streaming mode retains full unchunked temporal attention and reloads frozen weights between layers. No quality claim.')
        atomic_json(a.output/'result.json', report)
        print(json.dumps(report), flush=True)
        if not passed:
            raise RuntimeError('Export score does not reproduce suffix model')


if __name__ == '__main__':
    main()
