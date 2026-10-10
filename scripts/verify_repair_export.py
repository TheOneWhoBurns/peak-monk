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
    with BoundedTextRuntime(a.output, memory_limit=3_400_000_000, max_seconds=180) as runtime:
        from mlx_lm import load
        from conversation_objective import encode_conversation
        from repair_export_score import selected_score
        model, tok = load(ref['model'], adapter_path=str(a.adapter))
        model.eval()
        tokens, mask = encode_conversation(row, tok)
        runtime.event('full_model_score_begin', tokens=len(tokens))
        actual = selected_score(model, tokens, mask, prefill_chunk=64, boundary=runtime.response_done)
        reference = ref['score']
        if actual['target_tokens'] != reference['target_tokens']:
            raise ValueError('Target alignment differs')
        difference = max(abs(a-b) for a,b in zip(actual['target_log_probs'],reference['target_log_probs'], strict=True))
        passed = difference < .02 and abs(actual['mean_loss']-reference['mean_loss']) < .005
        report = dict(status='pass' if passed else 'fail', maximum_logp_difference=difference,
            mean_loss_difference=abs(actual['mean_loss']-reference['mean_loss']), actual=actual, reference=ref,
            metrics=runtime.response_done(), scope='Actual saved standard adapter reloaded on full36-layer base versus cached frozen-prefix suffix scoring; BF16 tolerance. No quality claim.')
        atomic_json(a.output/'result.json', report)
        print(json.dumps(report), flush=True)
        if not passed:
            raise RuntimeError('Export score does not reproduce suffix model')


if __name__ == '__main__':
    main()
