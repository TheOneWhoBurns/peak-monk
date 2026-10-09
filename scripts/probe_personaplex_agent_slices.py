"""Compare exact native traces before/after releasing unused depth-tail weights.

Runs only on a fully hash-verified artifact. It uses the pinned Edge generator,
whose agent sampling already stops after depth slice seven. Both traces include
identical stock-voice prefill and an empty role prompt. No character claim.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import mlx.core as mx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'vendor/personaplex-edge-source/src'))
from personaplex_edge.runtime import NativeRuntime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, default=Path('models/personaplex-edge'))
    parser.add_argument('--input', type=Path, default=Path('evaluation/audio-roleplay-v2/s01.npz'))
    parser.add_argument('--frames', type=int, default=128)
    parser.add_argument('--voice', default='NATM0')
    parser.add_argument('--seed', type=int, default=41)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a fresh output path')
    if args.frames < 3:
        raise ValueError('At least three frames are required')
    codes = np.load(args.input)['user_codes'][:, :args.frames]
    assert codes.shape == (8, args.frames) and np.all((codes >= 0) & (codes < 2048))
    mx.set_memory_limit(5_700_000_000)
    mx.set_cache_limit(32_000_000)
    recommended = mx.device_info()['max_recommended_working_set_size']
    mx.set_wired_limit(recommended)
    runtime = NativeRuntime(args.model, verify=True)
    assert len(runtime.model.depformer.slices) == 16

    def capture():
        runtime.generator.reset()
        gc.collect(); mx.clear_cache(); mx.reset_peak_memory()
        mx.random.seed(args.seed)
        start = time.monotonic()
        runtime.generator.prefill(runtime.artifact.voice_path(args.voice), runtime.text_tokenizer, '')
        prefill_seconds = time.monotonic() - start
        records = []
        start = time.monotonic()
        for i in range(args.frames):
            result = runtime.generator.step(mx.array(codes[:, i][None, :]))
            records.append(None if result is None else {
                'text': result.text_token,
                'audio': np.asarray(result.audio_tokens).astype(int).tolist()})
        return records, dict(prefill_seconds=prefill_seconds,
                             generation_seconds=time.monotonic()-start,
                             peak_bytes=mx.get_peak_memory(), active_bytes=mx.get_active_memory())

    full, full_metrics = capture()
    print('Full depth-weight trace complete', full_metrics, flush=True)
    runtime.generator.reset()
    runtime.model.depformer.slices = runtime.model.depformer.slices[:8]
    gc.collect(); mx.clear_cache()
    pruned, pruned_metrics = capture()
    differences = [i for i, (a, b) in enumerate(zip(full, pruned)) if a != b]
    report = dict(status='pass' if not differences else 'fail',
                  model=str(args.model), model_sha256=runtime.artifact.manifest['files']['model.q4.safetensors']['sha256'],
                  input=str(args.input), input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                  frames=args.frames, seed=args.seed, voice=args.voice, role_prompt='',
                  retained_depth_slices=8, source_depth_slices=16,
                  model_config='Unchanged; sixteen temporal audio inputs retained',
                  different_frames=differences, full=full_metrics, pruned=pruned_metrics,
                  working_set_recommendation=recommended,
                  active_memory_reduction_bytes=full_metrics['active_bytes']-pruned_metrics['active_bytes'],
                  scope='Exact emitted-token trace parity for this fixed prefix and pinned Edge generator. Not a PyTorch parity or character-quality test.',
                  full_trace=full, pruned_trace=pruned)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if not k.endswith('_trace')}, indent=2), flush=True)
    if differences:
        raise RuntimeError('Releasing the unused tail changed the native trace')


if __name__ == '__main__':
    main()
