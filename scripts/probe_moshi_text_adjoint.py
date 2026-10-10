"""One guarded, disposable text-only update to the actual native Moshi adapter."""
import argparse
import json
import math
from pathlib import Path
import tempfile
import time
from bounded_speech_runtime import BoundedSpeechRuntime
from bounded_text_runtime import atomic_json
from sample_dialogue_paths import sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--adapter-output', type=Path, required=True)
    parser.add_argument('--bos', choices=['sentencepiece', 'native'], default='native')
    args = parser.parse_args()
    if args.output.exists() or args.adapter_output.exists():
        raise ValueError('Preserve previous evidence')
    row_path = Path('data/moshi-text-probe.json')
    row = json.loads(row_path.read_text())
    assert row['split'] == 'train'
    weights = Path('models/moshi/model.q4.safetensors')
    parent = Path('adapters/speech-run08/checkpoint-0067.safetensors')
    parent_sha = sha(parent)
    assert sha(weights) == '7959d590e23c1ebc78cfa3501344a6ff331561aa0cadc4429b733b890bbc919c'
    assert parent_sha == 'aa02e8b1e91ad3f48dbf1c3dc8e7663a1c1f2c7494f57ed372bef554f7480a70'
    args.output.mkdir()
    args.adapter_output.mkdir(parents=True)
    tokenizer = Path('models/moshi/tokenizer_spm_32k_3.model')
    import sentencepiece as spm
    processor = spm.SentencePieceProcessor(model_file=str(tokenizer))
    initial = processor.bos_id() if args.bos == 'sentencepiece' else processor.get_piece_size()
    tokens = [initial] + processor.encode(row['text']) + [processor.eos_id()]
    assert 2 < len(tokens) <= 256 and min(tokens) >= 0
    sources = ['probe_moshi_text_adjoint.py', 'moshi_text_adjoint.py', 'moshi_weight_stream.py',
               'moshi_layerwise.py', 'moshi_disk_adjoint.py', 'moshi_training.py',
               'bounded_speech_runtime.py', 'bounded_text_runtime.py']
    atomic_json(args.output/'config.json', dict(parent=str(parent), parent_sha256=parent_sha,
        data_sha256=sha(row_path), tokenizer_sha256=sha(tokenizer), tokens=tokens,
        source_sha256={name:sha(Path(__file__).with_name(name)) for name in sources},
        serialization=f'{args.bos} initial token + original first-person prose + EOS; every token after initial supervised.',
        head_tokens=16, learning_rate=1e-5, scope='Feasibility only; no speech quality claim.'))
    with BoundedSpeechRuntime(args.output, memory_limit=1_400_000_000, max_seconds=600) as runtime:
        import mlx.core as mx
        from moshi_mlx.models import config_v0_1
        from moshi_text_adjoint import TextMoshiParts, disk_text_grad
        from moshi_disk_adjoint import disk_adamw_update
        parts = TextMoshiParts(weights, config_v0_1(), parent, rank=8, scale=2.)

        def boundary(phase, **details):
            metrics = runtime.response_done()
            runtime.event(phase, **details, **metrics)
            if metrics['mlx_peak_bytes'] > 1_500_000_000:
                raise RuntimeError('Text-only native probe exceeded declared1.5GB peak plan')

        start = time.monotonic()
        with tempfile.TemporaryDirectory(prefix='jun-native-text-') as temporary:
            root = Path(temporary)
            gradient = disk_text_grad(parts, mx.array([tokens]), mx.array([[0]+[1]*(len(tokens)-1)]),
                                     root/'gradient', head_tokens=16, boundary=boundary)
            assert math.isfinite(gradient['loss']) and math.isfinite(gradient['gradient_norm']) and gradient['gradient_norm'] > 0
            output = args.adapter_output/'candidate.safetensors'
            update = disk_adamw_update(parts, gradient, root/'optimizer', output,
                                      lr=1e-5, weight_decay=.01, clip_norm=1., expected_step=1)
            boundary('text_optimizer_complete')
        result = dict(status='pass', tokens=len(tokens), supervised_tokens=gradient['supervised_tokens'],
            loss=gradient['loss'], gradient_norm=gradient['gradient_norm'], update=update,
            seconds=time.monotonic()-start, metrics=runtime.response_done(),
            adapter=str(output), adapter_sha256=sha(output), parent_sha256=parent_sha,
            scope='One actual pretrained native-backbone text-only gradient/update. Disposable candidate; no audio replay, native generation, quality improvement or acceptance claim.')
        atomic_json(args.output/'result.json', result)
        print(json.dumps(result), flush=True)
    assert sha(parent) == parent_sha


if __name__ == '__main__':
    main()
