"""Diagnose plain-text loss and audio-padding prior in the native backbone.

Read-only, streamed teacher forcing. Conditional lexical CE is a diagnostic,
not a substitute for the full loss or a speech-quality measure.
"""
import argparse
import json
from pathlib import Path
import time
from bounded_speech_runtime import BoundedSpeechRuntime
from bounded_text_runtime import atomic_json
from sample_dialogue_paths import sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapter', type=Path)
    parser.add_argument('--bos', choices=['sentencepiece','native'], default='sentencepiece')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    row_path = Path('data/moshi-text-probe.json')
    row = json.loads(row_path.read_text())
    tokenizer = Path('models/moshi/tokenizer_spm_32k_3.model')
    import sentencepiece as spm
    processor = spm.SentencePieceProcessor(model_file=str(tokenizer))
    # Native generation reserves text_out_vocab_size as its initial token.
    # Compare that known convention with ordinary SentencePiece BOS explicitly.
    initial = processor.bos_id() if args.bos == 'sentencepiece' else processor.get_piece_size()
    tokens = [initial] + processor.encode(row['text']) + [processor.eos_id()]
    assert 2 < len(tokens) <= 256 and min(tokens) >= 0
    weights = Path('models/moshi/model.q4.safetensors')
    assert sha(weights) == '7959d590e23c1ebc78cfa3501344a6ff331561aa0cadc4429b733b890bbc919c'
    atomic_json(args.output/'config.json', dict(adapter=str(args.adapter) if args.adapter else None,
        adapter_sha256=sha(args.adapter) if args.adapter else None,
        data_sha256=sha(row_path), tokenizer_sha256=sha(tokenizer), tokens=tokens,
        source_sha256={name:sha(Path(__file__).with_name(name)) for name in
            ['score_moshi_text.py','moshi_text_adjoint.py','moshi_weight_stream.py','moshi_training.py']},
        head_tokens=16, diagnostic_excluded_ids=[0,3], initial_token_mode=args.bos,
        scope='Read-only plain-text teacher forcing. Compare full CE with conditional lexical CE excluding native EPAD/PAD; no acceptance claim.'))
    with BoundedSpeechRuntime(args.output, memory_limit=1_400_000_000, max_seconds=600) as runtime:
        import mlx.core as mx
        from moshi_mlx.models import config_v0_1
        from moshi_text_adjoint import TextMoshiParts
        from moshi_training import layer_forward
        parts = TextMoshiParts(weights, config_v0_1(), args.adapter, rank=8, scale=2.)
        start = time.monotonic()
        module = parts.load('inputs')
        hidden = module(mx.array([tokens[:-1]]))
        mx.eval(hidden)
        del module
        runtime.response_done()
        for index in range(parts.cfg.transformer.num_layers):
            module = parts.load(index)
            hidden = layer_forward(module, hidden)[0]
            mx.eval(hidden)
            del module
            metrics = runtime.response_done()
            if metrics['mlx_peak_bytes'] > 1_500_000_000:
                raise RuntimeError('Text scoring exceeded declared1.5GB peak plan')
        heads = parts.load('heads')
        hidden = heads.out_norm(hidden)
        mx.eval(hidden)
        rows = []
        for offset in range(0, hidden.shape[1], 16):
            end = min(offset+16, hidden.shape[1])
            logits = heads.text_linear(hidden[:, offset:end]).astype(mx.float32)[0]
            normalizer = mx.logsumexp(logits, axis=-1)
            lexical = mx.where((mx.arange(logits.shape[-1]) == 0) | (mx.arange(logits.shape[-1]) == 3), -mx.inf, logits)
            lexical_normalizer = mx.logsumexp(lexical, axis=-1)
            target = mx.array(tokens[offset+1:end+1])
            target_logits = mx.take_along_axis(logits, target[:,None], axis=-1)[:,0]
            ce = normalizer-target_logits
            conditional = lexical_normalizer-target_logits
            padding_mass = mx.exp(logits[:,0]-normalizer)+mx.exp(logits[:,3]-normalizer)
            top = mx.argmax(logits, axis=-1)
            mx.eval(ce, conditional, padding_mass, top)
            for local, token in enumerate(target.tolist()):
                rows.append(dict(position=offset+local+1, token=token, piece=processor.id_to_piece(token),
                    ce=ce[local].item(), lexical_ce=conditional[local].item(),
                    padding_probability=padding_mass[local].item(), top_token=top[local].item()))
            runtime.response_done()
        result = dict(status='complete', target_tokens=len(rows), full_ce=sum(r['ce'] for r in rows)/len(rows),
            lexical_ce=sum(r['lexical_ce'] for r in rows)/len(rows),
            mean_padding_probability=sum(r['padding_probability'] for r in rows)/len(rows),
            argmax_padding_count=sum(r['top_token'] in (0,3) for r in rows),
            seconds=time.monotonic()-start, metrics=runtime.response_done(), tokens=rows,
            scope='Single authored training prose sample, teacher forced; neither held-out text ability nor native speech quality.')
        atomic_json(args.output/'result.json', result)
        print(json.dumps({k:v for k,v in result.items() if k!='tokens'}), flush=True)


if __name__ == '__main__':
    main()
