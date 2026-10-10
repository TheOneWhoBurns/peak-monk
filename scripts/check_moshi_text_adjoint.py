"""Verify proposed text-only shared-Moshi gradients before any real training."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map
from moshi_mlx.models import Lm, config_v0_1
from moshi_training import attach_lora, layer_forward
from moshi_text_adjoint import TextMoshiParts, disk_text_grad
from moshi_disk_adjoint import disk_adamw_update


def text_logits(model, tokens):
    h = model.text_emb(tokens)
    for layer in model.transformer.layers:
        h = layer_forward(layer, h)[0]
    return model.text_linear(model.out_norm(h))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    mx.random.seed(211)
    cfg = config_v0_1()
    cfg = replace(cfg,
        transformer=replace(cfg.transformer, d_model=64, num_heads=4, num_layers=2, dim_feedforward=192),
        depformer=replace(cfg.depformer, transformer=replace(cfg.depformer.transformer,
            d_model=32, num_heads=4, num_layers=2, dim_feedforward=96)),
        text_in_vocab_size=65, text_out_vocab_size=64, audio_vocab_size=33)
    model = Lm(cfg)
    model.set_dtype(mx.bfloat16)
    nn.quantize(model, bits=4, group_size=32)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        base = root / 'base.safetensors'
        mx.save_safetensors(str(base), dict(tree_flatten(model.parameters())))
        attach_lora(model, layers=2, rank=2, scale=2.)
        for key, value in tree_flatten(model.trainable_parameters()):
            if key.endswith('.b'):
                model.load_weights([(key, mx.random.normal(value.shape)*.02)], strict=False)
        adapter = root / 'adapter0.safetensors'
        mx.save_safetensors(str(adapter), dict(tree_flatten(model.trainable_parameters())))
        tokens = mx.random.randint(4, 64, (1, 23))
        mask = mx.array([[0]*5 + [1]*6 + [0]*7 + [1]*5])
        # This pinned API is streaming: TransformerLayer supplies no mask for
        # a multi-token call. Feed one token at a time for a causal reference.
        original = mx.concatenate([model.forward_text(tokens[:, i:i+1])[1]
                                   for i in range(tokens.shape[1]-1)], axis=1)
        reference_logits = text_logits(model, tokens[:, :-1])
        logit_error = mx.max(mx.abs(original-reference_logits)).item()
        for cache in model.transformer_cache:
            cache.reset()
        # Isolate causal equivalence from BF16 differences between sequential
        # and batched matmul kernels; restore every original tensor afterwards.
        original_parameters = list(tree_flatten(model.parameters()))
        model.set_dtype(mx.float32)
        original32 = mx.concatenate([model.forward_text(tokens[:, i:i+1])[1]
                                     for i in range(tokens.shape[1]-1)], axis=1)
        batched32 = text_logits(model, tokens[:, :-1])
        fp32_logit_error = mx.max(mx.abs(original32-batched32)).item()
        assert fp32_logit_error < 1e-5, fp32_logit_error
        for cache in model.transformer_cache:
            cache.reset()
        model.load_weights(original_parameters, strict=True)
        del original, reference_logits, original32, batched32, original_parameters

        def objective(current):
            h = current.text_emb(tokens[:, :-1])
            for layer in current.transformer.layers:
                h = layer_forward(layer, h)[0]
            h = current.out_norm(h)
            total = mx.array(0., dtype=mx.float32)
            # Match head batch shapes; BF16 GEMM can differ with batching.
            for start in range(0, h.shape[1], 4):
                end = min(start+4, h.shape[1])
                logits = current.text_linear(h[:, start:end]).astype(mx.float32)
                ce = nn.losses.cross_entropy(logits, tokens[:, start+1:end+1])
                total = total + (ce * mask[:, start+1:end+1]).sum()/mask[:, 1:].sum()
            return total

        reports = []
        optimizer = optim.AdamW(learning_rate=1e-4, weight_decay=.01)
        for step in range(1, 3):
            reference, gradient = nn.value_and_grad(model, objective)(model)
            mx.eval(reference, gradient)
            parts = TextMoshiParts(base, cfg, adapter, rank=2, scale=2.)
            actual = disk_text_grad(parts, tokens, mask, root/f'gradient{step}', head_tokens=4)
            flat = {}
            for record in actual['gradients']:
                prefix = 'text_linear.' if record['part'] == 'heads' else f"transformer.layers.{record['part']}."
                flat.update({prefix+k:v for k,v in mx.load(record['path']).items()})
            expected = dict(tree_flatten(gradient))
            assert set(flat) == set(expected)
            grad_error = max(mx.max(mx.abs(flat[k]-v)).item() for k,v in expected.items())
            loss_error = abs(reference.item()-actual['loss'])
            assert loss_error < 1e-6 and grad_error < 1e-6, (loss_error, grad_error)
            norm = mx.sqrt(sum(mx.sum(v.astype(mx.float32)**2) for v in expected.values()))
            scale = mx.minimum(1., .3/(norm+1e-8))
            optimizer.update(model, tree_map(lambda g:g*scale, gradient))
            mx.eval(model.parameters(), optimizer.state)
            next_adapter = root / f'adapter{step}.safetensors'
            disk_adamw_update(parts, actual, root/'optimizer', next_adapter, clip_norm=.3, expected_step=step)
            saved = mx.load(str(next_adapter))
            params = dict(tree_flatten(model.trainable_parameters()))
            error = max(mx.max(mx.abs(v-saved[k])).item() for k,v in params.items())
            print(json.dumps(dict(step=step,loss_error=loss_error,gradient_error=grad_error,adapter_error=error)),flush=True)
            assert error < 1e-6, error
            reports.append(dict(step=step, loss_error=loss_error, gradient_error=grad_error,
                                adapter_error=error, supervised_tokens=actual['supervised_tokens']))
            adapter = next_adapter
    result = dict(status='pass', dtype='bfloat16', quantization='4bit/group32',
        official_forward_text_bf16_logit_difference=logit_error,
        official_forward_text_fp32_logit_difference=fp32_logit_error, cases=reports,
        scope='Tiny numerical proof only: causal agreement with original streaming forward_text in FP32; BF16 sequential/batch difference recorded. Matched-head-batch full-model versus streamed masked loss, full temporal gradients, clipped AdamW and complete adapter exports. No pretrained-model feasibility or speech transfer evidence.')
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
