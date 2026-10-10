"""Verify full-episode versus staged joint gradients on a tiny quantized Moshi."""
from dataclasses import replace
import argparse
import json
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from moshi_mlx.models import Lm, config_v0_1
from moshi_training import attach_lora, loss
from moshi_staged import staged_episode_grad
from moshi_layerwise import layerwise_episode_grad

p = argparse.ArgumentParser()
p.add_argument('--device', choices=['cpu', 'gpu'], default='cpu')
p.add_argument('--dtype', choices=['float32', 'bfloat16'], default='float32')
p.add_argument('--output', type=Path, default=Path('results/moshi-staged-gradient-check.json'))
a = p.parse_args()
if a.device == 'cpu' and a.dtype == 'bfloat16':
    raise ValueError('The uncheckpointed BF16 CPU reference segfaults in this pinned MLX0.26.5 runtime; use GPU for that separate check')
mx.set_default_device(mx.cpu if a.device == 'cpu' else mx.gpu)
reports = []
for dtype, tolerance in [(mx.float32, 1e-5)] if a.dtype == 'float32' else [(mx.bfloat16, .02)]:
    print('Checking', dtype, flush=True)
    mx.random.seed(57)
    cfg = config_v0_1()
    cfg = replace(cfg,
        transformer=replace(cfg.transformer, d_model=64, num_heads=4, num_layers=2, dim_feedforward=192),
        depformer=replace(cfg.depformer, transformer=replace(cfg.depformer.transformer,
            d_model=32, num_heads=4, num_layers=2, dim_feedforward=96)),
        text_in_vocab_size=33, text_out_vocab_size=32, audio_vocab_size=33)
    model = Lm(cfg)
    model.set_dtype(dtype)
    nn.quantize(model, bits=4, group_size=32)
    attach_lora(model, layers=2, rank=2, scale=2.)
    for key, value in tree_flatten(model.trainable_parameters()):
        if key.endswith('.b'):
            parts = key.split('.')
            current = model
            for part in parts[:-1]:
                current = current[int(part)] if isinstance(current, list) else getattr(current, part)
            setattr(current, parts[-1], mx.random.normal(value.shape) * .02)
    model.train()
    tokens = mx.random.randint(0, 31, (1, 17, 14))
    # Include padding and invalid delayed-stream boundaries in both losses.
    tokens[:, 0, 1:3] = mx.array([0, 3])
    tokens[:, 1:4, -1] = -1
    print('Full reference', flush=True)
    reference, reference_grad = nn.value_and_grad(model,
        lambda m: loss(m, tokens, text_padding_weight=.05, checkpoint_layers=False))(model)
    mx.eval(reference, reference_grad)
    reference_grad = dict(tree_flatten(reference_grad))
    reference_norm = sum(float(mx.sum(g.astype(mx.float32)**2).item()) for g in reference_grad.values())**.5
    assert reference_norm > 0
    for method, fn, frames in [(name, fn, frames) for name, fn in
            [('staged', staged_episode_grad), ('layerwise', layerwise_episode_grad)]
            for frames in [4, 8, 16]]:
        print(method, 'head frames', frames, flush=True)
        actual, grad = fn(model, tokens, head_frames=frames)
        mx.eval(actual, grad)
        assert set(dict(tree_flatten(grad))) == set(reference_grad)
        errors = [g.astype(mx.float32)-reference_grad[k].astype(mx.float32) for k, g in tree_flatten(grad)]
        max_error = max(float(mx.max(mx.abs(e)).item()) for e in errors)
        relative = sum(float(mx.sum(e**2).item()) for e in errors)**.5 / reference_norm
        error = abs(actual.item()-reference.item())
        assert error <= tolerance and max_error <= tolerance and relative <= tolerance, (dtype, frames, error, max_error, relative)
        reports.append(dict(method=method, dtype=str(dtype), head_frames=frames, loss_error=error,
            maximum_gradient_error=max_error, relative_gradient_l2_error=relative, tolerance=tolerance))
report = dict(status='pass', device=a.device, dtype=a.dtype, cases=reports,
    scope='Tiny random quantized Moshi, nonzero adapters, all trainable gradients against full uncheckpointed joint loss. Head chunks include ragged masked tails. Not pretrained-model memory or quality evidence.')
a.output.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
