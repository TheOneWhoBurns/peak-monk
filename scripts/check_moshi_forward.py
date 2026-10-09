"""Check training causality, streaming parity and nonzero adapter gradients.

Uses a tiny randomly initialized Moshi architecture. No trained-model or audio
quality claims follow from this implementation test.
"""
import json
from dataclasses import replace
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
from moshi_mlx.models.lm import Lm, config_v0_1
from moshi_training import temporal_forward, depth_forward, attach_lora, loss
from mlx.utils import tree_flatten
from moshi_split import Prefix, Suffix, cached_loss, delay_streams

mx.random.seed(7)
cfg = config_v0_1()
cfg = replace(cfg,
    transformer=replace(cfg.transformer, d_model=32, num_heads=4, num_layers=2, dim_feedforward=128),
    depformer=replace(cfg.depformer, transformer=replace(cfg.depformer.transformer,
        d_model=16, num_heads=2, num_layers=2, dim_feedforward=64)),
    text_in_vocab_size=33, text_out_vocab_size=32, audio_vocab_size=17)
model = Lm(cfg)
tokens = mx.random.randint(0, 16, (1, 17, 9))
hidden, logits = temporal_forward(model, tokens[:, :, :-1])
xs = model.text_emb(tokens[:, 0, :-1])
for i, emb in enumerate(model.audio_embs):
    xs = xs + emb(tokens[:, i + 1, :-1])
reference = []
cache = model.transformer.make_cache()
for t in range(xs.shape[1]):
    out = model.out_norm(model.transformer(xs[:, t:t+1], cache=cache))
    reference.append(model.text_linear(out))
streaming_error = mx.max(mx.abs(logits - mx.concatenate(reference, axis=1))).item()
changed = mx.concatenate([tokens[:, :, :4], (tokens[:, :, 4:] + 1) % 16], axis=2)
_, changed_logits = temporal_forward(model, changed[:, :, :-1])
future_error = mx.max(mx.abs(logits[:, :4] - changed_logits[:, :4])).item()

targets = tokens[:, :, 1:]
depth_logits = depth_forward(model, hidden, targets)
cache = model.depformer.slices[0].transformer.make_cache()
reference = []
for i, sub in enumerate(model.depformer.slices):
    x = sub.linear_in(hidden.reshape(8, 1, 32)) + sub.emb(targets[:, i].reshape(8, 1))
    reference.append(sub.linear_out(sub.transformer(x, cache=cache)).reshape(1, 8, -1))
depth_error = mx.max(mx.abs(depth_logits - mx.stack(reference, axis=1))).item()
attach_lora(model, layers=1, rank=2)
value, grad = nn.value_and_grad(model, loss)(model, tokens)
flat = tree_flatten(grad)
norm = sum(mx.sum(g.astype(mx.float32)**2).item() for _, g in flat)**0.5
assert streaming_error < 1e-4, streaming_error
assert future_error == 0, future_error
assert depth_error < 1e-4, depth_error
assert norm > 0 and mx.isfinite(value).item(), (norm, value)
assert all(k.endswith(('.a', '.b')) for k, _ in flat), [k for k, _ in flat]
prefix = Prefix(model, 1)
cached_hidden = mx.stop_gradient(prefix(tokens[:, :, :-1]))
mx.eval(cached_hidden)
suffix = Suffix(model, 1)
cached_value, cached_grad = nn.value_and_grad(suffix, cached_loss)(suffix, cached_hidden, targets)
split_loss_error = abs(value.item() - cached_value.item())
original_grad = dict(flat)
split_gradient_error = 0.0
for key, g in tree_flatten(cached_grad):
    full_key = key.replace('transformer.layers.0.', 'transformer.layers.1.')
    split_gradient_error = max(split_gradient_error, mx.max(mx.abs(g-original_grad[full_key])).item())
assert split_loss_error < 1e-5, split_loss_error
assert split_gradient_error < 1e-5, split_gradient_error
delayed = delay_streams(tokens, cfg)
assert delayed.shape == (1,17,10)
assert mx.array_equal(delayed[:,0,1:], tokens[:,0]).item()
assert mx.array_equal(delayed[:,2,2:], tokens[:,2,:-1]).item()
assert delayed[0,2,0].item() == cfg.audio_padding_token
assert delayed[0,2,1].item() == cfg.audio_padding_token
result = dict(scope='Tiny random model implementation check, not speech training',
    streaming_max_error=streaming_error, future_leakage_max_error=future_error,
    depth_max_error=depth_error, loss=value.item(), gradient_norm=norm,
    adapter_gradient_tensors=len(flat), split_loss_error=split_loss_error,
    split_gradient_max_error=split_gradient_error, delay_alignment_passed=True, passed=True)
Path('results/moshi-forward-check.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
