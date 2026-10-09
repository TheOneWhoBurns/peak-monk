"""Experimental, stateless teacher-forced forward for the official Moshi MLX model.

The upstream Mac forward is designed for one-frame streaming. This implementation
adds causal sequence training and retains its inter-codebook dependency attention.
Inputs are already delayed streams, including a leading initial-token frame.
This module does not claim that random-code smoke tests constitute speech training.
"""
import mlx.core as mx
import mlx.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base, rank=4, scale=1.0):
        super().__init__()
        self.base = base
        self.base.freeze()
        nout, nin = base.weight.shape
        if isinstance(base, nn.QuantizedLinear):
            nin = nin * 32 // base.bits
        # Small adapter updates can vanish when added directly to bf16 A values.
        # Keep the few trainable parameters in fp32 and cast only the residual.
        self.a = (mx.random.normal((nin, rank)) * (1 / nin**0.5)).astype(mx.float32)
        self.b = mx.zeros((rank, nout), dtype=mx.float32)
        self.scale = scale

    def __call__(self, x):
        y=self.base(x)
        return y + (self.scale * ((x @ self.a) @ self.b)).astype(y.dtype)


def attach_lora(model, layers=2, rank=4, scale=1.0):
    model.freeze()
    for layer in model.transformer.layers[-layers:]:
        layer.self_attn.out_proj = LoRALinear(layer.self_attn.out_proj, rank, scale)
        layer.gating.linear_out = LoRALinear(layer.gating.linear_out, rank, scale)
    model.text_linear = LoRALinear(model.text_linear, rank, scale)


def attention(attn, xs, previous=None):
    b, t, dim = xs.shape
    qkv = attn.in_proj(xs).reshape(b, t, 3, attn.cfg.num_heads, attn.cfg.head_dim)
    q, k, v = [qkv[:, :, i].transpose(0, 2, 1, 3) for i in range(3)]
    offset = 0 if previous is None else previous[0].shape[2]
    if attn.rope is not None:
        q, k = attn.rope(q, offset=offset), attn.rope(k, offset=offset)
    if previous is not None:
        k, v = mx.concatenate([previous[0], k], axis=2), mx.concatenate([previous[1], v], axis=2)
    qpos = mx.arange(t)[:, None] + offset
    kpos = mx.arange(k.shape[2])[None, :]
    mask = (kpos <= qpos) & (kpos >= qpos - attn.cfg.context)
    y = mx.fast.scaled_dot_product_attention(q, k, v, scale=attn.scale, mask=mask)
    y = y.transpose(0, 2, 1, 3).reshape(b, t, dim)
    return attn.out_proj(y), (k, v)


def layer_forward(layer, xs, previous=None):
    update, kv = attention(layer.self_attn, layer.norm1(xs), previous)
    xs = xs + layer.layer_scale_1(update)
    assert layer.cross_attention is None, 'This trainer currently supports unconditional Moshi only'
    xs = xs + layer.layer_scale_2(layer.gating(layer.norm2(xs)))
    return xs, kv


def temporal_forward(model, inputs):
    # [batch, 1 + audio_codebooks, frames]
    xs = embed(model.text_emb, inputs[:, 0])
    for i, emb in enumerate(model.audio_embs):
        xs = xs + embed(emb, inputs[:, i + 1])
    for layer in model.transformer.layers:
        xs, _ = layer_forward(layer, xs)
    xs = model.out_norm(xs)
    return xs, model.text_linear(xs)


def embed(module, tokens):
    # Quantizing ScaledEmbedding replaces its zero-index handling upstream.
    return mx.where((tokens < 0)[..., None], 0, module(mx.maximum(tokens, 0)))


def depth_forward(model, hidden, targets):
    b, t, dim = hidden.shape
    hidden = hidden.reshape(b * t, 1, dim)
    previous = [None] * len(model.depformer.slices[0].transformer.layers)
    logits = []
    for i, sub in enumerate(model.depformer.slices):
        # Slice 0 conditions on current text, subsequent slices on prior audio.
        token = targets[:, i].reshape(b * t, 1)
        xs = sub.linear_in(hidden) + embed(sub.emb, token)
        for j, layer in enumerate(sub.transformer.layers):
            xs, previous[j] = layer_forward(layer, xs, previous[j])
        logits.append(sub.linear_out(xs).reshape(b, t, -1))
    return mx.stack(logits, axis=1)


def loss(model, streams, text_padding_weight=0.5, audio_weight=1.0):
    inputs, targets = streams[:, :, :-1], streams[:, :, 1:]
    hidden, text_logits = temporal_forward(model, inputs)
    text_target = targets[:, 0]
    text_valid = (text_target >= 0) & (text_target < model.cfg.text_out_vocab_size)
    # Official defaults: 3 padding, 0 end-of-padding; both receive reduced weight.
    text_weights = mx.where((text_target == 3) | (text_target == 0), text_padding_weight, 1.0) * text_valid
    text_ce = nn.losses.cross_entropy(text_logits.astype(mx.float32), mx.clip(text_target, 0, model.cfg.text_out_vocab_size - 1))
    text_loss = (text_ce * text_weights).sum() / mx.maximum(text_weights.sum(), 1)
    if audio_weight == 0:
        return text_loss
    audio_logits = depth_forward(model, hidden, targets)
    audio_targets = targets[:, 1:1 + model.dep_q]
    valid = (audio_targets >= 0) & (audio_targets < model.cfg.audio_vocab_size - 1)
    safe = mx.clip(audio_targets, 0, model.cfg.audio_vocab_size - 2)
    ce = nn.losses.cross_entropy(audio_logits.astype(mx.float32), safe)
    # Preserve the official semantic-codebook emphasis without inflating total scale.
    weights = mx.array([100.0] + [1.0] * (model.dep_q - 1))[None, :, None] * valid
    audio_loss = (ce * weights).sum() / mx.maximum(weights.sum(), 1)
    return text_loss + audio_weight * audio_loss
