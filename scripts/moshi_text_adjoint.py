"""Text-only shared-backbone gradients; no audio embedding or depth objective.

Experimental transfer path, not a native speech quality claim. Inputs are Moshi
token IDs with an explicit target mask, never Qwen IDs or assumed chat tokens.
"""
import gc
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map
from moshi_mlx.models.lm import ScaledEmbedding
from moshi_training import LoRALinear, layer_forward
from moshi_weight_stream import MoshiParts
from moshi_layerwise import temporal_layer_vjp, norm_vjp


class TextInputs(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.text_emb = ScaledEmbedding(cfg.text_in_vocab_size, cfg.transformer.d_model,
                                       demux_second_stream=cfg.demux_second_stream)

    def __call__(self, tokens):
        return self.text_emb(tokens)


class TextHeads(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        dim = cfg.transformer.d_model
        self.out_norm = nn.RMSNorm(dim, 1e-8) if cfg.transformer.norm == 'rms_norm' else nn.LayerNorm(dim, 1e-5)
        self.text_linear = nn.Linear(dim, cfg.text_out_vocab_size, bias=False)


class TextMoshiParts(MoshiParts):
    def load(self, part):
        if part not in ('inputs', 'heads'):
            return super().load(part)
        module = TextInputs(self.cfg) if part == 'inputs' else TextHeads(self.cfg)
        prefixes = ['text_emb.'] if part == 'inputs' else ['out_norm.', 'text_linear.']
        module.set_dtype(mx.bfloat16)
        nn.quantize(module, bits=4, group_size=32)
        weights = self.weights.load(prefixes)
        module.load_weights(list(weights.items()), strict=True)
        del weights
        module.freeze()
        if part == 'heads' and self.adapter:
            module.text_linear = LoRALinear(module.text_linear, self.rank, self.scale)
            weights = self.adapter.load(['text_linear.'])
            if set(weights) != set(dict(tree_flatten(module.trainable_parameters()))):
                raise ValueError('Text adapter mapping mismatch')
            module.load_weights(list(weights.items()), strict=False)
        mx.eval(module.parameters())
        return module


def disk_text_grad(parts, tokens, target_mask, scratch, head_tokens=32, boundary=lambda **kw: None):
    if tokens.ndim != 2 or tokens.shape != target_mask.shape or tokens.shape[1] < 2:
        raise ValueError('Expected matching batch/token arrays')
    if head_tokens < 1 or not mx.all((tokens >= 0) & (tokens < parts.cfg.text_out_vocab_size)).item():
        raise ValueError('Invalid text token or head chunk size')
    if not mx.all((target_mask == 0) | (target_mask == 1)).item():
        raise ValueError('Expected binary target mask')
    denominator = target_mask[:, 1:].sum().item()
    if denominator <= 0:
        raise ValueError('No supervised text tokens')
    scratch = Path(scratch)
    scratch.mkdir(exist_ok=False)

    def release(phase, **kw):
        gc.collect()
        mx.clear_cache()
        boundary(phase=phase, **kw)

    def save_hidden(index, hidden):
        mx.eval(hidden)
        mx.save_safetensors(str(scratch / f'hidden-{index:02d}.safetensors'), {'hidden': hidden})

    inputs = parts.load('inputs')
    hidden = mx.stop_gradient(inputs(tokens[:, :-1]))
    save_hidden(0, hidden)
    del inputs
    release('text_embeddings_complete')
    for index in range(parts.cfg.transformer.num_layers):
        layer = parts.load(index)
        hidden = mx.stop_gradient(layer_forward(layer, hidden)[0])
        save_hidden(index + 1, hidden)
        del layer
        release('text_temporal_forward_complete', layer=index)
    heads = parts.load('heads')
    normalized = mx.stop_gradient(heads.out_norm(hidden))
    mx.eval(normalized)
    gradient = tree_map(mx.zeros_like, heads.text_linear.trainable_parameters())
    total, cotangents = mx.array(0., dtype=mx.float32), []

    def objective(params, h, labels, mask):
        heads.text_linear.update(params)
        ce = nn.losses.cross_entropy(heads.text_linear(h).astype(mx.float32), labels)
        return (ce * mask).sum() / denominator

    value_grad = mx.value_and_grad(objective, argnums=(0, 1))
    for start in range(0, normalized.shape[1], head_tokens):
        end = min(start + head_tokens, normalized.shape[1])
        value, (part, cotangent) = value_grad(heads.text_linear.trainable_parameters(),
            normalized[:, start:end], tokens[:, start+1:end+1], target_mask[:, start+1:end+1])
        total = total + value
        gradient = tree_map(lambda a, b: a + b, gradient, part)
        cotangents.append(mx.stop_gradient(cotangent))
        mx.eval(total, gradient, cotangents[-1])
        del value, part, cotangent
        release('text_head_gradient_chunk', token=end)
    cotangent = norm_vjp(heads.out_norm, hidden, mx.concatenate(cotangents, axis=1))
    value = total.item()
    gradients, squared_norm = [], 0.

    def save_gradient(part, grad):
        nonlocal squared_norm
        flat = dict(tree_flatten(grad))
        mx.eval(flat)
        if not all(mx.all(mx.isfinite(v)).item() for v in flat.values()):
            raise ValueError('Nonfinite text gradient')
        squared_norm += sum(mx.sum(v.astype(mx.float32)**2).item() for v in flat.values())
        path = scratch / f'gradient-{part}.safetensors'
        mx.save_safetensors(str(path), flat)
        gradients.append(dict(part=part, path=str(path)))

    save_gradient('heads', gradient)
    del heads, normalized, hidden, gradient, cotangents, total
    release('text_heads_gradient_complete', loss=value)
    for index in reversed(range(parts.cfg.transformer.num_layers)):
        hidden = mx.load(str(scratch / f'hidden-{index:02d}.safetensors'))['hidden']
        layer = parts.load(index)
        gradient, cotangent = temporal_layer_vjp(layer, hidden, cotangent)
        save_gradient(index, gradient)
        del layer, hidden, gradient
        release('text_temporal_backward_complete', layer=index)
    return dict(loss=value, gradient_norm=squared_norm**.5, gradients=gradients,
                supervised_tokens=denominator, objective='Masked text-only next-token CE')
