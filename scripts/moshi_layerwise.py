"""Explicit full-episode adjoints for a sequentially loaded Moshi backbone.

The in-memory driver below is a numerical reference for a future disk-streamed
runner. It does not itself demonstrate pretrained-model memory feasibility.
"""
import mlx.core as mx
from mlx.utils import tree_flatten
from moshi_training import embed, layer_forward
from moshi_staged import staged_head_grad


def temporal_layer_vjp(layer, hidden, cotangent):
    def scalar(params, inputs):
        layer.update(params)
        return (layer_forward(layer, inputs)[0] * cotangent).sum()
    _, (gradient, previous_cotangent) = mx.value_and_grad(scalar, argnums=(0, 1))(
        layer.trainable_parameters(), hidden)
    mx.eval(gradient, previous_cotangent)
    return gradient, mx.stop_gradient(previous_cotangent)


def norm_vjp(norm, hidden, cotangent):
    _, previous = mx.value_and_grad(lambda inputs: (norm(inputs) * cotangent).sum())(hidden)
    mx.eval(previous)
    return mx.stop_gradient(previous)


def layerwise_episode_grad(model, streams, head_frames=16, padding_weight=.05):
    for key, _ in tree_flatten(model.trainable_parameters()):
        if not key.startswith(('transformer.layers.', 'text_linear.')):
            raise ValueError('Only temporal and text-head adapters are supported')
    inputs, targets = streams[:, :, :-1], streams[:, :, 1:]
    hidden = embed(model.text_emb, inputs[:, 0])
    for i, emb in enumerate(model.audio_embs):
        hidden = hidden + embed(emb, inputs[:, i+1])
    hidden = mx.stop_gradient(hidden)
    mx.eval(hidden)
    boundaries = [hidden]
    for layer in model.transformer.layers:
        hidden = mx.stop_gradient(layer_forward(layer, hidden)[0])
        mx.eval(hidden)
        boundaries.append(hidden)
    normalized = mx.stop_gradient(model.out_norm(hidden))
    mx.eval(normalized)
    value, head_gradient, cotangent = staged_head_grad(model, normalized, targets, head_frames, padding_weight)
    cotangent = norm_vjp(model.out_norm, hidden, cotangent)
    gradients = [{} for _ in model.transformer.layers]
    for index in reversed(range(len(model.transformer.layers))):
        gradients[index], cotangent = temporal_layer_vjp(model.transformer.layers[index], boundaries[index], cotangent)
    return value, {'transformer': {'layers': gradients}, 'text_linear': head_gradient}
