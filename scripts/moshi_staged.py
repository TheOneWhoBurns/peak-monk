"""Full-episode gradients with sequential text/audio heads and temporal replay.

Unlike windowed truncated BPTT, the temporal replay attends and differentiates
through the complete episode. Only the independent per-frame depth-head batch
is chunked. Pretrained weights remain frozen; the existing temporal and text-head
adapters are the only supported trainable parameters. Moshi has no dropout here.
"""
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map
from moshi_training import LoRALinear, temporal_forward, depth_forward
from moshi_windowed import normalizers


def staged_episode_grad(model, streams, head_frames=16, padding_weight=.05):
    if head_frames < 1 or not 0 < padding_weight <= 1:
        raise ValueError('Positive head-frame count and valid padding weight required')
    if not isinstance(model.text_linear, LoRALinear):
        raise TypeError('Expected the audited Moshi text-head adapter')
    for key, _ in tree_flatten(model.trainable_parameters()):
        if not key.startswith(('transformer.layers.', 'text_linear.')):
            raise ValueError('Only temporal and text-head adapters are supported')
    inputs, targets = streams[:, :, :-1], streams[:, :, 1:]
    denominator = normalizers(model, targets, padding_weight)
    hidden = mx.stop_gradient(temporal_forward(model, inputs, checkpoint_layers=True)[0])
    mx.eval(hidden)
    head_gradient = tree_map(mx.zeros_like, model.text_linear.trainable_parameters())
    total = mx.array(0., dtype=mx.float32)
    cotangents = []

    def head_loss(params, h, y):
        model.text_linear.update(params)
        text = y[:, 0]
        valid = (text >= 0) & (text < model.cfg.text_out_vocab_size)
        weight = mx.where((text == 0) | (text == 3), padding_weight, 1.) * valid
        ce = nn.losses.cross_entropy(model.text_linear(h).astype(mx.float32),
                                     mx.clip(text, 0, model.cfg.text_out_vocab_size - 1))
        text_loss = (ce * weight).sum() / denominator[0]
        logits = depth_forward(model, h, y, checkpoint_layers=True)
        audio = y[:, 1:1 + model.dep_q]
        valid = (audio >= 0) & (audio < model.cfg.audio_vocab_size - 1)
        weight = mx.array([100.] + [1.] * (model.dep_q - 1))[None, :, None] * valid
        ce = nn.losses.cross_entropy(logits.astype(mx.float32),
                                     mx.clip(audio, 0, model.cfg.audio_vocab_size - 2))
        return text_loss + (ce * weight).sum() / denominator[1]

    value_grad = mx.value_and_grad(head_loss, argnums=(0, 1))
    for start in range(0, hidden.shape[1], head_frames):
        end = min(start + head_frames, hidden.shape[1])
        h, y = hidden[:, start:end], targets[:, :, start:end]
        length = end - start
        if length < head_frames:
            # Each frame is a separate depth-transformer batch item; padding
            # cannot change the real frames and contributes no target weight.
            h = mx.concatenate([h, mx.zeros((h.shape[0], head_frames-length, h.shape[2]), dtype=h.dtype)], axis=1)
            y = mx.concatenate([y, mx.full((*y.shape[:2], head_frames-length), -1, dtype=y.dtype)], axis=-1)
        value, (part, cotangent) = value_grad(model.text_linear.trainable_parameters(), h, y)
        total = total + value
        head_gradient = tree_map(lambda a, b: a + b, head_gradient, part)
        cotangent = mx.stop_gradient(cotangent[:, :length])
        mx.eval(total, head_gradient, cotangent)
        cotangents.append(cotangent)
        del value, part, cotangent, h, y
        mx.clear_cache()
    cotangent = mx.concatenate(cotangents, axis=1)
    mx.eval(cotangent)
    del hidden, cotangents
    mx.clear_cache()

    def replay(current):
        hidden = temporal_forward(current, inputs, checkpoint_layers=True)[0]
        return (hidden * cotangent).sum()

    _, gradient = nn.value_and_grad(model, replay)(model)
    gradient['text_linear'] = head_gradient
    mx.eval(total, gradient)
    return total, gradient
