"""Checkpoint the large vocabulary projection in chunks for tied Qwen3 LoRA.

Full-context transformer attention is retained. This is not truncated BPTT.
The head's dropout is sampled once over the original hidden-state shape, outside
checkpointed chunks, preserving the ordinary head's RNG order and mask.
"""
from functools import partial
import mlx.core as mx
import mlx.nn as nn
from mlx.nn.utils import checkpoint
from mlx_lm.tuner.lora import LoRAEmbedding


def projection_loss(head, hidden, dropped, target, mask):
    target, mask = mx.stop_gradient(target), mx.stop_gradient(mask)
    logits = head.embedding.as_linear(hidden)
    residual = (dropped @ head.lora_b.T) @ head.lora_a.T
    logits = logits + (head.scale * residual).astype(hidden.dtype)
    ce = nn.losses.cross_entropy(logits.astype(mx.float32), target)
    return mx.where(mask, ce, 0).sum()


def chunked_character_loss(model, batch, mask, denominator=None, chunk_size=64):
    if model.model_type != 'qwen3' or not model.args.tie_word_embeddings:
        raise ValueError('This implementation is limited to tied Qwen3 models')
    head = model.model.embed_tokens
    if not isinstance(head, LoRAEmbedding):
        raise TypeError('Expected the recorded tied LoRA embedding head')
    if mask.dtype != mx.bool_ or chunk_size < 1:
        raise ValueError('Character target mask and positive chunk size required')
    hidden = model.model(batch[:, :-1])
    dropped = head.dropout(hidden)
    targets, supervised = batch[:, 1:], mask[:, 1:]
    project = checkpoint(head, partial(projection_loss, head))
    total = mx.array(0., dtype=mx.float32)
    for start in range(0, hidden.shape[1], chunk_size):
        end = min(start + chunk_size, hidden.shape[1])
        total = total + project(hidden[:, start:end], dropped[:, start:end],
                                targets[:, start:end], supervised[:, start:end])
    count = supervised.sum()
    return total / (mx.maximum(count, 1) if denominator is None else denominator), count
