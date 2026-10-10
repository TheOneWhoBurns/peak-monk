"""Evaluate vocabulary-head gradients sequentially, then replay the transformer.

The first transformer pass supplies detached hidden states. Each small head
chunk is evaluated and released before the next. A second transformer pass
backpropagates the accumulated hidden-state cotangent through full attention.
Tied embedding gradients from both uses are added. Dropout RNG is replayed and
restored, so this changes the execution schedule rather than the objective.
"""
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map
from mlx_lm.tuner.lora import LoRAEmbedding
from character_delta import CharacterDelta


def restore_rng(state):
    # MLX0.32 exposes a read-only state sentinel. seed(uint64) reconstructs
    # its exact two-uint32 key; never shadow mx.random.state with a Python list.
    if len(state) != 1 or state[0].shape != (2,) or state[0].dtype != mx.uint32:
        raise ValueError('Unrecognized MLX RNG state representation')
    high, low = state[0].tolist()
    mx.random.seed((int(high) << 32) | int(low))


def staged_character_grad(model, batch, mask, chunk_size=64, denominator=None):
    if model.model_type != 'qwen3' or not model.args.tie_word_embeddings:
        raise ValueError('Only tied Qwen3 is supported')
    head = model.model.embed_tokens
    delta = isinstance(head, CharacterDelta) and head.is_embedding
    supported = isinstance(head, LoRAEmbedding) or (delta
        and isinstance(head.base, LoRAEmbedding) and isinstance(head.base.dropout, nn.Identity))
    if not supported or not isinstance(head.dropout, nn.Dropout):
        raise TypeError('Expected standard tied LoRA embedding or fresh delta around a frozen tied adapter')
    if chunk_size < 1 or mask.dtype != mx.bool_:
        raise ValueError('Positive chunk size and boolean target mask required')
    rng_before = list(mx.random.state)
    mx.eval(rng_before)
    hidden = mx.stop_gradient(model.model(batch[:, :-1]))
    mx.eval(hidden)
    keep = head.dropout._p_1 if head.dropout.training else 1.0
    # Match Dropout's draw shape and arithmetic order exactly, including BF16.
    dropout_mask = (mx.random.bernoulli(keep, hidden.shape) if keep != 1.0
                    else mx.ones(hidden.shape, dtype=mx.bool_))
    mx.eval(dropout_mask)
    rng_after = list(mx.random.state)
    mx.eval(rng_after)
    target, supervised = batch[:, 1:], mask[:, 1:]
    count = supervised.sum()
    divisor = mx.maximum(count, 1) if denominator is None else denominator
    # Reviewed on-policy prefixes are conditioning, not positive targets. Avoid
    # allocating a vocabulary projection for blocks with no selected tokens.
    # Still draw dropout over the original full shape to preserve RNG parity.
    selected_columns = mx.any(supervised, axis=0).tolist()

    def head_loss(params, h, dmask, y, selected):
        head.update(params)
        dropped = (dmask * h) * (1 / keep)
        residual = (dropped @ head.lora_b.T) @ head.lora_a.T
        embedding = head.base.embedding if delta else head.embedding
        if delta:
            old = (h @ head.base.lora_b.T) @ head.base.lora_a.T
            residual = old + residual
        logits = embedding.as_linear(h) + (head.scale * residual).astype(h.dtype)
        ce = nn.losses.cross_entropy(logits.astype(mx.float32), y)
        return mx.where(selected, ce, 0).sum() / divisor

    head_vg = mx.value_and_grad(head_loss, argnums=(0, 1))
    head_gradient = tree_map(mx.zeros_like, head.trainable_parameters())
    hidden_gradients = []
    total = mx.array(0., dtype=mx.float32)
    for start in range(0, hidden.shape[1], chunk_size):
        end = min(start + chunk_size, hidden.shape[1])
        if not any(selected_columns[start:end]):
            hidden_gradients.append(mx.zeros_like(hidden[:, start:end]))
            continue
        value, (grad, cotangent) = head_vg(
            head.trainable_parameters(), hidden[:, start:end],
            dropout_mask[:, start:end], target[:, start:end], supervised[:, start:end])
        total = total + value
        head_gradient = tree_map(lambda a, b: a + b, head_gradient, grad)
        mx.eval(total, head_gradient, cotangent)
        hidden_gradients.append(mx.stop_gradient(cotangent))
        del value, grad, cotangent
        mx.clear_cache()
    hidden_cotangent = mx.concatenate(hidden_gradients, axis=1)
    mx.eval(hidden_cotangent)
    del hidden, dropout_mask, hidden_gradients
    mx.clear_cache()

    def replay_loss(current):
        return (current.model(batch[:, :-1]) * hidden_cotangent).sum()

    try:
        restore_rng(rng_before)
        _, gradient = nn.value_and_grad(model, replay_loss)(model)
        if head_gradient:
            gradient['model']['embed_tokens'] = tree_map(
                lambda a, b: a + b, gradient['model']['embed_tokens'], head_gradient)
        mx.eval(total, count, gradient)
    finally:
        restore_rng(rng_after)
    return (total, count), gradient
