"""Checkpoint causal query blocks while retaining every preceding key/value.

Q/K normalization and RoPE follow the installed MLX-LM Qwen3 implementation
(Apple Inc., MIT). This changes attention's evaluation schedule, not its range.
Inference with KV caches and non-causal calls keep the installed implementation.
"""
import mlx.core as mx


def chunk_qwen_attention(model, query_chunk_size, include_eval=False):
    if query_chunk_size < 1 or model.model_type != 'qwen3':
        raise ValueError('Positive query block size and Qwen3 model required')
    cls = type(model.layers[0].self_attn)
    if cls.__module__ != 'mlx_lm.models.qwen3' or cls.__name__ != 'Attention':
        raise TypeError('Unrecognized attention implementation')
    if not getattr(cls, '_jun_query_checkpoint', False):
        original = cls.__call__

        def wrapped(current, x, mask=None, cache=None):
            size = getattr(current, '_jun_query_chunk_size', 0)
            if (not size or (not current.training and not getattr(current, '_jun_query_include_eval', False)) or cache is not None or
                not isinstance(mask, str) or mask != 'causal' or x.shape[1] <= size):
                return original(current, x, mask, cache)
            batch, length, _ = x.shape
            queries, keys, values = current.q_proj(x), current.k_proj(x), current.v_proj(x)
            queries = current.q_norm(queries.reshape(batch, length, current.n_heads, -1)).transpose(0, 2, 1, 3)
            keys = current.k_norm(keys.reshape(batch, length, current.n_kv_heads, -1)).transpose(0, 2, 1, 3)
            values = values.reshape(batch, length, current.n_kv_heads, -1).transpose(0, 2, 1, 3)
            queries, keys = current.rope(queries), current.rope(keys)

            def attend(q, k, v):
                # With K ending at the query block's end, causal alignment
                # K_length-Q_length gives the correct absolute query offset.
                return mx.fast.scaled_dot_product_attention(q, k, v, scale=current.scale, mask='causal')

            parts = []
            for start in range(0, length, size):
                end = min(length, start+size)
                parts.append(mx.checkpoint(attend)(queries[:, :, start:end], keys[:, :, :end], values[:, :, :end]))
            output = mx.concatenate(parts, axis=2).transpose(0, 2, 1, 3).reshape(batch, length, -1)
            return current.o_proj(output)

        cls.__call__ = wrapped
        cls._jun_query_checkpoint = True
    for layer in model.layers:
        layer.self_attn._jun_query_chunk_size = query_chunk_size
        layer.self_attn._jun_query_include_eval = include_eval
