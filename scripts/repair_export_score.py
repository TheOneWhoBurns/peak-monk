"""Score selected target tokens without allocating full-history vocabulary logits."""
import mlx.core as mx
import mlx.nn as nn


def selected_score(model, tokens, mask, input_embeddings=None, layerwise=False, boundary=None, prefill_chunk=0):
    batch = mx.array([tokens])
    indices = [i-1 for i in range(1, len(tokens)) if mask[i]]
    if not indices or len(indices) > 160:
        raise ValueError('Export check expects one short selected reply')
    if prefill_chunk:
        if input_embeddings is not None or layerwise:
            raise ValueError('Cached token prefill requires the complete backbone')
        from mlx_lm.models.cache import make_prompt_cache
        cache = make_prompt_cache(model)
        selected_hidden = []
        for start in range(0, len(tokens)-1, prefill_chunk):
            end = min(start+prefill_chunk, len(tokens)-1)
            hidden = model.model(batch[:, start:end], cache=cache)
            mx.eval(hidden, [c.state for c in cache])
            local = [i-start for i in indices if start <= i < end]
            if local:
                selected_hidden.append(hidden[:, mx.array(local), :])
                mx.eval(selected_hidden)
            del hidden
            if boundary:
                boundary()
        hidden = mx.concatenate(selected_hidden, axis=1)
    elif layerwise:
        from mlx_lm.models.base import create_attention_mask
        hidden = model.model.embed_tokens(batch[:, :-1]) if input_embeddings is None else input_embeddings
        attention_mask = create_attention_mask(hidden, None)
        for layer in model.layers:
            hidden = layer(hidden, attention_mask)
            mx.eval(hidden)
            if boundary:
                boundary()
        hidden = model.model.norm(hidden)
    else:
        hidden = model.model(batch[:, :-1], input_embeddings=input_embeddings)
    mx.eval(hidden)
    selected = mx.array(indices)
    logits = model.model.embed_tokens.as_linear(hidden if prefill_chunk else hidden[:, selected, :])
    targets = batch[:, selected+1]
    losses = nn.losses.cross_entropy(logits.astype(mx.float32), targets)
    mx.eval(losses)
    return dict(target_tokens=len(indices), mean_loss=losses.mean().item(),
                target_log_probs=(-losses[0]).tolist())
