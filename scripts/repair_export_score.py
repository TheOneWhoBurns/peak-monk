"""Score selected target tokens without allocating full-history vocabulary logits."""
import mlx.core as mx
import mlx.nn as nn


def selected_score(model, tokens, mask, input_embeddings=None):
    batch = mx.array([tokens])
    indices = [i-1 for i in range(1, len(tokens)) if mask[i]]
    if not indices or len(indices) > 160:
        raise ValueError('Export check expects one short selected reply')
    hidden = model.model(batch[:, :-1], input_embeddings=input_embeddings)
    mx.eval(hidden)
    selected = mx.array(indices)
    logits = model.model.embed_tokens.as_linear(hidden[:, selected, :])
    targets = batch[:, selected+1]
    losses = nn.losses.cross_entropy(logits.astype(mx.float32), targets)
    mx.eval(losses)
    return dict(target_tokens=len(indices), mean_loss=losses.mean().item(),
                target_log_probs=(-losses[0]).tolist())
