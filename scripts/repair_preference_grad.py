"""Masked DPO and chosen-reply NLL for complete cached student histories.

Use explicit masks through EOS; ChatML's trailing newline is not a target.
Dropout is disabled for both policy and frozen-reference probabilities.
"""
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map
from staged_character_grad import staged_character_grad


def masked_reply_score(model, tokens, mask, features=None, chunk_size=32):
    if model.training or mask.dtype != mx.bool_:
        raise ValueError('Evaluation mode and explicit boolean mask required')
    hidden = model.model(tokens[:, :-1], input_embeddings=features)
    mx.eval(hidden)
    selected = mask[:, 1:]
    count = selected.sum()
    if count.item() <= 0:
        raise ValueError('No reply targets')
    total = mx.array(0., dtype=mx.float32)
    for start in range(0, hidden.shape[1], chunk_size):
        end = min(start + chunk_size, hidden.shape[1])
        local = selected[:, start:end]
        if not mx.any(local).item():
            continue
        logits = model.model.embed_tokens.as_linear(hidden[:, start:end])
        ce = nn.losses.cross_entropy(logits.astype(mx.float32), tokens[:, start+1:end+1])
        total = total + mx.where(local, ce, 0).sum()
        mx.eval(total)
        del logits, ce
    return dict(logp=-total.item(), target_tokens=count.item())


def repair_preference_grad(model, chosen, rejected, reference_margin, beta=.1,
                           nll_weight=.2, chosen_features=None, rejected_features=None,
                           chunk_size=32):
    if model.training or beta <= 0 or nll_weight < 0:
        raise ValueError('Preference gradients require eval-mode modules and valid weights')
    (cnll, count), cg = staged_character_grad(model, *chosen, chunk_size=chunk_size,
        denominator=1, input_embeddings=chosen_features)
    (rnll, rcount), rg = staged_character_grad(model, *rejected, chunk_size=chunk_size,
        denominator=1, input_embeddings=rejected_features)
    # margin = log p(chosen) - log p(rejected) = rejected NLL - chosen NLL.
    z = beta * (cnll - rnll + reference_margin)
    coefficient = beta * mx.sigmoid(z)
    preference_loss = mx.logaddexp(mx.array(0.), z)
    value = preference_loss + nll_weight * cnll / count
    gradient = tree_map(lambda c, r: (coefficient + nll_weight/count)*c - coefficient*r, cg, rg)
    mx.eval(value, gradient, cnll, rnll, count, rcount, preference_loss)
    return value, gradient, dict(preference_loss=preference_loss.item(),
        chosen_mean_nll=(cnll/count).item(), rejected_mean_nll=(rnll/rcount).item(),
        chosen_logp=-cnll.item(), rejected_logp=-rnll.item(),
        reference_margin=reference_margin, relative_margin=rnll.item()-cnll.item()-reference_margin)
