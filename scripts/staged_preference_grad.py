"""Exact DPO chain rule with sequential chosen/rejected gradients and SFT replay.

Replaying unrelated positive conversations can anchor a narrow preference update.
The method is offline preference optimization, not GRPO or an external judge.
"""
import mlx.core as mx
from mlx.utils import tree_map
from staged_character_grad import staged_character_grad


def preference_grad(model, chosen, rejected, offset, ref_margin, beta, nll_weight,
                    replay=None, replay_weight=0., chunk_size=64):
    if beta <= 0 or nll_weight < 0 or replay_weight < 0:
        raise ValueError('Invalid objective weights')
    if replay_weight and replay is None:
        raise ValueError('A positive replay weight requires a replay conversation')
    selected_c = mx.broadcast_to(mx.arange(chosen.shape[1]) >= offset, chosen.shape)
    selected_r = mx.broadcast_to(mx.arange(rejected.shape[1]) >= offset, rejected.shape)
    (chosen_nll, count), cg = staged_character_grad(model, chosen, selected_c,
        chunk_size=chunk_size, denominator=1)
    (rejected_nll, _), rg = staged_character_grad(model, rejected, selected_r,
        chunk_size=chunk_size, denominator=1)
    # Existing objective: softplus(-beta*(cp-rp-ref)) - nll_weight*cp/count.
    z = beta * (chosen_nll - rejected_nll + ref_margin)
    preference_coefficient = beta * mx.sigmoid(z)
    value = mx.logaddexp(mx.array(0.), z) + nll_weight * chosen_nll / count
    mx.eval(value, preference_coefficient)
    gradient = tree_map(lambda c, r: (preference_coefficient+nll_weight/count)*c
                        - preference_coefficient*r, cg, rg)
    mx.eval(gradient)
    del cg, rg
    mx.clear_cache()
    replay_loss = None
    if replay_weight:
        (replay_loss, _), sg = staged_character_grad(model, *replay, chunk_size=chunk_size)
        value = value + replay_weight * replay_loss
        gradient = tree_map(lambda d, s: d + replay_weight*s, gradient, sg)
        mx.eval(value, gradient)
        del sg
        mx.clear_cache()
    return value, gradient
