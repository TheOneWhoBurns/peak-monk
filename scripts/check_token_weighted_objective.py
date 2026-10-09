"""Compare accumulated gradients with one unequal-length combined batch."""
import json
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from token_weighted_objective import token_weighted_loss
from conversation_objective import conversation_loss

mx.random.seed(41)
model = nn.Sequential(nn.Embedding(11, 7), nn.Linear(7, 11))
# Four conversations with strongly unequal target counts, including EOS targets.
batches = [mx.array([[1, 2, 3]]), mx.array([[2, 4, 5, 7, 8, 9]]),
           mx.array([[3, 6, 4, 2]]), mx.array([[2, 1, 3, 4, 5]])]
masks = [mx.array([[False, False, True]]), mx.array([[False, True, True, True, True, True]]),
         mx.array([[False, False, True, True]]), mx.array([[False, True, False, True, True]])]
total = sum(int(m[:, 1:].sum().item()) for m in masks)
padded = mx.concatenate([mx.pad(b, ((0, 0), (0, 6-b.shape[1]))) for b in batches])
combined_mask = mx.concatenate([mx.pad(m, ((0, 0), (0, 6-m.shape[1]))) for m in masks])
(reference, _), expected = nn.value_and_grad(model, conversation_loss)(model, padded, combined_mask)
actual, losses = {}, []
for batch, mask in zip(batches, masks):
    (loss, _), gradient = nn.value_and_grad(model, token_weighted_loss)(model, batch, mask, mx.array(total/4))
    losses.append(loss)
    for key, value in tree_flatten(gradient):
        actual[key] = actual.get(key, 0) + value / 4
errors = [float(mx.max(mx.abs(actual[k]-v)).item()) for k, v in tree_flatten(expected)]
loss_error = float(mx.abs(sum(losses)/4-reference).item())
assert max(errors) < 1e-6 and loss_error < 1e-6
# Conventional equal-microbatch weighting must differ for this unequal example.
equal = {}
for batch, mask in zip(batches, masks):
    _, gradient = nn.value_and_grad(model, conversation_loss)(model, batch, mask)
    for key, value in tree_flatten(gradient): equal[key] = equal.get(key, 0) + value/4
difference = max(float(mx.max(mx.abs(equal[k]-v)).item()) for k,v in tree_flatten(expected))
assert difference > 1e-3
result = dict(status='pass', target_tokens=total, maximum_gradient_error=max(errors),
              loss_error=loss_error, equal_microbatch_gradient_difference=difference,
              scope='Tiny deterministic model; no evidence of character quality')
Path('results/token-weighted-objective-check.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
