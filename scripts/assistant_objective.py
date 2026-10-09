"""Assistant-only likelihood through EOS, excluding post-EOS formatting/padding."""
import mlx.core as mx
import mlx.nn as nn


class ThroughEOS:
    def __init__(self, dataset, eos_id):
        self.dataset, self.eos_id = dataset, eos_id

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        return self.dataset[i]

    def process(self, item):
        tokens, offset = self.dataset.process(item)
        end = next((i + 1 for i in range(offset, len(tokens)) if tokens[i] == self.eos_id), len(tokens))
        if offset >= end:
            raise ValueError('No assistant target tokens')
        return tokens[:end], offset


def assistant_loss(model, batch, lengths):
    logits = model(batch[:, :-1])
    targets = batch[:, 1:]
    # Position j predicts original token j+1; length is exclusive, not inclusive.
    positions = mx.arange(1, batch.shape[1])
    mask = (positions >= lengths[:, :1]) & (positions < lengths[:, 1:])
    losses = nn.losses.cross_entropy(logits.astype(mx.float32), targets)
    count = mask.sum()
    return mx.sum(mx.where(mask, losses, 0)) / mx.maximum(count, 1), count
