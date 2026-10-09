"""Normalize each accumulated update by its total number of character tokens.

The upstream trainer averages microbatch gradients. Multiplying each microbatch
mean by G * its_token_count / group_token_count makes that average equal to the
mean token loss of a combined batch, without holding its activations in memory.
Validation retains the frozen assistant-through-EOS metric.
"""
from functools import partial
from itertools import islice
import mlx.core as mx
import mlx.nn as nn
from conversation_objective import ConversationDataset, conversation_batches, conversation_loss


def weighted_batches(dataset, batch_size, max_seq_length, loop=False, comm_group=None, accumulation=4):
    stream = conversation_batches(dataset, batch_size, max_seq_length, loop, comm_group)
    if not isinstance(dataset._data, ConversationDataset):
        yield from stream
        return
    if accumulation < 1:
        raise ValueError('Positive accumulation is required')
    while group := list(islice(stream, accumulation)):
        if len(group) != accumulation:
            raise ValueError('Token-weighted training requires complete accumulation groups')
        total = sum(int(mask[:, 1:].sum().item()) for _, mask in group)
        if total <= 0:
            raise ValueError('No target tokens in accumulation group')
        denominator = mx.array(total / accumulation, dtype=mx.float32)
        for batch, mask in group:
            yield batch, mask, denominator


def token_weighted_loss(model, batch, mask, denominator=None):
    if denominator is None:
        return conversation_loss(model, batch, mask)
    logits = model(batch[:, :-1])
    targets = mask[:, 1:]
    ce = nn.losses.cross_entropy(logits.astype(mx.float32), batch[:, 1:])
    return mx.where(targets, ce, 0).sum() / denominator, targets.sum()


def batch_iterator(accumulation):
    return partial(weighted_batches, accumulation=accumulation)
