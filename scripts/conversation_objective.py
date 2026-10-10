"""Train every assistant reply once per conversation, without predicting callers.

This changes training weighting, not the frozen last-reply validation metric.
Windows retain 128 tokens of context; overlapping targets are never counted twice.
"""
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.tuner.trainer import iterate_batches as upstream_batches
from assistant_objective import assistant_loss


def encode_conversation(row, tokenizer):
    messages = row['messages']
    if any(m['role'] not in ('user', 'assistant') for m in messages):
        raise ValueError('Character training accepts conversation only, no system persona')
    selected = row.get('supervised_message_indices')
    if selected is not None:
        if (not isinstance(selected, list) or not selected or
            any(type(i) is not int or i < 0 or i >= len(messages) or
                messages[i]['role'] != 'assistant' for i in selected) or
            len(set(selected)) != len(selected)):
            raise ValueError('Supervision indices must select distinct assistant messages')
        selected = set(selected)
    full = tokenizer.apply_chat_template(messages, return_dict=False)
    mask = [False] * len(full)
    for i, message in enumerate(messages):
        if message['role'] != 'assistant':
            continue
        if selected is not None and i not in selected:
            continue
        prefix = tokenizer.apply_chat_template(messages[:i], add_generation_prompt=True, return_dict=False)
        completed = tokenizer.apply_chat_template(messages[:i + 1], return_dict=False)
        start = len(prefix)
        if full[:start] != prefix or full[:len(completed)] != completed:
            raise ValueError('Template is not causal across conversation prefixes')
        end = next(j + 1 for j in range(start, len(completed)) if completed[j] == tokenizer.eos_token_id)
        mask[start:end] = [True] * (end - start)
    if not any(mask):
        raise ValueError('No assistant targets')
    return full, mask


def windows(tokens, mask, size, overlap=128):
    if not 0 < overlap < size:
        raise ValueError('Context overlap must fit window')
    previous_end = 0
    start = 0
    while start < len(tokens):
        end = min(start + size, len(tokens))
        local_mask = [bool(mask[j] and j >= previous_end and j > start) for j in range(start, end)]
        if any(local_mask):
            yield tokens[start:end], local_mask
        if end == len(tokens):
            break
        previous_end = end
        start = end - overlap


class ConversationDataset:
    def __init__(self, rows, tokenizer, max_length):
        self.items = [item for row in rows for item in windows(*encode_conversation(row, tokenizer), max_length)]
    def __len__(self): return len(self.items)
    def __getitem__(self, index): return self.items[index]
    def process(self, item): return item


def conversation_batches(dataset, batch_size, max_seq_length, loop=False, comm_group=None):
    if not isinstance(dataset._data, ConversationDataset):
        yield from upstream_batches(dataset=dataset, batch_size=batch_size, max_seq_length=max_seq_length, loop=loop, comm_group=comm_group)
        return
    if comm_group is not None and comm_group.size() != 1:
        raise ValueError('This local conversation iterator is single-device only')
    order = np.arange(len(dataset))
    while True:
        if loop: np.random.shuffle(order)
        for offset in range(0, len(order), batch_size):
            rows = [dataset[int(i)] for i in order[offset:offset + batch_size]]
            size = max(len(row[0]) for row in rows)
            batch = np.zeros((len(rows), size), dtype=np.int32)
            mask = np.zeros((len(rows), size), dtype=np.bool_)
            for i, (tokens, targets) in enumerate(rows):
                batch[i, :len(tokens)] = tokens
                mask[i, :len(tokens)] = targets
            yield mx.array(batch), mx.array(mask)
        if not loop: break


def conversation_loss(model, batch, mask_or_lengths):
    if mask_or_lengths.dtype != mx.bool_:
        return assistant_loss(model, batch, mask_or_lengths)
    logits = model(batch[:, :-1])
    mask = mask_or_lengths[:, 1:]
    ce = nn.losses.cross_entropy(logits.astype(mx.float32), batch[:, 1:])
    count = mask.sum()
    return mx.where(mask, ce, 0).sum() / mx.maximum(count, 1), count
