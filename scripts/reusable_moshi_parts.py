"""Inference-only temporal module reuse; weights and per-layer KV remain distinct.

The caller must finish evaluating each layer before requesting the next one.
This changes object construction, never quantization, trained weights or sampling.
"""
import time
import mlx.core as mx
from moshi_weight_stream import MoshiParts


class ReusableMoshiParts(MoshiParts):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._temporal = None
        self.load_count = 0
        self.load_seconds = 0.0

    def load(self, part):
        if not isinstance(part, int):
            return super().load(part)
        if not 0 <= part < self.cfg.transformer.num_layers:
            raise ValueError('Invalid temporal layer')
        start = time.monotonic()
        if self._temporal is None:
            self._temporal = super().load(part)
            self._temporal.eval()
        else:
            prefix = f'transformer.layers.{part}.'
            weights = self.weights.load([prefix], strip=prefix)
            if self.adapter:
                # The trained wrapper stores frozen linear tensors under .base.
                remapped = {}
                for key, value in weights.items():
                    for wrapped in ('self_attn.out_proj.', 'gating.linear_out.'):
                        if key.startswith(wrapped):
                            key = wrapped + 'base.' + key[len(wrapped):]
                            break
                    remapped[key] = value
                del weights
                weights = remapped
                weights.update(self.adapter.load([prefix], strip=prefix))
            self._temporal.load_weights(list(weights.items()), strict=True)
            del weights
            mx.eval(self._temporal.parameters())
        self.load_count += 1
        self.load_seconds += time.monotonic() - start
        return self._temporal
