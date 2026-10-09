"""A fresh low-rank update around frozen SFT adapter weights.

Combines residuals before their dtype cast. Export concatenates the two factor
pairs, so deployment uses ordinary MLX adapters, without dequantizing the base.
Floating-point reduction order can differ on export; evaluate exported weights.
"""
import math
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten
from mlx_lm.tuner.lora import LoRALinear, LoRAEmbedding


class CharacterDelta(nn.Module):
    def __init__(self, base, rank, dropout):
        super().__init__()
        if not isinstance(base, (LoRALinear, LoRAEmbedding)) or rank < 1:
            raise ValueError('A supported SFT adapter and positive new rank are required')
        self.base = base
        self.base.freeze()
        self.scale = base.scale
        self.is_embedding = isinstance(base, LoRAEmbedding)
        self.dropout = nn.Dropout(dropout)
        # SFT is the frozen reference policy, so its old training dropout is off.
        self.base.dropout = nn.Identity()
        left, right = base.lora_a.shape[0], base.lora_b.shape[1]
        bound = 1 / math.sqrt(left)
        self.lora_a = mx.random.uniform(low=-bound, high=bound, shape=(left, rank))
        self.lora_b = mx.zeros((rank, right))

    def __call__(self, x):
        if self.is_embedding:
            y = self.base.embedding(x)
            old = self.base.lora_a[x] @ self.base.lora_b
            new = self.dropout(self.lora_a[x] @ self.lora_b)
        else:
            y = self.base.linear(x)
            old = (x @ self.base.lora_a) @ self.base.lora_b
            new = (self.dropout(x) @ self.lora_a) @ self.lora_b
        return y + (self.scale * (old + new)).astype(y.dtype)

    def as_linear(self, x):
        if not self.is_embedding:
            raise TypeError('Only tied embeddings support as_linear')
        y = self.base.embedding.as_linear(x)
        old = (x @ self.base.lora_b.T) @ self.base.lora_a.T
        new = (self.dropout(x) @ self.lora_b.T) @ self.lora_a.T
        return y + (self.scale * (old + new)).astype(x.dtype)

    def exported_factors(self):
        return (mx.concatenate([self.base.lora_a, self.lora_a], axis=1),
                mx.concatenate([self.base.lora_b, self.lora_b], axis=0))


def attach_character_delta(model, rank, dropout):
    model.freeze()
    replacements = [(name, CharacterDelta(module, rank, dropout))
                    for name, module in list(model.named_modules())
                    if isinstance(module, (LoRALinear, LoRAEmbedding))]
    if not replacements:
        raise ValueError('No SFT LoRA modules found')
    model.update_modules(tree_unflatten(replacements))
    return len(replacements)


def export_character_delta(model):
    exported = {}
    for name, module in model.named_modules():
        if isinstance(module, CharacterDelta):
            a, b = module.exported_factors()
            exported[name+'.lora_a'] = a
            exported[name+'.lora_b'] = b
    if not exported:
        raise ValueError('No character delta to export')
    return exported
