"""Checkpoint pinned Qwen3 LoRA blocks while replaying identical dropout keys.

The installed helper retraces random calls during backward. This wrapper
reserves the block's draws once, then restores its starting key during every
recomputation. Only the seven known Qwen3 projection-dropout sites are supported.
No changes are made to installed MLX packages.
"""
import mlx.core as mx
from mlx_lm.tuner.lora import LoRALinear
from staged_character_grad import restore_rng


def snapshot_rng():
    state = list(mx.random.state)
    mx.eval(state)
    return state


def qwen_rng_checkpoint(layer):
    cls = type(layer)
    if getattr(cls, '_jun_rng_checkpoint', False):
        return
    if cls.__module__ != 'mlx_lm.models.qwen3' or cls.__name__ != 'TransformerBlock':
        raise TypeError('Only the pinned Qwen3 TransformerBlock is supported')
    original = cls.__call__
    if original.__name__ != '__call__':
        raise ValueError('Do not stack this wrapper over another checkpoint helper')

    def wrapped(current, *args, **kwargs):
        if not current.training:
            return original(current, *args, **kwargs)
        projections = [current.self_attn.q_proj, current.self_attn.k_proj,
                       current.self_attn.v_proj, current.self_attn.o_proj,
                       current.mlp.gate_proj, current.mlp.up_proj, current.mlp.down_proj]
        start = snapshot_rng()
        # A random operation consumes one key regardless of the sample shape.
        # These scalar outputs are unused; only advancement of the key matters.
        for projection in projections:
            if hasattr(projection, 'dropout') and not isinstance(projection, LoRALinear):
                raise TypeError('Unsupported projection dropout; audit its draw count first')
            if (isinstance(projection, LoRALinear) and projection.dropout.training
                    and projection.dropout._p_1 != 1.0):
                mx.random.uniform(shape=())

        def inner(params, *inputs, **options):
            saved = snapshot_rng()
            restore_rng(start)
            try:
                current.update(params)
                return original(current, *inputs, **options)
            finally:
                restore_rng(saved)

        return mx.checkpoint(inner)(current.trainable_parameters(), *args, **kwargs)

    cls.__call__ = wrapped
    cls._jun_rng_checkpoint = True
