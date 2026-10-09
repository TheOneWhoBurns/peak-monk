"""Compare all fresh-delta gradients and RNG for staged DPO plus SFT replay."""
import json
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.models.qwen3 import Model, ModelArgs
from mlx_lm.tuner.lora import LoRAEmbedding, LoRALinear
from character_delta import attach_character_delta
from conversation_objective import conversation_loss
from train_preferences import objective
from staged_preference_grad import preference_grad
from qwen_rng_checkpoint import qwen_rng_checkpoint

mx.set_default_device(mx.cpu)
mx.random.seed(123)
model = Model(ModelArgs(model_type='qwen3', hidden_size=32, num_hidden_layers=2,
    intermediate_size=64, num_attention_heads=4, rms_norm_eps=1e-6,
    vocab_size=64, num_key_value_heads=2, max_position_embeddings=256,
    head_dim=8, tie_word_embeddings=True, rope_theta=1e6))
nn.quantize(model, bits=4, group_size=32)
model.freeze()
model.model.embed_tokens = LoRAEmbedding.from_base(model.model.embed_tokens, r=2, scale=2., dropout=.2)
for layer in model.layers:
    for parent, names in [(layer.self_attn, ['q_proj','k_proj','v_proj','o_proj']),
                           (layer.mlp, ['gate_proj','up_proj','down_proj'])]:
        for name in names:
            setattr(parent, name, LoRALinear.from_base(getattr(parent,name), r=2, scale=2., dropout=.2))

def nonzero_b():
    for key, value in tree_flatten(model.trainable_parameters()):
        if key.endswith('lora_b'):
            parts = key.split('.')
            current = model
            for part in parts[:-1]:
                current = current[int(part)] if isinstance(current,list) else getattr(current,part)
            setattr(current, parts[-1], mx.random.normal(value.shape)*.02)

nonzero_b()
attach_character_delta(model, rank=1, dropout=.2)
nonzero_b()
model.train()
mx.eval(model.parameters())
chosen = mx.array([[1,2,3,4,5,6,7,8]])
rejected = mx.array([[1,2,3,4,9,10]])
replay = (mx.array([[12,13,14,15,16,17,18,19,20]]),
          mx.array([[False,False,True,True,False,False,True,True,True]]))

def combined(current, replay_weight):
    value = objective(current, chosen, rejected, 4, -.73, .1, .3)
    if replay_weight:
        value = value + replay_weight*conversation_loss(current, *replay)[0]
    return value

references = []
for weight in [0., .25]:
    mx.random.seed(791)
    value, gradient = nn.value_and_grad(model, combined)(model, weight)
    mx.eval(value, gradient)
    rng = mx.random.uniform(shape=(7,))
    mx.eval(rng)
    references.append((weight, value, dict(tree_flatten(gradient)), rng))
qwen_rng_checkpoint(model.layers[0])
checks = []
for weight, expected, expected_grad, expected_rng in references:
    mx.random.seed(791)
    value, gradient = preference_grad(model, chosen, rejected, 4, -.73, .1, .3,
        replay=replay if weight else None, replay_weight=weight, chunk_size=3)
    mx.eval(value, gradient)
    rng = mx.random.uniform(shape=(7,))
    mx.eval(rng)
    errors = {k:float(mx.max(mx.abs(v-expected_grad[k])).item()) for k,v in tree_flatten(gradient)}
    check = dict(replay_weight=weight, loss_error=abs(value.item()-expected.item()),
        maximum_gradient_error=max(errors.values()), rng_error=float(mx.max(mx.abs(rng-expected_rng)).item()))
    assert check['loss_error'] < 1e-5 and check['maximum_gradient_error'] < 1e-5 and check['rng_error'] == 0, check
    checks.append(check)
report = dict(status='pass', checks=checks,
    scope='Tiny quantized Qwen3, frozen nonzero SFT factors plus nonzero fresh deltas, dropout0.2, unequal candidate lengths, ragged head chunks. Every trainable gradient and RNG matches direct uncheckpointed objective. No actual-model memory or quality claim.')
Path('results/staged-preference-gradient-check.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
