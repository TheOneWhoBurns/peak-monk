"""Check value, every adapter gradient, and dropout RNG on a tiny actual Qwen3."""
from functools import partial
import json
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.models.qwen3 import Model, ModelArgs
from mlx_lm.tuner.lora import LoRAEmbedding, LoRALinear
from conversation_objective import conversation_loss
from staged_character_grad import staged_character_grad
from qwen_rng_checkpoint import qwen_rng_checkpoint

mx.set_default_device(mx.cpu)
mx.random.seed(410)
args = ModelArgs(model_type='qwen3', hidden_size=32, num_hidden_layers=2,
    intermediate_size=64, num_attention_heads=4, rms_norm_eps=1e-6,
    vocab_size=64, num_key_value_heads=2, max_position_embeddings=256,
    head_dim=8, tie_word_embeddings=True, rope_theta=1e6)
model = Model(args)
nn.quantize(model, bits=4, group_size=32)
model.freeze()
model.model.embed_tokens = LoRAEmbedding.from_base(model.model.embed_tokens, r=2, scale=2, dropout=.2)
for layer in model.layers:
    for parent, names in [(layer.self_attn, ['q_proj','k_proj','v_proj','o_proj']), (layer.mlp, ['gate_proj','up_proj','down_proj'])]:
        for name in names:
            setattr(parent,name,LoRALinear.from_base(getattr(parent,name),r=2,scale=2,dropout=.2))
# Nonzero B exercises gradients of both factors and both uses of the tied head.
for key, value in tree_flatten(model.trainable_parameters()):
    if key.endswith('lora_b'):
        target=model
        parts=key.split('.')
        for part in parts[:-1]: target=target[int(part)] if isinstance(target,list) else getattr(target,part)
        setattr(target,parts[-1],mx.random.normal(value.shape)*.02)
model.train();mx.eval(model.parameters())
batch=mx.array([[1,2,3,4,5,6,7,8,9,10,11],[3,2,1,5,4,8,7,6,9,0,0]])
mask=mx.array([[False,False,True,True,False,True,True,True,True,True,True],
               [False,True,True,False,True,True,False,True,True,False,False]])

def capture(fn, staged=False):
    mx.random.seed(413)
    (value,count),grad=(fn(model,batch,mask) if staged else nn.value_and_grad(model,fn)(model,batch,mask))
    mx.eval(value,count,grad)
    next_random=mx.random.uniform(shape=(7,));mx.eval(next_random)
    return value,count,dict(tree_flatten(grad)),next_random

value,count,grad,rng=capture(conversation_loss)
qwen_rng_checkpoint(model.layers[0])
checks=[]
for size in [3,8,64]:
    actual,n,g,r=capture(partial(staged_character_grad,chunk_size=size),staged=True)
    errors={k:float(mx.max(mx.abs(grad[k]-g[k])).item()) for k in grad}
    loss_error=float(mx.abs(value-actual).item())
    rng_error=float(mx.max(mx.abs(rng-r)).item())
    print(dict(chunk_size=size,loss_error=loss_error,maximum_gradient_error=max(errors.values()),rng_error=rng_error),flush=True)
    assert int(n.item())==int(count.item())
    assert loss_error < 1e-5 and max(errors.values()) < 1e-5 and rng_error==0
    checks.append(dict(chunk_size=size,loss_error=loss_error,maximum_gradient_error=max(errors.values()),rng_error=rng_error))
report=dict(status='pass',checks=checks,target_tokens=int(count.item()),
            scope='Tiny actual quantized Qwen3 with tied LoRA embedding, nonzero adapters and head dropout. All seven projections per layer plus head dropout; compare uncheckpointed reference against checkpointed staged gradients and RNG. Real-model memory and quality unmeasured.')
Path('results/staged-character-gradient-check.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
