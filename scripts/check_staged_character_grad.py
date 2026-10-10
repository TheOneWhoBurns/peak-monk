"""Check value, every adapter gradient, and dropout RNG on a tiny actual Qwen3."""
from functools import partial
import argparse
import json
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map
from mlx_lm.models.qwen3 import Model, ModelArgs
from mlx_lm.tuner.lora import LoRAEmbedding, LoRALinear
from conversation_objective import conversation_loss
from staged_character_grad import staged_character_grad
from qwen_rng_checkpoint import qwen_rng_checkpoint

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, default=Path('results/staged-character-gradient-check.json'))
cli = parser.parse_args()

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
ordinary_mask = mask
mask = mx.array([[False]*9+[True,True], [False]*10+[True]])
late_value, late_count, late_grad, late_rng = capture(conversation_loss)
mask = ordinary_mask
# Unequal sequence lengths and target counts exercise the actual trainer's
# denominator override. A mean of microbatch means would fail this check.
group = [(batch[:1, :8], mask[:1, :8]), (batch[1:, :10], mask[1:, :10])]
denominator = sum(int(m[:, 1:].sum().item()) for _, m in group)

def capture_group(staged):
    mx.random.seed(917)
    total = mx.array(0.)
    accumulated = None
    for x, selected in group:
        if staged:
            (loss, _), part = staged_character_grad(model, x, selected,
                chunk_size=3, denominator=denominator)
        else:
            def objective(current):
                mean, number = conversation_loss(current, x, selected)
                return mean * number / denominator
            loss, part = nn.value_and_grad(model, objective)(model)
        total = total + loss
        accumulated = part if accumulated is None else tree_map(lambda a, b: a + b, accumulated, part)
        mx.eval(total, accumulated)
    next_random = mx.random.uniform(shape=(7,))
    mx.eval(next_random)
    return total, dict(tree_flatten(accumulated)), next_random

group_value, group_grad, group_rng = capture_group(False)
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
gv, gg, gr = capture_group(True)
group_check = dict(loss_error=abs(gv.item()-group_value.item()),
    maximum_gradient_error=max(float(mx.max(mx.abs(group_grad[k]-gg[k])).item()) for k in gg),
    rng_error=float(mx.max(mx.abs(group_rng-gr)).item()), target_tokens=denominator)
assert group_check['loss_error'] < 1e-5 and group_check['maximum_gradient_error'] < 1e-5 and group_check['rng_error'] == 0
mask = mx.array([[False]*9+[True,True], [False]*10+[True]])
lv, ln, lg, lr = capture(partial(staged_character_grad, chunk_size=3), staged=True)
late_check = dict(loss_error=abs(lv.item()-late_value.item()),
    maximum_gradient_error=max(float(mx.max(mx.abs(late_grad[k]-lg[k])).item()) for k in lg),
    rng_error=float(mx.max(mx.abs(late_rng-lr)).item()), target_tokens=int(ln.item()))
assert late_check['loss_error'] < 1e-5 and late_check['maximum_gradient_error'] < 1e-5 and late_check['rng_error'] == 0
from qwen_chunked_attention import chunk_qwen_attention
chunk_qwen_attention(model, 3)
av, an, ag, ar = capture(partial(staged_character_grad, chunk_size=3), staged=True)
attention_check = dict(loss_error=abs(av.item()-late_value.item()),
    maximum_gradient_error=max(float(mx.max(mx.abs(late_grad[k]-ag[k])).item()) for k in ag),
    rng_error=float(mx.max(mx.abs(late_rng-ar)).item()), target_tokens=int(an.item()))
assert attention_check['loss_error'] < 1e-5 and attention_check['maximum_gradient_error'] < 1e-5 and attention_check['rng_error'] == 0
for layer in model.layers:
    layer.self_attn._jun_query_chunk_size = 0
model.model.embed_tokens.freeze()
model.model.embed_tokens.eval()
model.layers[0].freeze()
model.layers[0].eval()
fv, fn, fg, fr = capture(conversation_loss)
chunk_qwen_attention(model, 3)
sv, sn, sg, sr = capture(partial(staged_character_grad, chunk_size=3), staged=True)
frozen_check = dict(loss_error=abs(fv.item()-sv.item()),
    maximum_gradient_error=max(float(mx.max(mx.abs(fg[k]-sg[k])).item()) for k in sg),
    rng_error=float(mx.max(mx.abs(fr-sr)).item()), trainable_tensors=len(sg))
assert not any('embed_tokens' in k or '.layers.0.' in k for k in sg)
assert frozen_check['loss_error'] < 1e-5 and frozen_check['maximum_gradient_error'] < 1e-5 and frozen_check['rng_error'] == 0
from mlx_lm.models.base import create_attention_mask
prefix_hidden = model.model.embed_tokens(batch[:, :-1])
prefix_hidden = model.layers[0](prefix_hidden, create_attention_mask(prefix_hidden, None))
mx.eval(prefix_hidden)
model.model.layers = model.layers[1:]
cv, cn, cg, cr = capture(partial(staged_character_grad, chunk_size=3, input_embeddings=prefix_hidden), staged=True)
mapped = {k.replace('model.layers.0.', 'model.layers.1.'): v for k, v in cg.items()}
cache_check = dict(loss_error=abs(cv.item()-fv.item()),
    maximum_gradient_error=max(float(mx.max(mx.abs(fg[k]-mapped[k])).item()) for k in fg),
    rng_error=float(mx.max(mx.abs(fr-cr)).item()), target_tokens=int(cn.item()))
assert cache_check['loss_error'] < 1e-5 and cache_check['maximum_gradient_error'] < 1e-5 and cache_check['rng_error'] == 0
report=dict(status='pass',checks=checks,target_tokens=int(count.item()),
            unequal_microbatch_accumulation=group_check,
            late_targets_with_unsupervised_prefix=late_check,
            query_chunked_full_history_attention=attention_check,
            frozen_embedding_and_lower_layer=frozen_check,
            cached_frozen_prefix_and_unloaded_lower_layer=cache_check,
            scope='Tiny actual quantized Qwen3 with tied LoRA embedding, nonzero adapters and head dropout. All seven projections per layer plus head dropout; compare uncheckpointed reference against checkpointed staged gradients and RNG. Real-model memory and quality unmeasured.')
cli.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
