"""Tiny full-context/windowed forward parity; truncated gradients are intentional."""
from dataclasses import replace
import json
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten,tree_map
from moshi_mlx.models import Lm,config_v0_1
from moshi_training import attach_lora,loss,temporal_forward
from moshi_windowed import temporal_chunk,chunk_loss,normalizers,episode_grad
mx.random.seed(7)
cfg=config_v0_1()
cfg=replace(cfg,transformer=replace(cfg.transformer,d_model=32,num_heads=4,num_layers=2,dim_feedforward=128),
 depformer=replace(cfg.depformer,transformer=replace(cfg.depformer.transformer,d_model=16,num_heads=2,num_layers=2,dim_feedforward=64)),
 text_in_vocab_size=33,text_out_vocab_size=32,audio_vocab_size=17)
model=Lm(cfg);tokens=mx.random.randint(0,16,(1,17,13));attach_lora(model,layers=2,rank=2)
_,full=temporal_forward(model,tokens[:,:,:-1]);state=None;parts=[]
for start in range(0,12,4):
 _,out,state=temporal_chunk(model,tokens[:,:,start:start+4],state)
 parts.append(out);state=tree_map(mx.stop_gradient,state)
forward_error=mx.max(mx.abs(full-mx.concatenate(parts,axis=1))).item()
expected=loss(model,tokens,text_padding_weight=.05)
actual,grad=episode_grad(model,tokens,window=4,padding_weight=.05)
loss_error=abs(expected.item()-actual.item())
norm=sum(mx.sum(g**2).item() for _,g in tree_flatten(grad))**.5
assert forward_error<1e-5 and loss_error<1e-5
assert norm>0 and mx.isfinite(actual).item()
# One window has no truncation boundary and must reproduce the full gradient.
v,g=nn.value_and_grad(model,lambda m,t:loss(m,t,text_padding_weight=.05))(model,tokens)
w,wg=episode_grad(model,tokens,window=12,padding_weight=.05)
grad_error=max(mx.max(mx.abs(x-dict(tree_flatten(g))[k])).item() for k,x in tree_flatten(wg))
assert grad_error<1e-5
report=dict(scope='Tiny implementation test, not model quality',forward_error=forward_error,loss_error=loss_error,
 truncated_gradient_norm=norm,single_window_gradient_error=grad_error,multi_window_gradient_policy='Detached prior attention state; deliberately truncated credit assignment',passed=True)
Path('results/moshi-windowed-check.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
