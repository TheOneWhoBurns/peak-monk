"""Actual streamed loader/gradient/export parity on a tiny quantized Moshi."""
from dataclasses import replace
import argparse,json,tempfile
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten,tree_map
from moshi_mlx.models import Lm,config_v0_1
from moshi_training import attach_lora,loss
from moshi_weight_stream import MoshiParts
from moshi_disk_adjoint import disk_episode_grad,disk_adamw_update
from moshi_disk_metrics import disk_metrics
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
mx.set_default_device(mx.gpu);mx.random.seed(173)
cfg=config_v0_1();cfg=replace(cfg,transformer=replace(cfg.transformer,d_model=64,num_heads=4,num_layers=2,dim_feedforward=192),depformer=replace(cfg.depformer,transformer=replace(cfg.depformer.transformer,d_model=32,num_heads=4,num_layers=2,dim_feedforward=96)),text_in_vocab_size=33,text_out_vocab_size=32,audio_vocab_size=33)
model=Lm(cfg);model.set_dtype(mx.bfloat16);nn.quantize(model,bits=4,group_size=32)
reports=[]
with tempfile.TemporaryDirectory() as directory:
 root=Path(directory);base=root/'base.safetensors';mx.save_safetensors(str(base),dict(tree_flatten(model.parameters())))
 attach_lora(model,layers=2,rank=2,scale=2.)
 for key,value in tree_flatten(model.trainable_parameters()):
  if key.endswith('.b'):
   model.load_weights([(key,mx.random.normal(value.shape)*.02)],strict=False)
 adapter=root/'adapter0.safetensors';mx.save_safetensors(str(adapter),dict(tree_flatten(model.trainable_parameters())))
 tokens=mx.random.randint(0,31,(1,17,14));tokens[:,0,1:3]=mx.array([0,3]);tokens[:,1:4,-1]=-1
 optimizer=optim.AdamW(learning_rate=1e-4,weight_decay=.01)
 for step in range(1,3):
  reference,gradient=nn.value_and_grad(model,lambda m:loss(m,tokens,text_padding_weight=.05,checkpoint_layers=False))(model);mx.eval(reference,gradient)
  parts=MoshiParts(base,cfg,adapter,rank=2,scale=2.)
  diagnostic=disk_metrics(parts,tokens,head_frames=4)
  diagnostic_error=abs(diagnostic['joint_ce_by_padding_weight']['0.05']-reference.item())
  assert diagnostic_error<.02
  actual=disk_episode_grad(parts,tokens,root/f'gradient{step}',head_frames=4)
  flat={}
  for row in actual['gradients']:
   prefix='text_linear.' if row['part']=='heads' else f"transformer.layers.{row['part']}."
   flat.update({prefix+k:v for k,v in mx.load(row['path']).items()})
  expected=dict(tree_flatten(gradient));assert set(flat)==set(expected)
  norm=mx.sqrt(sum(mx.sum(v.astype(mx.float32)**2) for v in expected.values()));mx.eval(norm)
  graderror=max(mx.max(mx.abs(flat[k]-v)).item() for k,v in expected.items())
  assert abs(reference.item()-actual['loss'])<.02 and graderror<.02
  scale=mx.minimum(1.,.3/(norm+1e-8));optimizer.update(model,tree_map(lambda g:g*scale,gradient));mx.eval(model.parameters(),optimizer.state)
  next_adapter=root/f'adapter{step}.safetensors'
  update=disk_adamw_update(parts,actual,root/'optimizer',next_adapter,clip_norm=.3,expected_step=step)
  saved=mx.load(str(next_adapter));params=dict(tree_flatten(model.trainable_parameters()))
  paramerror=max(mx.max(mx.abs(v-saved[k])).item() for k,v in params.items())
  assert paramerror<1e-6,(step,paramerror)
  reports.append(dict(step=step,loss_error=abs(reference.item()-actual['loss']),maximum_gradient_error=graderror,parameter_error=paramerror,diagnostic_joint_loss_error=diagnostic_error,update=update))
  adapter=next_adapter
r=dict(status='pass',device='gpu',dtype='bfloat16',cases=reports,scope='Tiny quantized Moshi: actual selective tensor reader, disk boundary activations, reverse layer VJPs, global clipping, per-part optimizer state reload, two complete standard adapter exports compared against full-model gradients and updates. Not pretrained feasibility or speech quality.')
a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
