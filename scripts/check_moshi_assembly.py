"""Compare full-model weights and native sample trace after selective assembly."""
import argparse,json,tempfile
from pathlib import Path
from dataclasses import replace
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from moshi_mlx.models import Lm,LmGen,config_v0_1
from moshi_mlx.utils import Sampler
from moshi_training import attach_lora
from moshi_weight_stream import MoshiParts
from assemble_moshi_parts import assemble,assemble_streamed
from native_tensor_reader import NativeTensorRanges
from reusable_moshi_parts import ReusableMoshiParts
from selective_native_reader import SelectiveNativeRanges
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
mx.set_default_device(mx.gpu);mx.random.seed(281)
cfg=config_v0_1();cfg=replace(cfg,transformer=replace(cfg.transformer,d_model=64,num_heads=4,num_layers=2,dim_feedforward=192),depformer=replace(cfg.depformer,transformer=replace(cfg.depformer.transformer,d_model=32,num_heads=4,num_layers=2,dim_feedforward=96)),text_in_vocab_size=33,text_out_vocab_size=32,audio_vocab_size=33)
model=Lm(cfg);model.set_dtype(mx.bfloat16);nn.quantize(model,bits=4,group_size=32)
with tempfile.TemporaryDirectory() as tmp:
 root=Path(tmp);weight=root/'base.safetensors';adapter=root/'adapter.safetensors'
 mx.save_safetensors(str(weight),dict(tree_flatten(model.parameters())))
 attach_lora(model,layers=2,rank=2,scale=2.)
 for k,v in tree_flatten(model.trainable_parameters()):
  if k.endswith('.b'):model.load_weights([(k,mx.random.normal(v.shape)*.02)],strict=False)
 mx.save_safetensors(str(adapter),dict(tree_flatten(model.trainable_parameters())));model.eval()
 parts=MoshiParts(weight,cfg,adapter,rank=2,scale=2.)
 parts.weights=NativeTensorRanges(weight);parts.adapter=NativeTensorRanges(adapter)
 loaded=assemble(parts)
 expected=dict(tree_flatten(model.parameters()));actual=dict(tree_flatten(loaded.parameters()))
 assert set(expected)==set(actual)
 assert all(v.dtype==actual[k].dtype and mx.array_equal(v,actual[k]).item() for k,v in expected.items())
 codes=mx.random.randint(0,31,(1,8,12));mx.eval(codes)
 def trace(current):
  mx.random.seed(17);gen=LmGen(current,max_steps=17,text_sampler=Sampler(),audio_sampler=Sampler());rows=[]
  for i in range(12):
   output=gen.step(codes[:,:,i]);token=output[0] if isinstance(output,tuple) else output;audio=gen.last_audio_tokens()
   rows.append(dict(text=token.tolist(),audio=None if audio is None else audio.tolist()))
  return rows
 streamed=assemble_streamed(parts)
 reuse_parts=ReusableMoshiParts(weight,cfg,adapter,rank=2,scale=2.)
 reuse_parts.weights=NativeTensorRanges(weight);reuse_parts.adapter=NativeTensorRanges(adapter)
 reused=assemble_streamed(reuse_parts)
 ref=trace(model);result=trace(loaded);stream_trace=trace(streamed);reused_trace=trace(reused)
 selective_parts=ReusableMoshiParts(weight,cfg,adapter,rank=2,scale=2.)
 selective_parts.weights=SelectiveNativeRanges(weight);selective_parts.adapter=SelectiveNativeRanges(adapter)
 for part in ['inputs',0,1,'heads']:
  native=dict(tree_flatten(parts.load(part).parameters()))
  selective=dict(tree_flatten(selective_parts.load(part).parameters()))
  assert set(native)==set(selective)
  assert all(v.dtype==selective[k].dtype and mx.array_equal(v,selective[k]).item() for k,v in native.items())
 selective_trace=trace(assemble_streamed(selective_parts))
 assert ref==result==stream_trace==reused_trace==selective_trace
 report=dict(status='pass',parameter_tensors=len(expected),matched_native_frames=len(ref),scope='Tiny actual quantized Moshi with nonzero adapters; every assembled and selective-reader parameter exact; resident, disk-streamed, reusable-module and virtual-file native text/audio sampling traces exact, including initializer RNG preservation. Real model memory remains separate.')
 a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
