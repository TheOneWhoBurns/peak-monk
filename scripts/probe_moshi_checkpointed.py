"""Measure full-model checkpointed LoRA memory on a real training episode.

This disposable update is a feasibility measurement, not a release checkpoint.
"""
import argparse
from functools import partial
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten,tree_map
from moshi_mlx.models import Lm,config_v0_1
from moshi_training import attach_lora,loss
from moshi_split import delay_streams
p=argparse.ArgumentParser()
p.add_argument('--episode',default='name')
p.add_argument('--frames',type=int,default=16,help='0 means complete training episode')
p.add_argument('--output',required=True)
args=p.parse_args();out=Path(args.output)
if out.exists():raise ValueError('Preserve previous probe')
weight=Path('models/moshi/model.q4.safetensors')
with weight.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()=='7959d590e23c1ebc78cfa3501344a6ff331561aa0cadc4429b733b890bbc919c'
path=Path('data/speech_codes')/(args.episode+'.npz');meta=json.loads(path.with_suffix('.json').read_text())
assert meta['split']=='train'
raw=np.load(path)['codes'];original_frames=raw.shape[1]
if args.frames:raw=raw[:,:args.frames]
mx.set_memory_limit(5_600_000_000);mx.set_cache_limit(32_000_000)
mx.set_wired_limit(mx.metal.device_info()['max_recommended_working_set_size'])
mx.random.seed(41);model=Lm(config_v0_1());model.set_dtype(mx.bfloat16)
nn.quantize(model,bits=4,group_size=32);model.load_weights(str(weight),strict=True)
attach_lora(model,layers=32,rank=8,scale=2);mx.eval(model.parameters());model.train()
params=sum(v.size for _,v in tree_flatten(model.trainable_parameters()))
print('Loaded',mx.get_active_memory(),'bytes;',params,'trainable parameters',flush=True)
streams=delay_streams(mx.array(raw[None,:,:]),model.cfg)
optimizer=optim.AdamW(learning_rate=1e-4,weight_decay=.01)
fn=nn.value_and_grad(model,partial(loss,text_padding_weight=.05,checkpoint_layers=True))
start=time.monotonic();value,grad=fn(model,streams)
norm=mx.sqrt(sum(mx.sum(g.astype(mx.float32)**2) for _,g in tree_flatten(grad)))
grad=tree_map(lambda g:g*mx.minimum(1.,1./(norm+1e-8)),grad)
optimizer.update(model,grad);mx.eval(value,norm,model.parameters(),optimizer.state)
assert mx.isfinite(value).item() and mx.isfinite(norm).item()
report=dict(scope='Disposable actual-episode feasibility update; no adapter saved, no quality claim',
    episode=args.episode,frames=raw.shape[1],original_frames=original_frames,checkpoint_layers=True,
    trainable_layers=32,trainable_parameters=params,rank=8,scale=2,loss=value.item(),gradient_norm=norm.item(),
    seconds=time.monotonic()-start,peak_memory_bytes=mx.get_peak_memory(),active_memory_bytes=mx.get_active_memory(),
    within_recommended_working_set=mx.get_peak_memory()<=mx.metal.device_info()['max_recommended_working_set_size'])
out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
