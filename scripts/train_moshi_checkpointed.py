"""Full-temporal native LoRA with activation checkpointing on reviewed audio.

All pretrained weights stay quantized and frozen. Adapters in all 32 temporal
blocks and the text head train with the joint text/audio objective.
"""
import argparse
from functools import partial
import hashlib
import json
from pathlib import Path
import random
import subprocess
import time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map
from moshi_split import delay_streams
from moshi_training import attach_lora, loss
from moshi_mlx.models import Lm, config_v0_1
from moshi_windowed import episode_grad,offload_optimizer,restore_optimizer

p=argparse.ArgumentParser()
p.add_argument('--data',default='data/speech_codes')
p.add_argument('--output',required=True)
p.add_argument('--steps',type=int,default=160)
p.add_argument('--layers',type=int,default=32)
p.add_argument('--rank',type=int,default=8)
p.add_argument('--scale',type=float,default=2.0)
p.add_argument('--lr',type=float,default=1e-4)
p.add_argument('--seed',type=int,default=41)
p.add_argument('--evaluate-every',type=int,default=20)
p.add_argument('--gradient-window',type=int,default=0,
    help='Positive values use truncated BPTT with full carried attention context')
p.add_argument('--offload-optimizer',action='store_true')
p.add_argument('--text-padding-weight',type=float,default=.05,
    help='Training only; frozen validation remains at 0.5 for comparability')
args=p.parse_args()
if not 0 < args.text_padding_weight <= 1:
    raise ValueError('Text padding weight must be in (0, 1]')
out=Path(args.output)
if out.exists(): raise ValueError('Use a fresh experiment output directory')
out.mkdir(parents=True)
if args.layers!=32:raise ValueError('This experiment trains all32temporalblocks')
rows=[]
for path in sorted(Path(args.data).glob('*.npz')):
    row=json.loads(path.with_suffix('.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest()==row['file_sha256'],path
    row['path']=str(path);rows.append(row)
train=[r for r in rows if r['split']=='train'];valid=[r for r in rows if r['split']=='valid']
assert train and valid,'Independent training and validation audio are required'
mx.random.seed(args.seed);rng=random.Random(args.seed)
mx.metal.set_memory_limit(5_600_000_000);mx.metal.set_cache_limit(32_000_000)
mx.metal.set_wired_limit(mx.metal.device_info()['max_recommended_working_set_size'])
weight=Path('models/moshi/model.q4.safetensors')
with weight.open('rb') as f:
    assert hashlib.file_digest(f,'sha256').hexdigest()=='7959d590e23c1ebc78cfa3501344a6ff331561aa0cadc4429b733b890bbc919c'
model=Lm(config_v0_1());model.set_dtype(mx.bfloat16)
nn.quantize(model,bits=4,group_size=32);model.load_weights(str(weight),strict=True)
attach_lora(model,layers=args.layers,rank=args.rank,scale=args.scale)
mx.eval(model.parameters())
optimizer=optim.AdamW(learning_rate=args.lr,weight_decay=.01)
optimizer_file=out/'optimizer-offload.safetensors'
value_grad=nn.value_and_grad(model,partial(loss,text_padding_weight=args.text_padding_weight,checkpoint_layers=True))
metadata=dict(vars(args),train_ids=[r['id'] for r in train],valid_ids=[r['id'] for r in valid],
    source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
    trainable_parameters=sum(v.size for _,v in tree_flatten(model.trainable_parameters())),
    model_sha256='7959d590e23c1ebc78cfa3501344a6ff331561aa0cadc4429b733b890bbc919c',
    loss='text CE + semantic-weighted agent audio CE, original delayed streams',
    validation_text_padding_weight=.5,
    checkpoint_layers=True,
    gradient_policy='Truncated BPTT; one update after a complete episode' if args.gradient_window else 'Full episode',
    data_hashes={r['id']:r['file_sha256'] for r in rows})
(out/'config.json').write_text(json.dumps(metadata,indent=2)+'\n')
log=out/'metrics.jsonl'


def record(row):
    row['peak_memory_bytes']=mx.metal.get_peak_memory()
    with log.open('a') as f:f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)


def load_episode(row):
    raw=mx.array(np.load(row['path'])['codes'][None,:,:])
    return delay_streams(raw,model.cfg)


def export_adapters(model, layers, path):
    mx.save_safetensors(str(path),dict(tree_flatten(model.trainable_parameters())))


def evaluate(step):
    model.eval();values=[]
    for row in valid:
        data=load_episode(row)
        value=loss(model,data,text_padding_weight=.5,checkpoint_layers=True);mx.eval(value)
        values.append((value.item(),row['frames']))
        del data,value;mx.metal.clear_cache()
    # Full-episode objectives, weighted by frame count; report aggregation explicitly.
    result=sum(v*n for v,n in values)/sum(n for _,n in values)
    record({'kind':'validation','step':step,'frame_weighted_episode_loss':result})
    return result


baseline=evaluate(0);best=baseline;start=time.monotonic()
order=list(train);rng.shuffle(order)
for step in range(1,args.steps+1):
    if (step-1)%len(order)==0: rng.shuffle(order)
    row=order[(step-1)%len(order)];data=load_episode(row)
    model.train();tick=time.monotonic()
    if args.gradient_window:
        value,grad=episode_grad(model,data,window=args.gradient_window,padding_weight=args.text_padding_weight)
    else:
        value,grad=value_grad(model,data)
    norm=mx.sqrt(sum(mx.sum(g.astype(mx.float32)**2) for _,g in tree_flatten(grad)))
    grad=tree_map(lambda g:g*mx.minimum(1.,1./(norm+1e-8)),grad)
    mx.eval(value,norm)
    if not mx.isfinite(value).item() or not mx.isfinite(norm).item():
        raise FloatingPointError(f'Non-finite loss or gradient before update {step}, episode {row["id"]}')
    if args.offload_optimizer and optimizer_file.exists():restore_optimizer(optimizer,optimizer_file)
    optimizer.update(model,grad);mx.eval(value,norm,model.parameters(),optimizer.state)
    if not mx.isfinite(value).item() or not mx.isfinite(norm).item(): raise FloatingPointError('Non-finite speech update')
    record({'kind':'train','step':step,'family':row['id'],'loss':value.item(),
            'gradient_norm_before_clip':norm.item(),'seconds':time.monotonic()-tick})
    del data,grad,value,norm;mx.metal.clear_cache()
    if args.offload_optimizer:offload_optimizer(optimizer,optimizer_file)
    if step%args.evaluate_every==0 or step==args.steps:
        val=evaluate(step)
        export_adapters(model,args.layers,out/f'step-{step:04d}.safetensors')
        if val<best:
            best=val;export_adapters(model,args.layers,out/'candidate.safetensors')
            (out/'candidate.json').write_text(json.dumps({'step':step,'validation_loss':val,
                'status':'Loss-selected candidate; native streaming and speech quality audit still required'},indent=2)+'\n')
record({'kind':'completion','steps':args.steps,'seconds':time.monotonic()-start,'baseline_loss':baseline,'best_loss':best})
