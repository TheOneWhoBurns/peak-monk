"""Train actual speech adapters from reviewed audio and frozen-prefix caches.

The full model's frozen prefix is not resident during this process. Adapters are
exported with original full-model names for native streaming inference.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import time
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map
from moshi_split import load_part, cached_loss, export_adapters
from moshi_training import attach_lora

p=argparse.ArgumentParser()
p.add_argument('--cache',default='data/speech_cache/suffix-4')
p.add_argument('--output',required=True)
p.add_argument('--steps',type=int,default=160)
p.add_argument('--layers',type=int,default=4)
p.add_argument('--rank',type=int,default=8)
p.add_argument('--scale',type=float,default=2.0)
p.add_argument('--lr',type=float,default=1e-5)
p.add_argument('--seed',type=int,default=41)
p.add_argument('--evaluate-every',type=int,default=20)
args=p.parse_args()
out=Path(args.output)
if out.exists(): raise ValueError('Use a fresh experiment output directory')
out.mkdir(parents=True)
rows=json.loads((Path(args.cache)/'manifest.json').read_text())
assert all(r['suffix_layers']==args.layers for r in rows)
train=[r for r in rows if r['split']=='train'];valid=[r for r in rows if r['split']=='valid']
assert train and valid,'Independent training and validation audio are required'
mx.random.seed(args.seed);rng=random.Random(args.seed)
mx.metal.set_memory_limit(5_500_000_000);mx.metal.set_cache_limit(64_000_000)
mx.metal.set_wired_limit(mx.metal.device_info()['max_recommended_working_set_size'])
model=load_part('models/moshi/model.q4.safetensors',args.layers,'suffix')
attach_lora(model,layers=args.layers,rank=args.rank,scale=args.scale)
mx.eval(model.parameters())
optimizer=optim.AdamW(learning_rate=args.lr,weight_decay=.01)
value_grad=nn.value_and_grad(model,cached_loss)
metadata=dict(vars(args),train_ids=[r['id'] for r in train],valid_ids=[r['id'] for r in valid],
    source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
    trainable_parameters=sum(v.size for _,v in tree_flatten(model.trainable_parameters())),
    model_sha256='7959d590e23c1ebc78cfa3501344a6ff331561aa0cadc4429b733b890bbc919c',
    loss='text CE + semantic-weighted agent audio CE, original delayed streams',
    cache_manifest_sha256=hashlib.sha256((Path(args.cache)/'manifest.json').read_bytes()).hexdigest())
(out/'config.json').write_text(json.dumps(metadata,indent=2)+'\n')
log=out/'metrics.jsonl'


def record(row):
    row['peak_memory_bytes']=mx.metal.get_peak_memory()
    with log.open('a') as f:f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)


def evaluate(step):
    model.eval();values=[]
    for row in valid:
        data=mx.load(row['path'])
        value=cached_loss(model,data['hidden'],data['targets']);mx.eval(value)
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
    row=order[(step-1)%len(order)];data=mx.load(row['path'])
    model.train();tick=time.monotonic()
    value,grad=value_grad(model,data['hidden'],data['targets'])
    norm=mx.sqrt(sum(mx.sum(g.astype(mx.float32)**2) for _,g in tree_flatten(grad)))
    grad=tree_map(lambda g:g*mx.minimum(1.,1./(norm+1e-8)),grad)
    optimizer.update(model,grad);mx.eval(value,norm,model.parameters(),optimizer.state)
    if not mx.isfinite(value).item() or not mx.isfinite(norm).item(): raise FloatingPointError('Non-finite speech update')
    record({'kind':'train','step':step,'family':row['id'],'loss':value.item(),
            'gradient_norm_before_clip':norm.item(),'seconds':time.monotonic()-tick})
    del data,grad,value,norm;mx.metal.clear_cache()
    if step%args.evaluate_every==0 or step==args.steps:
        val=evaluate(step)
        export_adapters(model,args.layers,out/f'step-{step:04d}.safetensors')
        if val<best:
            best=val;export_adapters(model,args.layers,out/'candidate.safetensors')
            (out/'candidate.json').write_text(json.dumps({'step':step,'validation_loss':val,
                'status':'Loss-selected candidate; native streaming and speech quality audit still required'},indent=2)+'\n')
record({'kind':'completion','steps':args.steps,'seconds':time.monotonic()-start,'baseline_loss':baseline,'best_loss':best})
