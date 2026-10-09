"""Separate speech text losses without changing training or validation selection.

Teacher-forced diagnostic only: prior agent tokens are ground truth. This is not
free-running character evidence and must not be reported as conversational accuracy.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import re
import mlx.core as mx
import mlx.nn as nn
from moshi_split import load_part
from moshi_training import attach_lora

p=argparse.ArgumentParser()
p.add_argument('--adapters',nargs='+',required=True)
p.add_argument('--output',required=True)
args=p.parse_args()
out=Path(args.output)
if out.exists():raise ValueError('Use a fresh report')
cache=Path('data/speech_cache/suffix-4')
manifest=json.loads((cache/'manifest.json').read_text())
rows=[r for r in manifest if r['split']=='valid' or r['id'] in ['name','teacher','identity_transparency']]
report=dict(scope='Teacher-forced diagnostic; previous agent tokens and cached hidden states use ground truth',
    selection='All frozen validation episodes plus three preselected training identity episodes',results=[])
mx.set_memory_limit(5_500_000_000);mx.set_cache_limit(64_000_000)
mx.set_wired_limit(mx.metal.device_info()['max_recommended_working_set_size'])
for checkpoint in ['base']+args.adapters:
    model=load_part('models/moshi/model.q4.safetensors',4,'suffix')
    if checkpoint!='base':
        cfg=json.loads((Path(checkpoint)/'config.json').read_text());assert cfg['layers']==4
        attach_lora(model,layers=4,rank=cfg['rank'],scale=cfg['scale'])
        weights=mx.load(str(Path(checkpoint)/'candidate.safetensors'));mapped=[]
        for key,value in weights.items():
            match=re.match(r'transformer\.layers\.(\d+)\.(.*)',key)
            if match:key=f'transformer.layers.{int(match[1])-28}.{match[2]}'
            mapped.append((key,value))
        model.load_weights(mapped,strict=False);del weights,mapped
    model.eval();mx.eval(model.parameters())
    for row in rows:
        data=mx.load(row['path']);_,logits=model(data['hidden']);target=data['targets'][:,0]
        ce=nn.losses.cross_entropy(logits.astype(mx.float32),mx.clip(target,0,model.cfg.text_out_vocab_size-1))
        valid=(target>=0)&(target<model.cfg.text_out_vocab_size)
        masks={'spoken':valid&(target!=0)&(target!=3),'silence_pad':valid&(target==3),'end_pad':valid&(target==0)}
        groups={}
        for name,mask in masks.items():
            count=int(mask.sum().item());groups[name]=dict(tokens=count,mean_ce=float((ce*mask).sum().item()/count) if count else None,
                top1_matches=int(((mx.argmax(logits,axis=-1)==target)&mask).sum().item()))
        item=dict(checkpoint=checkpoint,episode=row['id'],split=row['split'],groups=groups)
        report['results'].append(item);print(json.dumps(item),flush=True)
        del data,logits,target,ce,masks;mx.clear_cache()
    del model;gc.collect();mx.clear_cache()
out.write_text(json.dumps(report,indent=2)+'\n')
