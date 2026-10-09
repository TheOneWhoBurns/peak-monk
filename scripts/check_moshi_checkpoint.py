"""Check exported adapter equivalence on an actual pretrained speech episode.

This is a numerical implementation check, not a conversational quality test.
The full model and cached suffix are loaded sequentially to fit the 8 GB Mac.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import re
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from moshi_mlx.models import Lm, config_v0_1
from moshi_training import attach_lora, loss
from moshi_split import delay_streams, load_part, cached_loss

p=argparse.ArgumentParser()
p.add_argument('--adapter',required=True)
p.add_argument('--config',required=True)
p.add_argument('--episode',default='name')
p.add_argument('--output',required=True)
args=p.parse_args()
cfg=json.loads(Path(args.config).read_text())
adapter_sha=hashlib.sha256(Path(args.adapter).read_bytes()).hexdigest()
mx.metal.set_memory_limit(5_700_000_000)
mx.metal.set_cache_limit(64_000_000)
mx.metal.set_wired_limit(mx.metal.device_info()['max_recommended_working_set_size'])
raw=mx.array(np.load(Path('data/speech_codes')/(args.episode+'.npz'))['codes'][None,:,:])
streams=delay_streams(raw,config_v0_1())
model=Lm(config_v0_1());model.set_dtype(mx.bfloat16)
nn.quantize(model,bits=4,group_size=32)
model.load_weights('models/moshi/model.q4.safetensors',strict=True)
attach_lora(model,layers=cfg['layers'],rank=cfg['rank'],scale=cfg['scale'])
model.load_weights(args.adapter,strict=False);model.eval()
mx.eval(model.parameters())
full=loss(model,streams);mx.eval(full);full_value=full.item()
full_peak=mx.metal.get_peak_memory()
print('Full exported checkpoint loss:',full_value,flush=True)
del model,full,streams,raw
gc.collect();mx.metal.clear_cache()
model=load_part('models/moshi/model.q4.safetensors',cfg['layers'],'suffix')
attach_lora(model,layers=cfg['layers'],rank=cfg['rank'],scale=cfg['scale'])
weights=mx.load(args.adapter);mapped=[]
for key,value in weights.items():
    match=re.match(r'transformer\.layers\.(\d+)\.(.*)',key)
    if match:key=f'transformer.layers.{int(match[1])-(32-cfg["layers"])}.{match[2]}'
    mapped.append((key,value))
model.load_weights(mapped,strict=False);model.eval()
data=mx.load(str(Path(cfg['cache'])/(args.episode+'.safetensors')))
cached=cached_loss(model,data['hidden'],data['targets']);mx.eval(cached)
cached_value=cached.item()
# Predeclared tolerance accommodates BF16 evaluation/fusion rounding, while
# still catching a materially different objective or incorrectly mapped adapter.
tolerance=.01
report=dict(scope='Actual pretrained quantized model and trained exported adapter; full versus cached episode forward loss',
    episode=args.episode,adapter=args.adapter,adapter_sha256=adapter_sha,
    full_loss=full_value,cached_loss=cached_value,absolute_difference=abs(full_value-cached_value),
    tolerance=tolerance,passed=abs(full_value-cached_value)<=tolerance,
    full_peak_memory_bytes=full_peak,peak_memory_bytes=mx.metal.get_peak_memory())
Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)
if not report['passed']:raise SystemExit('Exported checkpoint parity failed')
