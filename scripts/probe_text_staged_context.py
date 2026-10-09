"""Disposable real-base memory probe with staged tied-head gradients.

Repeated training tokens are only a deterministic stress input, not new data.
No adapter or optimizer is exported. Source weights remain untouched.
"""
import argparse
from functools import partial
import hashlib
import json
from pathlib import Path
import time
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.tuner.utils import load_adapters
from qwen_rng_checkpoint import qwen_rng_checkpoint
from conversation_objective import encode_conversation
from staged_character_grad import staged_character_grad

p=argparse.ArgumentParser()
p.add_argument('--adapter',default='adapters/run17-step0600')
p.add_argument('--tokens',type=int,default=2048)
p.add_argument('--steps',type=int,default=2)
p.add_argument('--chunk-size',type=int,default=64)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
if a.output.exists():raise ValueError('Use a fresh output')
if a.tokens<16 or a.steps<1:raise ValueError('Positive stress-probe dimensions required')
mx.set_memory_limit(5_600_000_000);mx.set_cache_limit(32_000_000)
recommended=mx.metal.device_info()['max_recommended_working_set_size'];mx.set_wired_limit(recommended)
mx.random.seed(41)
model,tok=load('models/base4b');model.freeze();load_adapters(model,a.adapter)
params=tree_flatten(model.trainable_parameters())
assert all(k.endswith(('.lora_a','.lora_b')) for k,_ in params)
trainable=sum(v.size for _,v in params);del params
qwen_rng_checkpoint(model.layers[0]);model.train();mx.eval(model.parameters())
rows=[json.loads(line) for line in Path('data/prepared-full-long-v4/train.jsonl').read_text().splitlines()]
row=next(r for r in rows if r.get('source_file')=='data/grounded_long_v6.txt')
tokens,mask=encode_conversation(row,tok)
repetitions=(a.tokens+1+len(tokens)-1)//len(tokens)
batch=mx.array([(tokens*repetitions)[:a.tokens+1]])
targets=mx.array([(mask*repetitions)[:a.tokens+1]])
value_grad=partial(staged_character_grad,chunk_size=a.chunk_size)
optimizer=optim.Adam(learning_rate=1e-5)
records=[];start=time.monotonic()
for step in range(1,a.steps+1):
 tick=time.monotonic();(value,count),gradient=value_grad(model,batch,targets)
 norm=mx.sqrt(sum(mx.sum(g.astype(mx.float32)**2) for _,g in tree_flatten(gradient)))
 mx.eval(value,norm)
 assert mx.isfinite(value).item() and mx.isfinite(norm).item()
 optimizer.update(model,gradient);mx.eval(model.parameters(),optimizer.state)
 assert all(bool(mx.all(mx.isfinite(v)).item()) for _,v in tree_flatten(model.trainable_parameters()))
 record=dict(step=step,loss=value.item(),gradient_norm=norm.item(),target_tokens=int(count.item()),seconds=time.monotonic()-tick,peak_bytes=mx.get_peak_memory())
 records.append(record);print(json.dumps(record),flush=True)
 del gradient,value,norm;mx.clear_cache()
report=dict(status='pass',tokens=a.tokens,steps=a.steps,chunk_size=a.chunk_size,trainable_parameters=trainable,
 parent=a.adapter,parent_sha256=hashlib.sha256((Path(a.adapter)/'adapters.safetensors').read_bytes()).hexdigest(),
 source_family=row['family'],input_scope='Repeated training dialogue only as a disposable memory stress input',
 peak_memory_bytes=mx.get_peak_memory(),recommended_working_set=recommended,seconds=time.monotonic()-start,
 records=records,scope='Full-context attention with sequential vocabulary-head gradients and RNG-correct layer checkpointing, two finite optimizer updates. No quality result, no exported checkpoint.')
a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
