"""Capture unprompted candidate replies to training-only preference prompts."""
import argparse,json,time,hashlib
from pathlib import Path
import mlx.core as mx
from mlx_lm import load,stream_generate
from mlx_lm.sample_utils import make_sampler
p=argparse.ArgumentParser()
p.add_argument('--model',required=True);p.add_argument('--adapter',required=True)
p.add_argument('--pairs',default='data/preferences-v2.jsonl');p.add_argument('--output',required=True)
args=p.parse_args();out=Path(args.output)
if out.exists():raise ValueError('Preserve existing rollout output')
model,tok=load(args.model,adapter_path=args.adapter)
adapter_sha256=hashlib.sha256((Path(args.adapter)/'adapters.safetensors').read_bytes()).hexdigest()
rows=[json.loads(l) for l in Path(args.pairs).read_text().splitlines()]
for row in rows:
 mx.random.seed(41)
 prompt=tok.apply_chat_template(row['messages'],add_generation_prompt=True,tokenize=False,enable_thinking=False)
 text='';tick=time.monotonic()
 for chunk in stream_generate(model,tok,prompt,max_tokens=140,sampler=make_sampler(temp=0.0)):text+=chunk.text
 captured=dict(id=row['id'],messages=row['messages'],candidate=text,model=args.model,adapter=args.adapter,adapter_sha256=adapter_sha256,temperature=0.0,seed=41,finish_reason=chunk.finish_reason,seconds=time.monotonic()-tick,review='pending; candidate is not automatically a rejected answer')
 with out.open('a') as f:f.write(json.dumps(captured,ensure_ascii=False)+'\n')
 print(row['id'],text[:180].replace('\n',' '),flush=True)
 mx.clear_cache()
