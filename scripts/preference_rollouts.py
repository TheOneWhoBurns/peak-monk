"""Capture unprompted candidate replies to training-only preference prompts."""
import argparse,json,time,hashlib
from pathlib import Path
import mlx.core as mx
from mlx_lm import load,stream_generate
from mlx_lm.sample_utils import make_sampler
p=argparse.ArgumentParser()
p.add_argument('--model',required=True);p.add_argument('--adapter',required=True)
p.add_argument('--pairs',default='data/preferences-v2.jsonl');p.add_argument('--output',required=True)
p.add_argument('--samples-per-prompt',type=int,default=1)
p.add_argument('--temperature',type=float,default=0.)
p.add_argument('--top-p',type=float,default=1.)
p.add_argument('--seed',type=int,default=41)
p.add_argument('--max-tokens',type=int,default=140)
args=p.parse_args();out=Path(args.output)
config_path=out.with_suffix('.config.json')
if out.exists() or config_path.exists():raise ValueError('Preserve existing rollout output')
if args.samples_per_prompt<1 or args.temperature<0 or not 0<args.top_p<=1 or args.max_tokens<1:
 raise ValueError('Invalid fixed sampling protocol')
out.parent.mkdir(parents=True,exist_ok=True)
mx.set_memory_limit(5_600_000_000);mx.set_cache_limit(32_000_000)
mx.set_wired_limit(mx.metal.device_info()['max_recommended_working_set_size'])
model,tok=load(args.model,adapter_path=args.adapter)
adapter_sha256=hashlib.sha256((Path(args.adapter)/'adapters.safetensors').read_bytes()).hexdigest()
rows=[json.loads(l) for l in Path(args.pairs).read_text().splitlines()]
assert len({r['id'] for r in rows})==len(rows)
assert all(m['role']!='system' for r in rows for m in r['messages'])
config_path.write_text(json.dumps(dict(vars(args),adapter_sha256=adapter_sha256,
 pairs_sha256=hashlib.sha256(Path(args.pairs).read_bytes()).hexdigest(),
 scope='Training-only candidate capture. No sample is automatically labeled rejected; author review required.',
 seed_policy='Fixed prompt-ID hash plus base seed and sample index, independent of prompt iteration order'),indent=2)+'\n')
for row in rows:
 for draw in range(args.samples_per_prompt):
  seed=(args.seed+int(hashlib.sha256(row['id'].encode()).hexdigest()[:8],16)+997*draw)%(2**31)
  mx.random.seed(seed)
  prompt=tok.apply_chat_template(row['messages'],add_generation_prompt=True,tokenize=False,enable_thinking=False)
  text='';tick=time.monotonic();last=None
  for chunk in stream_generate(model,tok,prompt,max_tokens=args.max_tokens,
    sampler=make_sampler(temp=args.temperature,top_p=args.top_p)):
   text+=chunk.text;last=chunk
  if last is None:raise RuntimeError('No generation result')
  ident=row['id'] if args.samples_per_prompt==1 else f"{row['id']}_sample{draw:02d}"
  captured=dict(id=ident,prompt_id=row['id'],sample_index=draw,messages=row['messages'],candidate=text,
   model=args.model,adapter=args.adapter,adapter_sha256=adapter_sha256,temperature=args.temperature,
   top_p=args.top_p,seed=seed,finish_reason=last.finish_reason,seconds=time.monotonic()-tick,
   review='pending; candidate is not automatically a rejected answer')
  with out.open('a') as f:f.write(json.dumps(captured,ensure_ascii=False)+'\n')
  print(ident,text[:180].replace('\n',' '),flush=True)
  mx.clear_cache()
