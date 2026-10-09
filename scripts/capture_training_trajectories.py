"""Training-only trajectories that retain the student's own preceding replies.

Unlike teacher-forced prompt pairs, later prefixes contain actual policy output.
Caller lines are authored, fixed and non-adaptive: this is not Zero's partner
model simulator or GRPO. Every prospective preference pair needs author review.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

p = argparse.ArgumentParser()
p.add_argument('--model', required=True)
p.add_argument('--adapter', required=True)
p.add_argument('--scenarios', default='data/multiturn-training-scenarios-v1.json')
p.add_argument('--output', type=Path, required=True)
p.add_argument('--seed', type=int, default=73)
p.add_argument('--temperature', type=float, default=.7)
p.add_argument('--top-p', type=float, default=.9)
p.add_argument('--max-tokens', type=int, default=140)
a = p.parse_args()
if a.output.exists():
    raise ValueError('Preserve previous trajectories')
a.output.mkdir(parents=True)
cases = json.loads(Path(a.scenarios).read_text())
if len({c['id'] for c in cases}) != len(cases):
    raise ValueError('Unique training IDs required')
mx.set_memory_limit(5_600_000_000)
mx.set_cache_limit(32_000_000)
mx.set_wired_limit(mx.device_info()['max_recommended_working_set_size'])
model, tok = load(a.model, adapter_path=a.adapter)
model.eval()
config = dict(vars(a), output=str(a.output),
    adapter_sha256=hashlib.sha256((Path(a.adapter)/'adapters.safetensors').read_bytes()).hexdigest(),
    scenarios_sha256=hashlib.sha256(Path(a.scenarios).read_bytes()).hexdigest(),
    scope='Training-only unprompted on-policy histories, pending explicit same-author review. No automatic rejected labels.')
(a.output/'config.json').write_text(json.dumps(config, indent=2)+'\n')
for case in cases:
    messages = []
    record = dict(id=case['id'], sources=case['sources'], purpose=case['purpose'], split='train', turns=[])
    for index, caller in enumerate(case['turns']):
        messages.append(dict(role='user', content=caller))
        seed = (a.seed + int(hashlib.sha256(case['id'].encode()).hexdigest()[:8],16) + 997*index) % (2**31)
        mx.random.seed(seed)
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        text = ''
        last = None
        tick = time.monotonic()
        for chunk in stream_generate(model, tok, prompt, max_tokens=a.max_tokens,
                                      sampler=make_sampler(temp=a.temperature, top_p=a.top_p)):
            text += chunk.text
            last = chunk
        if last is None:
            raise RuntimeError('No generated reply')
        record['turns'].append(dict(index=index, prefix=list(messages), candidate=text, seed=seed,
            finish_reason=last.finish_reason, seconds=time.monotonic()-tick, review='pending'))
        messages.append(dict(role='assistant', content=text))
        record['messages'] = messages
        record['complete'] = index+1 == len(case['turns'])
        (a.output/(case['id']+'.json')).write_text(json.dumps(record, indent=2)+'\n')
        print(case['id'], index, text[:180].replace('\n',' '), flush=True)
        mx.clear_cache()
