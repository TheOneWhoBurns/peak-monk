"""Generate traceable multi-turn character distillation data on the local Mac.

Writes raw teacher outputs and mechanically screened records. Screening is not a
quality rating: explicitly review the outputs before adding them to training.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time
import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

ROOT = Path(__file__).resolve().parents[1]


def validate(messages, forbidden):
    if not isinstance(messages, list) or not 6 <= len(messages) <= 10 or len(messages) % 2:
        raise ValueError('Expected three to five complete exchanges')
    for i, m in enumerate(messages):
        if m.get('role') != ('user' if i % 2 == 0 else 'assistant'):
            raise ValueError('Invalid role alternation')
        s = m.get('content')
        if not isinstance(s, str) or not 2 <= len(s) <= 1600:
            raise ValueError('Invalid utterance length')
        if any(p.casefold() in s.casefold() for p in forbidden):
            raise ValueError('Exact held-out prompt contamination')
        if any(x in s for x in ('<|im_start|>', '<think>', '```')):
            raise ValueError('Markup contamination')
    replies = [m['content'].strip().lower() for m in messages[1::2]]
    if len(set(replies)) != len(replies):
        raise ValueError('Exact repeated reply')
    return [{'role':m['role'], 'content':m['content'].strip()} for m in messages]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/synthesis.json')
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--limit', type=int, default=24)
    args = p.parse_args()
    cfg = json.loads((ROOT/args.config).read_text())
    model, tok = load(str(ROOT/cfg['teacher']))
    core = (ROOT/'character/core.md').read_text()
    sources = {s['id']:s for s in json.loads((ROOT/'sources/manifest.json').read_text())}
    forbidden = [turn for name in ('dev','final')
                 for c in json.loads((ROOT/f'evaluation/{name}.json').read_text()) for turn in c['turns']]
    out = ROOT/'data/synthesis'; out.mkdir(parents=True, exist_ok=True)
    teacher = json.loads((ROOT/'stronger_provenance.json').read_text())
    for i in range(args.start, min(args.start + args.limit, cfg['conversations'])):
        dest = out/f'{i:04d}.json'
        if dest.exists():
            continue
        rng = random.Random(cfg['seed'] + i)
        topic = cfg['topics'][i % len(cfg['topics'])]
        mood = cfg['moods'][(i // len(cfg['topics'])) % len(cfg['moods'])]
        dynamic = rng.choice(cfg['dynamics'])
        influences = '\n'.join(sources[s]['influence'] for s in topic['sources'])
        prompt = f'''Write an ORIGINAL natural conversation with four user/assistant exchanges.
The assistant is Jun, described below. The caller is an ordinary person with their own preferences.
Return only a JSON object with a "messages" array of objects with "role" and "content".
Use roles "user" and "assistant", alternating, beginning with user and ending with assistant.
Jun must address each utterance specifically, retain facts across turns, and sound like a person
talking naturally. Usually one to three short sentences per Jun reply. No stage directions.
No moral or question required at the end of a reply. Give the caller distinct concrete details.
Do not write an essay, a sermon, a therapy checklist, or a generic assistant exchange.
Do not quote or impersonate any existing monk or fictional character. Jun is original.

Character specification:
{core}

Topic: {topic['subject']}
Jun's current mood: {mood}
Conversation dynamic: {dynamic}
Behavioral influences, to use subtly rather than mention by name:
{influences}
'''
        mx.random.seed(cfg['seed'] + i)
        encoded = tok.apply_chat_template([{'role':'user','content':prompt}],
                    add_generation_prompt=True, tokenize=False, enable_thinking=False)
        start = time.monotonic()
        raw = generate(model, tok, prompt=encoded, max_tokens=cfg['max_tokens'],
                       sampler=make_sampler(temp=cfg['temperature'], top_p=cfg['top_p']), verbose=False)
        record = {'index':i, 'family':f"local_{topic['id']}_{i:04d}", 'sources':topic['sources'],
                  'author':'Local Qwen3-4B-Instruct-2507 synthetic generation', 'teacher':teacher,
                  'seed':cfg['seed']+i, 'temperature':cfg['temperature'], 'top_p':cfg['top_p'],
                  'mood':mood, 'dynamic':dynamic, 'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
                  'raw':raw, 'seconds':time.monotonic()-start, 'quality_review':'pending', 'split':'train'}
        try:
            obj = json.loads(raw[raw.index('{'):raw.rindex('}')+1])
            record['messages'] = validate(obj['messages'], forbidden)
            record['mechanical_screen'] = 'pass'
        except (ValueError, KeyError, TypeError) as e:
            record['mechanical_screen'] = 'reject'
            record['reason'] = str(e)
        dest.write_text(json.dumps(record, indent=2, ensure_ascii=False)+'\n')
        print(i, record['mechanical_screen'], round(record['seconds'], 1), flush=True)
        mx.clear_cache()


if __name__ == '__main__':
    main()
