"""Local writer-mode synthesis pilot; every line requires author review."""
import argparse
import json
from pathlib import Path
import re
import mlx.core as mx
from mlx_lm import load
from simulate_dialogues import respond,screen,digest
from tokenizer_preflight import check_tokenizer

p=argparse.ArgumentParser()
p.add_argument('--output',required=True)
p.add_argument('--model',default='models/qwen35')
args=p.parse_args();out=Path(args.output)
if out.exists():raise ValueError('Use a fresh output directory')
out.mkdir(parents=True)
openings=json.loads(Path('data/teacher-openings-v5.json').read_text())
core=Path('character/simulation-core.txt').read_text()
style=Path('character/style.txt').read_text()
manifest={s['id']:s for s in json.loads(Path('sources/manifest.json').read_text())}
cfg=json.loads(Path('configs/synthesis.json').read_text())
forbidden=[u for name in ('dev','final','pressure-v2') for c in json.loads(Path(f'evaluation/{name}.json').read_text()) for u in c['turns']]
preflight=check_tokenizer(args.model)
mx.set_memory_limit(5_500_000_000);mx.set_cache_limit(64_000_000)
model,tok=load(args.model)
for index in [0,21]:
    topic=cfg['topics'][index];sources=topic['sources']
    system="""You write plausible, concrete spoken dialogue between two people. Keep the speakers' locations and experiences separate. Follow the provided character facts. Do not add a narrator, stage directions, poetic symbolism, commentary about writing, or an assistant offering services. The dialogue should develop through specific details and responses, not speeches about lessons or wisdom."""
    prompt='Write one original conversation of exactly20lines, alternating Caller: and Jun:, starting with Caller:. Each line is one spoken turn, usually one or two sentences.\nCharacter facts (second-person statements here describe Jun):\n'+core+'\nSpeech style:\n'+style+"\nScene: Jun is alone at his mountain hermitage. The caller is elsewhere, speaking through an audio connection. Jun cannot see the caller, their room, their belongings, or any activity they describe. The caller cannot hand Jun objects or ask him to touch their things.\nFirst caller line, verbatim: "+openings[str(index)]+"\nLet the caller bring up another concrete detail and let Jun respond from his own experience. Keep continuity across all10exchanges. Jun can have preferences and disagree. He does not need to give advice or mention his teacher. End naturally, without a moral.\nBroad literary/monastic influence, not a quotation or a person to impersonate:\n"+'\n'.join(manifest[s]['influence'] for s in sources)+'\nOutput only the20dialogue lines.'
    raw,metrics=respond(model,tok,[dict(role='system',content=system),dict(role='user',content=prompt)],29101+index,.7,1600)
    messages=[];problems=[]
    for line in raw.splitlines():
        if not line.strip():continue
        match=re.fullmatch(r'(Caller|Jun):\s*(.+)',line.strip())
        if not match:problems.append('Unparsed line: '+line);continue
        messages.append(dict(role='user' if match[1]=='Caller' else 'assistant',content=match[2]))
    if len(messages)!=20:problems.append('Expected20speakerturns')
    if any(m['role']!=('user' if i%2==0 else 'assistant') for i,m in enumerate(messages)):problems.append('Speaker alternation failed')
    if not messages or messages[0]['content']!=openings[str(index)]:problems.append('Opening changed')
    problems+=screen(messages,forbidden)
    record=dict(index=index,family=f'writer_pilot_{index}',split='train',sources=sources,
        author='Local writer-mode pilot, unreviewed',teacher=args.model,
        generation_method='One model writes both sides jointly; not Zero alternating-speaker simulation',
        character_version='human_character_v2',quality_review='pending',mechanical_problems=problems,
        messages=messages,raw_generation=raw,metrics=metrics,tokenizer=preflight,
        generator_prompt=dict(system=system,user=prompt),prompt_assets={n:digest(n) for n in ['character/simulation-core.txt','character/style.txt','sources/manifest.json']})
    (out/f'{index:04d}.json').write_text(json.dumps(record,indent=2)+'\n');print(index,len(messages),problems,flush=True)
