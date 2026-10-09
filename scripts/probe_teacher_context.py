"""Preselected first-turn teacher context comparison; no automatic data admission."""
import json
from pathlib import Path
import random
import sys
import mlx.core as mx
from mlx_lm import load
from simulate_dialogues import World, respond, digest
from tokenizer_preflight import check_tokenizer

out=Path('results/teacher-context-ablation.json')
if out.exists():raise ValueError('Preserve existing trials')
indices=[0,3,20,21]
openings=json.loads(Path('data/teacher-openings-v5.json').read_text())
cfg=json.loads(Path('configs/synthesis.json').read_text())
graph=json.loads(Path('character/associations.json').read_text())
core=Path('character/simulation-core.txt').read_text()
style=Path('character/style.txt').read_text()
influences={x['id']:x for x in json.loads(Path('sources/manifest.json').read_text())}
compact="""Speak as Jun, a human man living alone in a small stone hermitage just below a mountain summit. You used to repair roofs in the valley. Your teacher was Mara, a woman who died years ago. You have ordinary chores, memories, preferences and an imperfect temper. You are not enlightened. You chose this life and still like people.
SCENE: Jun is at his mountain hermitage. The caller is somewhere else. This is a voice-only connection. Things the caller mentions belong to the caller's surroundings. You only know those things through their words. Speak to the caller from your own place and experience.
Give only Jun's next spoken reply, one to three plain sentences. Respond to the particular remark. Be curious or disagree when appropriate. No stage directions, sermons, metaphors about souls, or offers to act as an assistant. You need not solve the caller's problem."""
examples="""
Examples of conversational perspective and tone, not past events with this caller:
Caller: I knocked a glass off my kitchen counter.
Jun: Broken, or did it survive?
Caller: Broken. Can you see where the shards went?
Jun: From up here? You'll have to look under the cupboard yourself.
Caller: Finish my work report for me.
Jun: You spent the week there. What would I put in it?
Caller: You must have learned everything alone.
Jun: Hardly. Mara had to show me the same things over and over.
End of examples. Now answer the actual caller below."""
report=dict(status='running',scope='Generator setup ablation, not learned-character evaluation',indices=indices,
    temperature=.75,top_p=.9,thinking=False,max_tokens=220,seed_policy='Same18931+index*100+1 per variant',
    variants=['current_full','compact_no_graph','compact_graph','compact_graph_examples'],
    tokenizer=check_tokenizer('models/qwen35'),results=[])
out.write_text(json.dumps(report,indent=2)+'\n')
mx.set_memory_limit(5_500_000_000);mx.set_cache_limit(64_000_000)
model,tok=load('models/qwen35')
for index in indices:
    caller=openings[str(index)];rng=random.Random(18931+index);world=World(graph,rng)
    nodes,state=world.step(caller);memory='\n'.join(n['description'] for n in nodes)
    length=rng.choices(['a few words','one brief sentence','two or three sentences','a longer reply if this subject warrants it'],weights=[15,35,40,10])[0]
    feeling='mildly cheerful' if state['valence']>.2 else 'a little subdued' if state['valence']<-.2 else 'matter-of-fact'
    topic=cfg['topics'][index%len(cfg['topics'])]
    influence='\n'.join(influences[s]['influence'] for s in topic['sources'])
    original=core+'\n'+style+"\nThis is an audio-only conversation with someone elsewhere. You cannot touch their things or hand them objects. Your memories are yours; the caller's experiences are theirs."+f'\nYou feel {feeling}. Let the reply be {length}.\nBackground dispositions; express them through conduct only when relevant, without reciting them:\n'+influence+'\nAssociations that came to mind; use only if relevant and never recite them:\n'+memory
    local_memory="\nJun's own recollections, not observations of the caller. Ignore them if irrelevant to the conversation:\n"+memory
    variants=dict(current_full=original,compact_no_graph=compact,compact_graph=compact+local_memory,compact_graph_examples=compact+local_memory+examples)
    for name,system in variants.items():
        reply,metrics=respond(model,tok,[dict(role='system',content=system),dict(role='user',content=caller)],18931+index*100+1,.75,220)
        report['results'].append(dict(index=index,variant=name,system=system,caller=caller,reply=reply,metrics=metrics,review='pending'))
        out.write_text(json.dumps(report,indent=2)+'\n');print(index,name,reply,flush=True)
report['status']='complete, author review pending';out.write_text(json.dumps(report,indent=2)+'\n')
