"""Local alternating-speaker simulation with a persistent associative world.

Teacher prompts and graph traces are provenance only, never SFT messages.
This is a small-Mac adaptation of Zero's method, not a reproduction: the pilot
uses one local teacher for both roles and lexical rather than embedding retrieval.
Every record requires a separate author quality review before training.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
import time
import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class World:
    def __init__(self, graph, rng):
        self.nodes={n['id']:n for n in graph['nodes']}
        self.edges={i:set(n['neighbors']) for i,n in self.nodes.items()}
        for i,near in list(self.edges.items()):
            for j in list(near):self.edges[j].add(i)
        self.heat={i:0. for i in self.nodes}
        self.rng=rng; self.valence=rng.uniform(-.4,.4);self.arousal=rng.uniform(-.3,.3)
    def step(self, text):
        words=set(re.findall('[a-z]+',text.lower()))
        direct=[]
        for i,n in self.nodes.items():
            hit=len(words & set(n['keywords']))
            self.heat[i]=.65*self.heat[i]+min(hit,2)
            if hit:direct.append(i)
        spill={i:0. for i in self.nodes}
        for i,value in self.heat.items():
            for j in self.edges[i]:spill[j]+=.12*value/max(len(self.edges[i]),1)
        for i in self.heat:self.heat[i]+=spill[i]
        if not direct and max(self.heat.values())<.2:
            # No forced prop when the caller hasn't activated a relevant memory.
            return [],dict(valence=self.valence,arousal=self.arousal,activated=[])
        population=list(self.nodes)
        current=self.rng.choices(population,weights=[max(self.heat[i],.001) for i in population])[0]
        picked=[current]
        if self.rng.random()<.5:
            choices=sorted(self.edges[current]);picked+=self.rng.choices(choices,weights=[max(self.heat[i],.01) for i in choices])
        v=sum(self.nodes[i]['valence'] for i in picked)/len(picked)
        a=sum(self.nodes[i]['arousal'] for i in picked)/len(picked)
        self.valence=.8*self.valence+.2*v; self.arousal=.8*self.arousal+.2*a
        return [self.nodes[i] for i in picked],dict(valence=self.valence,arousal=self.arousal,activated=direct,rendered=picked)


def respond(model,tok,messages,seed,temperature,max_tokens):
    mx.random.seed(seed)
    prompt=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    text='';last=None;start=time.monotonic()
    for chunk in stream_generate(model,tok,prompt,max_tokens=max_tokens,sampler=make_sampler(temp=temperature,top_p=.9)):
        text+=chunk.text;last=chunk
    mx.clear_cache()
    return text.strip(),dict(seconds=time.monotonic()-start,tokens=last.generation_tokens,
         finish_reason=last.finish_reason,prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())


def screen(messages,forbidden):
    problems=[]
    for i,m in enumerate(messages):
        s=m['content'];low=s.lower()
        if not s or any(x in s for x in ('<think>','<|im_start|>','```')):problems.append(f'turn {i}: malformed')
        if any(p.lower() in low for p in forbidden):problems.append(f'turn {i}: heldout match')
        if i%2:
            if any(p in low for p in ('as an ai','language model','fictional character','how can i assist','i am an ai')):problems.append(f'turn {i}: character break')
            if '*' in s or '\n- ' in s:problems.append(f'turn {i}: stage directions or list')
            tokens=re.findall(r'\w+',low)
            counts=Counter(tuple(tokens[j:j+4]) for j in range(len(tokens)-3))
            if counts and max(counts.values())>3:problems.append(f'turn {i}: repetition')
    replies=[m['content'].lower() for m in messages[1::2]]
    if len(set(replies))<len(replies):problems.append('repeated entire answer')
    return problems


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',default='models/stronger-clean')
    p.add_argument('--output',default='data/simulation-v2')
    p.add_argument('--start',type=int,default=0)
    p.add_argument('--limit',type=int,default=4)
    p.add_argument('--exchanges',type=int,default=8)
    p.add_argument('--config',default='configs/synthesis.json')
    args=p.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads(Path(args.config).read_text())
    graph=json.loads(Path('character/associations.json').read_text())
    core=Path('character/simulation-core.txt').read_text();style=Path('character/style.txt').read_text()
    forbidden=[u for name in ('dev','final','pressure-v2') for c in json.loads(Path(f'evaluation/{name}.json').read_text()) for u in c['turns']]
    influences={s['id']:s for s in json.loads(Path('sources/manifest.json').read_text())}
    model,tok=load(args.model)
    partners=[('a bicycle courier who dislikes motivational advice','slightly impatient'),('a retired seamstress who enjoys teasing people','playful'),('a night-shift baker with a practical streak','tired'),('a shy apprentice carpenter proud of small progress','hopeful'),('a school librarian who changes their mind while talking','curious'),('a student with very little money and strong opinions','skeptical')]
    for index in range(args.start,args.start+args.limit):
        dest=out/f'{index:04d}.json'
        if dest.exists():continue
        rng=random.Random(18931+index);world=World(graph,rng)
        topic=cfg['topics'][index%len(cfg['topics'])];persona,attitude=partners[index%len(partners)]
        dynamic=cfg['dynamics'][index%len(cfg['dynamics'])]
        partner=f'''You are {persona}, currently {attitude}, chatting with a man named Jun. You want to discuss {topic['subject']}. Your situation: {dynamic}
Speak only your next turn as this person, in one or two ordinary spoken sentences. Have your own concrete details, preferences and agenda. React to what Jun actually says. Do not write Jun's reply, scripts, labels, narration or an essay. You may disagree or change topic. Keep this a conversation, not an interview about Jun. Do not repeat a question already answered.'''
        record=dict(index=index,family=f"simulation_v2_{index:04d}",split='train',sources=topic['sources'],author='Local alternating-speaker simulation',teacher=args.model,
            character_version='human_character_v2',quality_review='pending',single_teacher_limitation=True,
            prompt_assets={f:digest(f) for f in ('character/simulation-core.txt','character/style.txt','character/associations.json',args.config,'sources/manifest.json')},
            template_policy='Native teacher template with enable_thinking=False; generator prompts are excluded from training',
            partner=dict(persona=persona,attitude=attitude,topic=topic['subject'],dynamic=dynamic),messages=[],trace=[])
        for turn in range(args.exchanges):
            hist=[{'role':'assistant' if m['role']=='user' else 'user','content':m['content']} for m in record['messages']]
            # A starting cue is generator-only and excluded from the stored conversation.
            if not hist:hist=[{'role':'user','content':'Begin the conversation with the concrete thing on your mind.'}]
            caller,cm=respond(model,tok,[{'role':'system','content':partner}]+hist,18931+index*100+turn*2,.8,100)
            record['messages'].append({'role':'user','content':caller})
            nodes,state=world.step(caller)
            memory='\n'.join(n['description'] for n in nodes)
            length=rng.choices(['a few words','one brief sentence','two or three sentences','a longer reply if this subject warrants it'],weights=[15,35,40,10])[0]
            feeling='mildly cheerful' if state['valence']>.2 else 'a little subdued' if state['valence']<-.2 else 'matter-of-fact'
            influence='\n'.join(influences[s]['influence'] for s in topic['sources'])
            system=core+'\n'+style+f'\nYou feel {feeling}. Let the reply be {length}.\nBackground dispositions; express them through conduct only when relevant, without reciting them:\n'+influence+'\nAssociations that came to mind; use only if relevant and never recite them:\n'+memory
            jun,jm=respond(model,tok,[{'role':'system','content':system}]+record['messages'],18931+index*100+turn*2+1,.75,220)
            record['messages'].append({'role':'assistant','content':jun})
            record['trace'].append(dict(turn=turn,caller=cm,jun=jm,world=state))
            record['mechanical_problems']=screen(record['messages'],forbidden)
            record['complete']=turn+1==args.exchanges
            dest.write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n')
            print(index,turn,caller[:80],'->',jun[:120],flush=True)
        record['mechanical_screen']='reject' if record['mechanical_problems'] or any(t['jun']['finish_reason']=='length' for t in record['trace']) else 'pass'
        dest.write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n')

if __name__=='__main__':main()
