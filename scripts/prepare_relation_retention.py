"""Counterfactual variants of three authored ordinary-conversation templates.

These are 3 scenario families, not 24 independent scenario designs. All variants
stay in training. Names/objects are selected here, not drawn from evaluation.
"""
import json
from pathlib import Path
variants=[('Daria','Malik','Fenn','Tuesday','plums'),('Ines','Tomas','Bramble','Monday','olives'),('Ravi','Selma','Button','Saturday','apples'),('Nell','Hugo','Cricket','Sunday','walnuts'),('Esme','Luca','Clover','Wednesday','figs'),('Petra','Anton','Pippin','Friday','cherries'),('Saira','Felix','Tansy','Tuesday','lemons'),('Mina','Otto','Biscuit','Saturday','almonds')]
rows=[]
for index,(caller,other,pet,day,food) in enumerate(variants):
 dialogues={
 'travel_and_care':[
 (f'My sister {other} is going away on {day}. I am {caller}, by the way.',f'Hello, {caller}. Are you going with your sister?'),
 (f'No, I am staying to look after her rabbit, {pet}.',f'Then you will have some company while {other} is away. Have you looked after {pet} before?'),
 ('Just once. Which one of us is traveling?',f'Your sister {other} is traveling on {day}. You are staying to look after her rabbit.'),
 (f'And who will feed {pet}?',f'You, {caller}. You said you were looking after {other}\'s rabbit.')],
 'shopping_ownership':[
 (f'I am {caller}. My neighbor {other} asked me to bring {food} for supper.',f'Hello, {caller}. Does {other} want them for a particular dish?'),
 ('For a surprise. I promised to bring them, but I am still at home.',f'Then {other} may have to wait a little longer for the surprise to acquire its ingredients.'),
 ('Which of us promised to do the shopping?',f'You did. {other} is your neighbor, and you promised to bring the {food}.'),
 ('You are not coming to supper, though.',f'No. I am here on the mountain. You are the one with a shopping errand before supper.')],
 'pet_and_friend':[
 (f'My dog is called {pet}. I am {caller}.',f'Hello, {caller}. How long has {pet} been with you?'),
 (f'Six years. My friend {other} chose the name.',f'Has {other} ever explained why that name suited your dog?'),
 ('Apparently it was the first thing that came to mind. Do you remember who the dog belongs to?',f'To you, {caller}. {other} is the friend who chose the name.'),
 (f'And if I say {pet}, who do I mean?',f'Your dog. I have not mistaken {pet} for the person I am talking to.')]
 }
 for template,dialogue in dialogues.items():
  rows.append(dict(family=f'retention_{template}_{index:02d}',template_family=template,variation_index=index,sources=['tnh','lawrence'],author='Codex authored template with deterministic counterfactual name and object substitution',messages=[dict(role=role,content=text) for pair in dialogue for role,text in zip(('user','assistant'),pair)]))
out=Path('data/relation_retention_v2.txt')
lines=['# Original authored templates with deterministic slot variations; all training only.']
for row in rows:
 lines+=['',f"## {row['family']} | tnh,lawrence"]
 lines += [('U: ' if m['role']=='user' else 'A: ')+m['content'] for m in row['messages']]
out.write_text('\n'.join(lines)+'\n')
Path('data/relation_retention_v2-manifest.json').write_text(json.dumps(dict(authored_scenario_families=3,variants_per_family=8,records=len(rows),assistant_replies=96,source='scripts/prepare_relation_retention.py',purpose='Retain caller/other/self distinctions during character SFT; not a benchmark',split='train only',variants=variants),indent=2)+'\n')
print('3 authored scenario families, 24 variants, 96 assistant replies')
