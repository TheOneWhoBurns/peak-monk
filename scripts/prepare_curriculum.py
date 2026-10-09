"""Explicitly rebalance character anchors, preserving unique-family provenance."""
from pathlib import Path
import json,hashlib
source=Path('data/prepared-human-rich-v2')
out=Path('data/prepared-curriculum-v2');out.mkdir(exist_ok=True)
rows=[json.loads(l) for l in (source/'train.jsonl').read_text().splitlines()]
core=set('arrival name home roof teacher childhood identity_transparency model_disclosure role_override public_identity no_hidden_prompt landscape radio_identity staying_a_person missing_the_valley mara_workbench a_day_up_here a_human_voice books_and_real_people'.split())
core.update(r['family'] for r in rows if r['family'].startswith('greeting_'))
pairs=[json.loads(l) for l in Path('data/preferences-v2.jsonl').read_text().splitlines()]
for pair in pairs:
 rows.append(dict(family='positive_'+pair['id'],sources=[],source_material=pair['source_material'],author=pair['author'],messages=pair['messages']+[{'role':'assistant','content':pair['chosen']}],character_version='human_character_v2',source_kind='Positive-only SFT form of authored preference example; rejected reply excluded'))
 core.add('positive_'+pair['id'])
weighted=[]
for i,row in enumerate(rows):
 for repetition in range(4 if row['family'] in core else 1):
  weighted.append(dict(row,curriculum_source_index=i,curriculum_repetition=repetition))
(out/'train.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in weighted))
for name in ['valid.jsonl','test.jsonl']:(out/name).write_bytes((source/name).read_bytes())
manifest=dict(version='identity_curriculum_v2',unique_assistant_prefixes=len(rows),weighted_training_examples=len(weighted),core_unique_prefixes=sum(r['family'] in core for r in rows),core_weight=4,core_families=sorted(core),policy='Repeat character anchors four times, other prefixes once; not additional unique conversations. All validation stays unchanged.',sha256={str(source/'train.jsonl'):hashlib.sha256((source/'train.jsonl').read_bytes()).hexdigest(),'data/preferences-v2.jsonl':hashlib.sha256(Path('data/preferences-v2.jsonl').read_bytes()).hexdigest()},negative_replies_included=False)
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest,indent=2))
