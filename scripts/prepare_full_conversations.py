"""Weighted complete conversations for all-assistant-span supervision."""
import hashlib
import json
from pathlib import Path
source = Path('data/prepared-human-rich-v2')
out = Path('data/prepared-full-conversation-v2')
out.mkdir(exist_ok=True)
rows = [json.loads(l) for l in (source / 'conversations.jsonl').read_text().splitlines()]
rows = [row for row in rows if row['split'] == 'train']
core = set(json.loads(Path('data/prepared-curriculum-v2/manifest.json').read_text())['core_families'])
for pair in map(json.loads, Path('data/preferences-v2.jsonl').read_text().splitlines()):
    rows.append(dict(family='positive_' + pair['id'], sources=[], author=pair['author'], source_material=pair['source_material'], messages=pair['messages'] + [{'role':'assistant', 'content':pair['chosen']}], character_version='human_character_v2', source_kind='Positive-only authored preference; rejected response excluded'))
weighted = [dict(row, curriculum_repetition=n) for row in rows for n in range(4 if row['family'] in core else 1)]
(out / 'train.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in weighted))
for name in ['valid.jsonl', 'test.jsonl']: (out / name).write_bytes((source / name).read_bytes())
metadata = dict(unique_conversations=len(rows), weighted_conversations=len(weighted), assistant_turns_unique=sum(sum(m['role']=='assistant' for m in r['messages']) for r in rows), core_weight=4, objective='Every assistant span through EOS; callers and message formatting excluded; target-preserving windows with 128 context tokens overlap.', validation='Unchanged last-assistant-reply assistant_eos_v2 loss.', sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source/'conversations.jsonl', Path('data/preferences-v2.jsonl'),Path('data/prepared-curriculum-v2/manifest.json')]})
(out / 'manifest.json').write_text(json.dumps(metadata, indent=2)+'\n')
print(json.dumps(metadata, indent=2))
