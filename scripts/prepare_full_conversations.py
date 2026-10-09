"""Weighted complete conversations for all-assistant-span supervision."""
import argparse
from prepare import parse
import hashlib
import json
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument('--source', default='data/prepared-human-rich-v2', type=Path)
parser.add_argument('--output', default='data/prepared-full-conversation-v2', type=Path)
parser.add_argument('--core-extra-source', action='append', default=[], type=Path)
parser.add_argument('--retention-weight', type=int, default=1)
args = parser.parse_args()
source = args.source
out = args.output
out.mkdir(exist_ok=True)
rows = [json.loads(l) for l in (source / 'conversations.jsonl').read_text().splitlines()]
rows = [row for row in rows if row['split'] == 'train']
core = set(json.loads(Path('data/prepared-curriculum-v2/manifest.json').read_text())['core_families'])
for path in args.core_extra_source: core.update(row['family'] for row in parse(path))
if args.retention_weight < 1: raise ValueError('Positive retention weight required')
for pair in map(json.loads, Path('data/preferences-v2.jsonl').read_text().splitlines()):
    rows.append(dict(family='positive_' + pair['id'], sources=[], author=pair['author'], source_material=pair['source_material'], messages=pair['messages'] + [{'role':'assistant', 'content':pair['chosen']}], character_version='human_character_v2', source_kind='Positive-only authored preference; rejected response excluded'))
weighted = [dict(row, curriculum_repetition=n) for row in rows for n in range(4 if row['family'] in core else args.retention_weight if row['family'].startswith('retention_') else 1)]
(out / 'train.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in weighted))
for name in ['valid.jsonl', 'test.jsonl']: (out / name).write_bytes((source / name).read_bytes())
metadata = dict(unique_conversations=len(rows), weighted_conversations=len(weighted), assistant_turns_unique=sum(sum(m['role']=='assistant' for m in r['messages']) for r in rows), core_weight=4, retention_variant_weight=args.retention_weight, core_families=sorted(core), core_extra_sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in args.core_extra_source}, objective='Every assistant span through EOS; callers and message formatting excluded; target-preserving windows with 128 context tokens overlap.', validation='Unchanged last-assistant-reply assistant_eos_v2 loss.', sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source/'conversations.jsonl', Path('data/preferences-v2.jsonl'),Path('data/prepared-curriculum-v2/manifest.json')]})
(out / 'manifest.json').write_text(json.dumps(metadata, indent=2)+'\n')
print(json.dumps(metadata, indent=2))
