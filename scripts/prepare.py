"""Deterministic family-first corpus preparation, never random turn splitting."""
import hashlib
import json
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def parse(path):
    rows = []
    record = None
    for raw in path.read_text().splitlines():
        if raw.startswith('## '):
            if record:
                rows.append(record)
            family, sources = raw[3:].split(' | ')
            record = {'family': family, 'sources': sources.split(','),
                      'author': 'Codex original synthetic dialogue', 'messages': []}
        elif raw.startswith(('U: ', 'A: ')):
            assert record is not None
            role = 'user' if raw.startswith('U: ') else 'assistant'
            record['messages'].append({'role': role, 'content': raw[3:]})
        elif raw and not raw.startswith('#'):
            raise ValueError(f'Unexpected line: {raw}')
    if record:
        rows.append(record)
    return rows

def expand(rows):
    return [dict(row, messages=row['messages'][:i+1])
            for row in rows for i, msg in enumerate(row['messages'])
            if msg['role'] == 'assistant']

def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--include-augmentation',action='store_true')
    parser.add_argument('--character-overrides',type=Path)
    parser.add_argument('--extra-source',type=Path,action='append',default=[])
    parser.add_argument('--output',type=Path,default=ROOT/'data/prepared')
    args=parser.parse_args()
    train, valid = parse(ROOT/'data/seeds.txt'), parse(ROOT/'data/validation.txt')
    if args.include_augmentation:
        train+=parse(ROOT/'data/augmentation_pool.txt')
    for source in args.extra_source:
        train+=parse(source)
    assert len({r['family'] for r in train})==len(train),'Duplicate training family'
    if args.character_overrides:
        overrides=json.loads(args.character_overrides.read_text())
        seen=set()
        for row in train:
            replacements=overrides['families'].get(row['family'])
            if replacements is None: continue
            targets=[m for m in row['messages'] if m['role']=='assistant']
            assert len(targets)==len(replacements),row['family']
            for target,replacement in zip(targets,replacements): target['content']=replacement
            row['character_version']=overrides['version']
            seen.add(row['family'])
        assert seen==set(overrides['families']), 'Unknown override family'
    assert not ({r['family'] for r in train} & {r['family'] for r in valid})
    write(args.output/'train.jsonl', expand(train))
    write(args.output/'valid.jsonl', expand(valid))
    # MLX's test command is used for repeated DEVELOPMENT loss, not final holdout.
    write(args.output/'test.jsonl', expand(valid))
    conversations=args.output/'conversations.jsonl' if args.output!=ROOT/'data/prepared' else ROOT/'data/conversations.jsonl'
    write(conversations, [dict(r, split=s) for s, rows in [('train',train),('validation',valid)] for r in rows])
    paths = ['data/seeds.txt','data/validation.txt','evaluation/dev.json','evaluation/final.json','character/core.md','sources/manifest.json']
    if args.include_augmentation:
        paths.append('data/augmentation_pool.txt')
    if args.character_overrides:
        paths.append(str(args.character_overrides))
    paths.extend(str(p) for p in args.extra_source)
    manifest = {'sha256': {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},
                'train_conversations': len(train), 'train_assistant_turns': len(expand(train)),
                'validation_conversations': len(valid), 'validation_assistant_turns': len(expand(valid)),
                'augmentation_included':args.include_augmentation,
                'split_policy': 'Explicit scenario families before turn expansion; development and final prompts authored separately.'}
    manifest_path=args.output/'manifest.json' if args.output!=ROOT/'data/prepared' else ROOT/'data/manifest.json'
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))

if __name__ == '__main__':
    main()
