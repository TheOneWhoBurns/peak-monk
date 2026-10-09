"""Deterministic family-first corpus preparation, never random turn splitting."""
import hashlib
import json
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
    train, valid = parse(ROOT/'data/seeds.txt'), parse(ROOT/'data/validation.txt')
    assert not ({r['family'] for r in train} & {r['family'] for r in valid})
    write(ROOT/'data/prepared/train.jsonl', expand(train))
    write(ROOT/'data/prepared/valid.jsonl', expand(valid))
    # MLX's test command is used for repeated DEVELOPMENT loss, not final holdout.
    write(ROOT/'data/prepared/test.jsonl', expand(valid))
    write(ROOT/'data/conversations.jsonl', [dict(r, split=s) for s, rows in [('train',train),('validation',valid)] for r in rows])
    paths = ['data/seeds.txt','data/validation.txt','evaluation/dev.json','evaluation/final.json','character/core.md','sources/manifest.json']
    manifest = {'sha256': {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},
                'train_conversations': len(train), 'train_assistant_turns': len(expand(train)),
                'validation_conversations': len(valid), 'validation_assistant_turns': len(expand(valid)),
                'split_policy': 'Explicit scenario families before turn expansion; development and final prompts authored separately.'}
    (ROOT/'data/manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))

if __name__ == '__main__':
    main()
