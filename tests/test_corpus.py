import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def test_split_integrity_and_provenance():
    records=[json.loads(l) for l in (ROOT/'data/conversations.jsonl').read_text().splitlines()]
    sources={s['id'] for s in json.loads((ROOT/'sources/manifest.json').read_text())}
    seen=set()
    for r in records:
        assert r['family'] not in seen
        seen.add(r['family'])
        assert set(r['sources'])<=sources
        assert len(r['messages'])%2==0
        assert all(m['role']==('user' if i%2==0 else 'assistant') and m['content'].strip()
                   for i,m in enumerate(r['messages']))
    train={m['content'].lower() for r in records if r['split']=='train' for m in r['messages']}
    for suite in ['dev','final']:
        for case in json.loads((ROOT/f'evaluation/{suite}.json').read_text()):
            assert not (train & {u.lower() for u in case['turns']})

def test_frozen_evaluation():
    lock=json.loads((ROOT/'evaluation/frozen.json').read_text())
    for path,sha in lock.items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha, path

def test_turn_expansion_masks_only_complete_responses():
    families={}
    for line in (ROOT/'data/prepared/train.jsonl').read_text().splitlines():
        r=json.loads(line)
        assert r['messages'][-1]['role']=='assistant'
        assert not any(m['role']=='system' for m in r['messages'])
        families.setdefault(r['family'],[]).append(r['messages'])
    for prefixes in families.values():
        for a,b in zip(prefixes,prefixes[1:]):
            assert len(b)==len(a)+2 and b[:len(a)]==a
