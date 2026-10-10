"""Merge explicitly reviewed sampled failures and retention targets.

Caller turns and prior student replies remain unchanged. Only a repaired final
reply or a reviewed positive reply is supervised for each new sampled example.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def main():
    selection_path = Path('data/on-policy-review-selection-v3.json')
    selection = json.loads(selection_path.read_text())
    if selection['status'] != 'Author review complete':
        raise ValueError('Complete explicit review before constructing the dataset')
    prior = Path('data/on-policy-repairs-v2/train.jsonl')
    retained = Path('data/student-retention-v1/train.jsonl')
    canonical = Path('data/on-policy-repairs-v1/replay.jsonl')
    repairs, replay = read_rows(prior), read_rows(retained) + read_rows(canonical)
    reviews = []
    for kind, destination in [('repairs', repairs), ('retention', replay)]:
        for item in selection[kind]:
            path = Path(item['path'])
            if sha(path) != item['sha256']:
                raise ValueError('Reviewed source changed')
            record = json.loads(path.read_text())
            assert record['status'] == 'complete' and record['split'] == 'train'
            index = item['turn_index']
            assert 0 <= index < len(record['turns'])
            prefix_count = len(record['messages']) - 2*len(record['turns'])
            end = prefix_count + 2*(index+1)
            messages = [dict(m) for m in record['messages'][:end]]
            assert messages[-1]['role'] == 'assistant'
            assert messages[-1]['content'] == record['turns'][index]['reply']
            original = messages[-1]['content']
            if kind == 'repairs':
                assert item['replacement'].strip() and item['replacement'] != original
                messages[-1]['content'] = item['replacement']
            ident = f"{kind}_{record['id']}_{index:02d}"
            source = dict(path=str(path), sha256=item['sha256'], exchange=end//2)
            row = dict(id=ident, family=record['family'], root_id=record['root_id'],
                split='train', review='author-approved', messages=messages,
                supervised_message_indices=[end-1], source=source,
                review_reason=item['reason'])
            destination.append(row)
            reviews.append(dict(id=ident, kind=kind, source=source,
                original_reply=original, selected_reply=messages[-1]['content'], reason=item['reason']))
    ids = [r['id'] for r in repairs+replay if 'id' in r]
    assert len(ids) == len(set(ids))
    assert all(r['split'] == 'train' for r in repairs+replay)
    out = Path('data/on-policy-repairs-v3')
    out.mkdir(exist_ok=False)
    for name, rows in [('train', repairs), ('replay', replay)]:
        (out/(name+'.jsonl')).write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in rows))
    report = dict(status='Prepared, not trained', reviewer='Codex author; not independent',
        sources={str(p):sha(p) for p in [prior, retained, canonical, selection_path]},
        repairs=len(repairs), replay=len(replay), reviews=reviews,
        replay_composition=dict(student_positive=len(read_rows(retained)),
                                canonical=len(read_rows(canonical)),
                                additional_sampled_positive=len(selection['retention'])),
        repair_families=dict(Counter(r['family'] for r in repairs)),
        data_sha256={name:sha(out/(name+'.jsonl')) for name in ['train','replay']},
        objective='Final-reply-only supervision for all sampled examples; original canonical replay masks retained.',
        limitations='Only two sampled scenario families. Small selected training set, not a generalization benchmark. Older final/pressure evaluation prompts and answers are excluded.')
    (out/'manifest.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='reviews'}, indent=2))


if __name__ == '__main__':
    main()
