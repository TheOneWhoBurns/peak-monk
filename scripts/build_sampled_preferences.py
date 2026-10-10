"""Recover actual sampled alternatives for previously reviewed training repairs."""
import argparse
import copy
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def actual_turn(path, expected_hash, message_index):
    if sha(path) != expected_hash:
        raise ValueError('Sampled ancestry changed')
    record = json.loads(Path(path).read_text())
    prefix = record['prefix_source']
    count = 2 * prefix['exchanges'] if prefix else 0
    if message_index < count:
        return actual_turn(prefix['path'], prefix['sha256'], message_index)
    index = (message_index-count)//2
    turn = record['turns'][index]
    assert record['messages'][message_index]['role'] == 'assistant'
    assert turn['reply'] == record['messages'][message_index]['content']
    return dict(path=str(path), sha256=expected_hash, turn_index=index, **turn)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--repairs',type=Path,default=Path('data/on-policy-repairs-v3/train.jsonl'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(exist_ok=False,parents=True)
    accepted=[];excluded=[]
    for line in args.repairs.read_text().splitlines():
        row=json.loads(line);source=row['source']
        path=source['path'] if isinstance(source,dict) else source
        expected=source['sha256'] if isinstance(source,dict) else row['source_sha256']
        assert row['split']=='train' and row['review']=='author-approved'
        original=json.loads(Path(path).read_text());index=len(row['messages'])-1
        assert sha(path)==expected and original['messages'][:index]==row['messages'][:index]
        rejected=original['messages'][index]['content']
        if row.get('rejected') is not None:assert row['rejected']==rejected
        turn=actual_turn(path,expected,index)
        if turn['finish_reason']!='stop':
            excluded.append(dict(id=row['id'],reason='Censored sampled alternative; do not add an invented EOS target',source=turn))
            continue
        pair=copy.deepcopy(row)
        pair['rejected']=rejected
        pair['negative_provenance']=turn
        pair['preference_reason']=row.get('reason',row.get('repair_reason',row.get('review_reason')))
        assert pair['preference_reason'] and pair['messages'][-1]['content']!=rejected
        accepted.append(pair)
    output=args.output/'train.jsonl'
    output.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in accepted))
    manifest=dict(source=str(args.repairs),source_sha256=sha(args.repairs),pairs=len(accepted),
        output_sha256=sha(output),excluded=excluded,
        scope='Offline author preferences over actual sampled student replies and previously reviewed synthetic replacements. Includes declared canonical preferences, not only definite factual errors. No reward model, independent judge, held-out data, or online reinforcement-learning claim.')
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(pairs=len(accepted),excluded=len(excluded),output=str(output))))


if __name__=='__main__':main()
