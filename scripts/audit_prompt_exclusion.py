"""Check caller-turn exclusion against explicit frozen evaluation suite schemas.

Exact normalized matching is a narrow integrity check, not semantic independence.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re


def normalize(text):
    return ' '.join(re.findall(r'\w+', text.casefold()))


def audit(suites, corpora):
    reserved = {}
    suite_counts = {}
    data_counts = {}
    collisions = []
    for path in suites:
        cases = json.loads(path.read_text())
        if not isinstance(cases, list) or not cases:
            raise ValueError(f'Expected a nonempty evaluation case list: {path}')
        count = 0
        for case in cases:
            turns = case.get('turns')
            if not case.get('id') or not isinstance(turns, list) or not turns:
                raise ValueError(f'Missing case ID or caller turns: {path}')
            for index, turn in enumerate(turns):
                if not isinstance(turn, str) or not normalize(turn):
                    raise ValueError(f'Invalid evaluation caller text: {path}')
                reserved.setdefault(normalize(turn), []).append(dict(suite=str(path),case=case['id'],evaluation_turn=index))
                count += 1
        suite_counts[str(path)] = count
    if not reserved:
        raise ValueError('No evaluation caller turns were checked')
    for path in corpora:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if not rows:
            raise ValueError(f'Empty corpus: {path}')
        count = 0
        for row_index,row in enumerate(rows):
            messages = row.get('messages')
            if not isinstance(messages,list) or not messages:
                raise ValueError(f'Missing messages: {path}')
            callers = 0
            for index,message in enumerate(messages):
                if message.get('role') not in ('user','assistant') or not isinstance(message.get('content'),str):
                    raise ValueError(f'Unknown message schema: {path}')
                if message['role']=='user':
                    key = normalize(message['content'])
                    if not key: raise ValueError(f'Empty caller text: {path}')
                    count += 1
                    callers += 1
                    for source in reserved.get(key,[]):
                        collisions.append(dict(corpus=str(path),row=row_index,id=row.get('id',row.get('family')),turn=index,**source))
            if not callers: raise ValueError(f'Corpus row has no callers: {path}')
        data_counts[str(path)] = count
    if not data_counts:
        raise ValueError('No corpora checked')
    return dict(status='pass' if not collisions else 'fail',kind='Exact normalized caller-turn exclusion',
        evaluation_caller_turn_counts=suite_counts,unique_evaluation_callers=len(reserved),
        corpus_caller_turn_counts=data_counts,collisions=collisions,
        evaluation_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in suites},
        data_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in corpora},
        limitations='Exact matching does not establish semantic independence. Same-author development-guided iteration remains disclosed; no deliberate final or pressure prompt rewriting is permitted.')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--suite',type=Path,action='append',required=True)
    p.add_argument('--corpus',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    # Every future project audit includes registered release exclusions even
    # when an older caller explicitly names only the v2 suites.
    root=Path(__file__).resolve().parents[1]
    registry=root/'evaluation/release-v3-frozen.json'
    registered=json.loads(registry.read_text())
    for name,expected in registered['suites'].items():
        path=root/'evaluation'/name
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError('Frozen release audit changed: '+str(path))
    suites=list(a.suite)
    for value in registered['required_exclusion_suites']:
        path=Path(value)
        if path.resolve() not in {item.resolve() for item in suites}:
            suites.append(path)
    report=audit(suites,a.corpus)
    report['required_registry_sha256']=hashlib.sha256(registry.read_bytes()).hexdigest()
    report['checker_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if report['status']!='pass':raise SystemExit(1)


if __name__=='__main__':main()
