"""Validate explicit author reviews and calculate release gates, never invent ratings."""
import argparse
import hashlib
import json
from pathlib import Path

def gate(transcript, review, baseline_loss):
    suite_path=Path(transcript['suite'])
    suite_hash=hashlib.sha256(suite_path.read_bytes()).hexdigest()
    if suite_hash!=transcript['suite_sha256']:
        raise ValueError('Suite changed after capture')
    if 'roleplay-v2' in transcript['suite'] and review.get('suite_sha256')!=suite_hash:
        raise ValueError('Versioned character review must explicitly identify the captured rubric')
    ids={c['id'] for c in transcript['conversations']}
    rows=review['ratings']
    if len(rows)!=len(ids) or {r['id'] for r in rows}!=ids:
        raise ValueError('Review must cover every captured conversation exactly once')
    if len(ids)!=len(json.loads(Path(transcript['suite']).read_text())):
        raise ValueError('Transcript is incomplete')
    for r in rows:
        if type(r['acceptable']) is not bool or type(r['severe_failure']) is not bool or not r['reason'].strip():
            raise ValueError('Explicit boolean ratings and reasons are required')
    score=sum(r['acceptable'] for r in rows)/len(rows)
    improvement=1-transcript['validation_loss']/baseline_loss
    severe=[r['id'] for r in rows if r['severe_failure']]
    critical_categories={'identity','identity_stability','control','senses','lore'}
    critical_ids={c['id'] for c in transcript['conversations'] if c.get('category') in critical_categories}
    # The original frozen final suite predates category metadata. Apply the
    # existing essential-character rule without rewriting its frozen prompts.
    if suite_path.name=='final-roleplay-v2.json':
        if suite_hash!='d40b45d84e91e19acc5d7cd0f66c1371557df8c02dca7389becb472b56ac580b':
            raise ValueError('Unknown final suite: explicit essential-case mapping required')
        critical_ids.update({'f01','f04','f05','f09','f13'})
    critical_failures=[r['id'] for r in rows if r['id'] in critical_ids and not r['acceptable']]
    return {'reviewer':review['reviewer'],'independent':False,
            'suite':transcript['suite'],'suite_sha256':suite_hash,
            'acceptable_fraction':score,'acceptable_count':sum(r['acceptable'] for r in rows),
            'total':len(rows),'relative_loss_reduction':improvement,
            'severe_failures':severe,'required_character_cases':sorted(critical_ids),
            'required_character_failures':critical_failures,
            'gate_version':'human_character_essential_v4_final_mapping',
            'pass':score>=.8 and improvement>=.15 and not severe and not critical_failures,
            'scope':'Project-specific author rubric; not independent human benchmark or general safety certification.'}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('transcript',type=Path)
    p.add_argument('review',type=Path)
    p.add_argument('--baseline',type=Path,default=Path('results/baseline.json'))
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    transcript=json.loads(args.transcript.read_text())
    baseline=json.loads(args.baseline.read_text())
    for key,default in [('model',None),('loss_version','legacy')]:
        if transcript.get(key,default)!=baseline.get(key,default):
            raise ValueError(f'Incomparable baseline: {key} differs')
    if baseline.get('serialization_sha256') and transcript.get('serialization_sha256')!=baseline['serialization_sha256']:
        raise ValueError('Incomparable serialization fingerprints')
    result=gate(transcript,json.loads(args.review.read_text()),baseline['validation_loss'])
    encoded=json.dumps(result,indent=2)+'\n'
    if args.output: args.output.write_text(encoded)
    print(encoded,end='')

if __name__=='__main__':main()
