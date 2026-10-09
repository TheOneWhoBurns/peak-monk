"""Validate explicit author reviews and calculate release gates, never invent ratings."""
import argparse
import json
from pathlib import Path

def gate(transcript, review, baseline_loss):
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
    return {'reviewer':review['reviewer'],'independent':False,
            'acceptable_fraction':score,'acceptable_count':sum(r['acceptable'] for r in rows),
            'total':len(rows),'relative_loss_reduction':improvement,
            'severe_failures':severe,'pass':score>=.8 and improvement>=.15 and not severe,
            'scope':'Project-specific author rubric; not independent human benchmark or general safety certification.'}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('transcript',type=Path)
    p.add_argument('review',type=Path)
    p.add_argument('--baseline',type=Path,default=Path('results/baseline.json'))
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    result=gate(json.loads(args.transcript.read_text()),json.loads(args.review.read_text()),
                json.loads(args.baseline.read_text())['validation_loss'])
    encoded=json.dumps(result,indent=2)+'\n'
    if args.output: args.output.write_text(encoded)
    print(encoded,end='')

if __name__=='__main__':main()
