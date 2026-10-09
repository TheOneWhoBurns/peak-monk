"""Assemble reviewed, measured overlap supervision without changing old data.

Raw TTS and its unprompted ASR remain intact. A separate approval file must bind
the render manifest and every alignment hash. This script does not approve its
own transcripts or admit its output to a training run.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import soundfile as sf
from stitch_duplex import mix, read_mono, RATE


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def choose_boundary(words, agent_seconds, caller_seconds, kind):
    """Closest interior completed word to the midpoint; no output-quality search."""
    remaining = 2.0 if kind == 'barge_in' else caller_seconds + 2.0
    candidates = [w['end'] for w in words
                  if 2.0 <= w['end'] <= agent_seconds - remaining]
    if not candidates:
        raise ValueError('No interior word boundary meeting the duration gate')
    return min(candidates, key=lambda end: (abs(end-agent_seconds/2), end))


def event_gaps(agent_seconds, caller_seconds, boundary, kind):
    if kind == 'barge_in':
        return -.48, .28
    if kind != 'backchannel':
        raise ValueError('Unknown overlap kind')
    return boundary-agent_seconds, max(0., agent_seconds-boundary-caller_seconds)+.28


def overlap_seconds(pcm, placements):
    event = placements[2]
    start = round(event['start_seconds']*RATE)
    end = round(event['end_seconds']*RATE)
    segment = pcm[start:end].astype(np.float32)/32768
    blocks = len(segment)//480
    rms = np.sqrt((segment[:blocks*480].reshape(blocks,480,2)**2).mean(axis=1))
    return float(np.all(rms > 10**(-45/20), axis=1).sum()*.02)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--draft',type=Path,default=Path('data/speech_overlap_v1.json'))
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--approved',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    approval=json.loads(a.approved.read_text())
    if approval.get('render_manifest_sha256')!=digest(a.manifest):
        raise ValueError('Approval must bind this exact render manifest')
    if approval.get('draft_sha256')!=digest(a.draft):
        raise ValueError('Approval must bind this exact authored draft')
    kinds={c['id']:c['kind'] for c in json.loads(a.draft.read_text())['cases']}
    all_rows=[json.loads(line) for line in a.manifest.read_text().splitlines()]
    accepted=set(approval['accepted_ids'])
    if not accepted or not accepted <= {r['id'] for r in all_rows}:
        raise ValueError('Approval contains missing or no rendered cases')
    a.output.mkdir(parents=True,exist_ok=False)
    outputs=[];evidence=[]
    for raw in all_rows:
        if raw['id'] not in accepted:
            continue
        ident=raw['id'];kind=kinds[ident]
        audit_path=a.manifest.parent/f'{ident}.alignment.json'
        if approval['alignment_sha256'].get(ident)!=digest(audit_path):
            raise ValueError(f'Alignment differs from approval: {ident}')
        audit=json.loads(audit_path.read_text())
        if audit.get('initial_prompt','missing') is not None:
            raise ValueError('This experiment requires unprompted ASR')
        if not audit.get('accepted') or not all(t.get('accepted') for t in audit['turns']):
            raise ValueError('Resolve rejected source alignments before assembly')
        row=copy.deepcopy(raw);turns=row['turns'];original_audit=copy.deepcopy(audit)
        if [t['role'] for t in turns] != ['user','assistant','user','assistant']:
            raise ValueError('Expected exactly four alternating turns')
        for turn,aligned in zip(turns,audit['turns'],strict=True):
            if turn['wav']!=aligned['wav']:
                raise ValueError('Turn/alignment ordering mismatch')
            source=a.manifest.parent/turn['wav']
            if digest(source)!=turn['wav_sha256']:
                raise ValueError('Source audio changed since rendering')
            if not aligned['words'] or not all(0<=w['start']<w['end']<=turn['duration_seconds']+.1 for w in aligned['words']):
                raise ValueError('Invalid word alignment')
        applied=[]
        for correction in approval.get('word_corrections',[]):
            if correction['id']!=ident:
                continue
            aligned=next(t for t in audit['turns'] if t['wav']==correction['wav'])
            word=aligned['words'][correction['word_index']]
            if word['word']!=correction['asr_word']:
                raise ValueError('Stale spelling correction')
            word['word']=correction['replacement'];applied.append(correction)
        agent=read_mono(a.manifest.parent/turns[1]['wav'])
        caller=read_mono(a.manifest.parent/turns[2]['wav'])
        original_duration=len(agent)/RATE
        boundary=choose_boundary(audit['turns'][1]['words'],original_duration,len(caller)/RATE,kind)
        boundary=round(boundary*RATE)/RATE
        if kind=='barge_in':
            cut=round(boundary*RATE);agent=agent[:cut].copy()
            fade=min(120,len(agent))
            agent[-fade:]=np.rint(agent[-fade:]*np.linspace(1.,0.,fade)).astype(np.int32)
            aligned=audit['turns'][1]
            aligned['words']=[w for w in aligned['words'] if w['end']<=boundary+1/RATE]
            if not aligned['words']:
                raise ValueError('No emitted words survived clipping')
            turns[1]['content']=' '.join(w['word'].strip() for w in aligned['words'])
            turns[1]['duration_seconds']=len(agent)/RATE
            aligned['reference']=turns[1]['content'];aligned['recognized']=turns[1]['content']
            aligned['scope']='Retained prefix of approved original ASR, not a new transcription or WER measurement'
            aligned.pop('wer',None)
        caller_gap,followup_gap=event_gaps(original_duration,len(caller)/RATE,boundary,kind)
        turns[0]['gap_ms']=400;turns[1]['gap_ms']=400
        turns[2]['gap_ms']=caller_gap*1000;turns[3]['gap_ms']=followup_gap*1000
        for index,turn in enumerate(turns):
            dest=a.output/turn['wav']
            if index==1 and kind=='barge_in':
                sf.write(dest,agent.astype(np.int16),RATE,subtype='PCM_16')
            else:
                os.link((a.manifest.parent/turn['wav']).resolve(),dest)
            turn['wav_sha256']=digest(dest)
        pcm,placements=mix(turns,a.output)
        simultaneous=overlap_seconds(pcm,placements)
        if len(pcm)/RATE>36.0:
            raise ValueError(f'{ident}: episode exceeds declared36second budget')
        if simultaneous < .08:
            raise ValueError(f'{ident}: insufficient measured activity overlap ({simultaneous}s)')
        if kind=='backchannel' and placements[1]['end_seconds']-placements[2]['end_seconds']<2.0:
            raise ValueError('Insufficient continuation after backchannel')
        sf.write(a.output/f'{ident}.stereo.wav',pcm,RATE,subtype='PCM_16')
        audit['accepted']=True
        audit['scope']='Author-approved source transcript, modified only by documented prefix clipping and explicit spelling corrections; output requires admission review'
        audit['original_alignment_sha256']=digest(audit_path)
        audit['original_asr']=original_audit
        (a.output/f'{ident}.alignment.json').write_text(json.dumps(audit,indent=2)+'\n')
        row['quality_status']='Source author-approved; measured overlap assembly, pending training admission'
        outputs.append(row)
        evidence.append(dict(id=ident,kind=kind,original_agent_seconds=original_duration,
            boundary_seconds=boundary,simultaneous_rms_active_seconds=simultaneous,
            episode_seconds=len(pcm)/RATE,placements=placements,word_corrections=applied))
    (a.output/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in outputs))
    report=dict(status='Assembled; pending admission to a separately declared experiment',
        approval_sha256=digest(a.approved),draft_sha256=digest(a.draft),
        source_manifest_sha256=digest(a.manifest),output_manifest_sha256=digest(a.output/'manifest.jsonl'),
        cases=evidence,scope='Designed synthetic overlap; RMS gate is not an intelligibility or naturalness rating')
    (a.output/'overlap-assembly.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='cases'},indent=2))


if __name__=='__main__':
    main()
