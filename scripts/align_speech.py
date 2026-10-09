"""ASR-check rendered utterances and retain actual Whisper word timestamps.

The ASR output, word error rate and exclusions are visible. We do not spread
transcript words uniformly over a WAV or substitute invented timestamps.
"""
import argparse
import json
from pathlib import Path
import re
import time
import subprocess
import hashlib
import numpy as np
from faster_whisper import WhisperModel


def words(s):
    return re.findall(r"[a-z]+(?:'[a-z]+)?|\d+",s.lower())


def distance(a,b):
    row=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        new=[i]
        for j,y in enumerate(b,1): new.append(min(new[-1]+1,row[j]+1,row[j-1]+(x!=y)))
        row=new
    return row[-1]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',default='data/renders/manifest.jsonl')
    p.add_argument('--model',default=str(Path.home()/'.cache/huggingface/hub/models--Systran--faster-whisper-large-v3/snapshots/edaa852ec7e145841d8ffdb056a99866b5f0a478'))
    p.add_argument('--limit',type=int,default=8)
    p.add_argument('--max-wer',type=float,default=.20)
    p.add_argument('--retry-suspect',action='store_true',help='Preserve prior audit and rerun only rejected or nonexact turns')
    args=p.parse_args()
    manifest=Path(args.manifest)
    rows=[json.loads(l) for l in manifest.read_text().splitlines()][:args.limit]
    model=WhisperModel(args.model,device='cpu',compute_type='int8',cpu_threads=4,local_files_only=True)
    for row in rows:
        dest=manifest.parent/f"{row['id']}.alignment.json"
        previous=None
        if dest.exists():
            if not args.retry_suspect: continue
            previous=json.loads(dest.read_text())
            if all(t['accepted'] and t['wer']==0 for t in previous['turns']):continue
            archive=manifest.parent/'alignment-history';archive.mkdir(exist_ok=True)
            sha=hashlib.sha256(dest.read_bytes()).hexdigest()
            backup=archive/f"{row['id']}-{sha[:16]}.json"
            if not backup.exists():backup.write_bytes(dest.read_bytes())
        audit={'id':row['id'],'asr_model':args.model,'word_timestamps':'Whisper decoder alignment estimates',
               'max_wer':args.max_wer,'turns':[],'accepted':True}
        if previous:audit['prior_audit']=str(backup)
        for turn in row['turns']:
            prior=next((t for t in previous['turns'] if t['wav']==turn['wav']),None) if previous else None
            if prior and prior['accepted'] and prior['wer']==0:
                audit['turns'].append(dict(prior,asr_model=previous['asr_model']));continue
            start=time.monotonic()
            # PyAV 19 removed a keyword still used by faster-whisper 1.2.1.
            # Supplying decoded 16 kHz samples bypasses that incompatible adapter.
            pcm=subprocess.check_output(['ffmpeg','-v','error','-i',str(manifest.parent/turn['wav']),
                                         '-f','f32le','-ac','1','-ar','16000','-'])
            samples=np.frombuffer(pcm,dtype='<f4').copy()
            segments,info=model.transcribe(samples,language='en',
                beam_size=1,word_timestamps=True,vad_filter=args.retry_suspect,condition_on_previous_text=False,
                initial_prompt='The speaker names in these recordings include Jun and Mara.')
            segments=list(segments)
            recognized=' '.join(s.text.strip() for s in segments)
            ref,hyp=words(turn['content']),words(recognized)
            wer=distance(ref,hyp)/max(len(ref),1)
            aligned=[{'word':w.word.strip(),'start':w.start,'end':w.end,'probability':w.probability}
                     for s in segments for w in (s.words or [])]
            timing_valid=bool(aligned) and all(0<=w['start']<w['end']<=turn['duration_seconds']+.1 for w in aligned)
            accepted=wer<=args.max_wer and timing_valid
            audit['accepted'] &= accepted
            audit['turns'].append({'wav':turn['wav'],'reference':turn['content'],'recognized':recognized,
                'wer':wer,'accepted':accepted,'words':aligned,'seconds':time.monotonic()-start,'asr_model':args.model})
        dest.write_text(json.dumps(audit,indent=2,ensure_ascii=False)+'\n')
        print(row['id'],audit['accepted'],[round(t['wer'],2) for t in audit['turns']],flush=True)


if __name__=='__main__': main()
