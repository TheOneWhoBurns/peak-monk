"""Build real stereo Moshi training audio from rendered, explicitly timed turns.

Input JSONL records:
{"id":"...","split":"train","turns":[
 {"role":"user","text":"...","wav":"renders/u.wav","gap_ms":400},
 {"role":"assistant","text":"...","wav":"renders/a.wav","gap_ms":-180}]}

gap_ms is relative to the previous turn's END. Negative values produce overlap.
Input must be mono 24 kHz PCM16. No fake audio or fabricated word alignment.
The .turns.json files are provenance, NOT Moshi's word-timestamp annotations.
Run the official moshi-finetune annotate.py on each output manifest afterwards.
"""
import argparse
import json
from pathlib import Path
import wave
import numpy as np

RATE=24000

def read_mono(path):
    with wave.open(str(path),'rb') as f:
        if (f.getnchannels(),f.getsampwidth(),f.getframerate())!=(1,2,RATE):
            raise ValueError(f'{path}: expected mono PCM16 at {RATE} Hz')
        samples=np.frombuffer(f.readframes(f.getnframes()),dtype='<i2').astype(np.int32)
    if len(samples)==0 or not np.any(samples):
        raise ValueError(f'{path}: empty or silent rendered turn')
    return samples

def mix(turns, root):
    placed=[]
    previous_end=0
    annotations=[]
    for turn in turns:
        if turn['role'] not in ('user','assistant'):
            raise ValueError('Only user/assistant audio is accepted')
        audio=read_mono(root/turn['wav'])
        start=previous_end+round(turn.get('gap_ms',400)*RATE/1000)
        if start<0:
            raise ValueError('Turn would start before the recording')
        end=start+len(audio)
        channel=0 if turn['role']=='assistant' else 1
        placed.append((start,end,channel,audio))
        annotations.append(dict(turn,start_seconds=start/RATE,end_seconds=end/RATE,channel=channel))
        previous_end=end
    if not placed:
        raise ValueError('No turns')
    # Preserve an earlier long overlapping turn even when a short backchannel ends first.
    output=np.zeros((max(p[1] for p in placed)+RATE,2),dtype=np.int32)
    occupancy=np.zeros_like(output,dtype=bool)
    for start,end,channel,audio in placed:
        if occupancy[start:end,channel].any():
            raise ValueError('Overlapping utterances from the same speaker need explicit editing')
        output[start:end,channel]+=audio
        occupancy[start:end,channel]=True
    if np.max(np.abs(output))>32767:
        raise ValueError('Clipping detected; normalize source renders first')
    return output.astype('<i2'),annotations

def main():
    p=argparse.ArgumentParser()
    p.add_argument('manifest',type=Path)
    p.add_argument('--output',type=Path,default=Path('data/duplex'))
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    manifests={}
    seen=set()
    for line in args.manifest.read_text().splitlines():
        record=json.loads(line)
        ident=record['id']
        if not ident.replace('-','').replace('_','').isalnum() or ident in seen:
            raise ValueError('IDs must be unique simple names')
        seen.add(ident)
        split=record['split']
        if split not in ('train','valid','test'):
            raise ValueError('Invalid split')
        audio,annotations=mix(record['turns'],args.manifest.parent)
        dest=args.output/f'{ident}.wav'
        with wave.open(str(dest),'wb') as f:
            f.setnchannels(2);f.setsampwidth(2);f.setframerate(RATE)
            f.writeframes(audio.tobytes())
        dest.with_suffix('.turns.json').write_text(json.dumps({
            'id':ident,'split':split,'agent_channel':0,'user_channel':1,
            'word_alignment':'NOT GENERATED; run official annotate.py',
            'turns':annotations},indent=2)+'\n')
        manifests.setdefault(split,[]).append({'path':str(dest.resolve()),'duration':len(audio)/RATE})
    for split,rows in manifests.items():
        (args.output/f'{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    print(json.dumps({split:len(rows) for split,rows in manifests.items()}))

if __name__=='__main__': main()
