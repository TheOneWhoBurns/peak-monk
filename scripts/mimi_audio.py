"""Local Mimi codec utility, with separate runtime measurements."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import soundfile as sf
import rustymimi

p=argparse.ArgumentParser()
p.add_argument('mode',choices=['encode','decode'])
p.add_argument('input')
p.add_argument('--output',required=True)
p.add_argument('--codes-key',choices=['agent_codes','user_codes'],default='agent_codes',help='For decoding: distinguish generated audio from an input-codec round trip')
args=p.parse_args()
weight=Path('models/moshi/tokenizer-e351c8d8-checkpoint125.safetensors')
with weight.open('rb') as f:
    assert hashlib.file_digest(f,'sha256').hexdigest()=='09b782f0629851a271227fb9d36db65c041790365f11bbe5d3d59369cf863f50'
mimi=rustymimi.Tokenizer(str(weight),num_codebooks=8)
start=time.monotonic();chunks=[]
if args.mode=='encode':
    raw=subprocess.check_output(['ffmpeg','-v','error','-i',args.input,'-f','f32le','-ar','24000','-ac','1','-'])
    pcm=np.frombuffer(raw,dtype='<f4').copy();duration=len(pcm)/24000
    pcm=np.pad(pcm,(0,(-len(pcm))%1920))
    for i in range(0,len(pcm),1920):
        codes=mimi.encode_step(pcm[i:i+1920][None,None,:])
        if codes is not None: chunks.append(codes)
    codes=np.concatenate(chunks,axis=-1)[0]
    assert codes.shape==(8,len(pcm)//1920)
    np.savez_compressed(args.output,user_codes=codes)
else:
    codes=np.load(args.input)[args.codes_key].astype(np.uint32)
    duration=codes.shape[1]*.08
    for i in range(codes.shape[1]):
        pcm=mimi.decode_step(np.ascontiguousarray(codes[None,:,i:i+1]))
        if pcm is not None: chunks.append(pcm)
    pcm=np.concatenate(chunks,axis=-1)[0,0]
    assert np.isfinite(pcm).all()
    sf.write(args.output,np.clip(pcm,-.999,.999),24000,subtype='PCM_16')
elapsed=time.monotonic()-start
report={'mode':args.mode,'input':args.input,'audio_seconds':duration,'wall_seconds':elapsed,
        'codes_key':args.codes_key if args.mode=='decode' else 'user_codes',
        'realtime_factor':elapsed/duration,'scope':'CPU codec including audio I/O, excluding model-load time'}
Path(args.output+'.metrics.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
