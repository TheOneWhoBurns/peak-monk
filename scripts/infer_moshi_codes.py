"""Run native duplex inference on prerecorded user Mimi codes on this Mac.

Input must be a NumPy archive with user_codes shaped [8, frames]. The other
speaker remains a continuously sampled native Moshi stream. This is not an
ASR/text/TTS cascade. Decode returned agent codes separately to avoid loading
the codec alongside the full speech model on an 8 GB machine.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from moshi_mlx.models import Lm, LmGen, config_v0_1
from moshi_mlx.utils import Sampler
import sentencepiece as spm
from moshi_training import attach_lora

p=argparse.ArgumentParser()
p.add_argument('input')
p.add_argument('--adapter')
p.add_argument('--adapter-config')
p.add_argument('--output',required=True)
p.add_argument('--seed',type=int,default=41)
p.add_argument('--max-frames',type=int,help='Explicit short feasibility probe, not a complete dialogue audit')
args=p.parse_args()
out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
codes=np.load(args.input)['user_codes']
original_frames=codes.shape[1]
if args.max_frames is not None:
    if args.max_frames<3:raise ValueError('Need at least three frames for delayed audio')
    codes=codes[:,:args.max_frames]
assert codes.ndim==2 and codes.shape[0]==8 and np.all((codes>=0)&(codes<2048))
weight=Path('models/moshi/model.q4.safetensors')
with weight.open('rb') as f:
    assert hashlib.file_digest(f,'sha256').hexdigest()=='7959d590e23c1ebc78cfa3501344a6ff331561aa0cadc4429b733b890bbc919c'
mx.metal.set_memory_limit(5_700_000_000);mx.metal.set_cache_limit(64_000_000)
wired_limit=mx.metal.device_info()['max_recommended_working_set_size']
prior_wired_limit=mx.metal.set_wired_limit(wired_limit)
print('Wired memory limit:',prior_wired_limit,'->',wired_limit,flush=True)
mx.random.seed(args.seed)
load_start=time.monotonic()
model=Lm(config_v0_1());model.set_dtype(mx.bfloat16)
nn.quantize(model,bits=4,group_size=32);model.load_weights(str(weight),strict=True)
if args.adapter:
    cfg=json.loads(Path(args.adapter_config).read_text())
    attach_lora(model,layers=cfg['layers'],rank=cfg['rank'],scale=cfg['scale'])
    model.load_weights(args.adapter,strict=False)
mx.eval(model.parameters());model.eval()
load_seconds=time.monotonic()-load_start
print('Loaded native Moshi:',mx.metal.get_active_memory(),'active bytes;',load_seconds,'seconds',flush=True)
warmup_start=time.monotonic();model.warmup()
warmup_seconds=time.monotonic()-warmup_start
print('Warmup complete:',warmup_seconds,'seconds',flush=True)
# Adapter initialization and warmup must not shift the matched sampling stream.
mx.random.seed(args.seed)
tok=spm.SentencePieceProcessor(model_file='models/moshi/tokenizer_spm_32k_3.model')
text_sampler=Sampler();audio_sampler=Sampler()
gen=LmGen(model,max_steps=codes.shape[1]+5,text_sampler=text_sampler,audio_sampler=audio_sampler)
text=[];audio=[];frames=[];start=time.monotonic()
for i in range(codes.shape[1]):
    tick=time.monotonic()
    result=gen.step(mx.array(codes[:,i][None,:]))
    token=result[0] if isinstance(result,tuple) else result
    token=int(token.item())
    agent=gen.last_audio_tokens()
    if agent is not None: audio.append(np.array(agent).astype(np.uint32)[0])
    piece=tok.id_to_piece(token).replace('▁',' ') if token not in (0,3) else ''
    text.append(piece)
    frames.append({'input_frame':i,'text_token':token,'piece':piece,'seconds':time.monotonic()-tick,
                   'agent_frame_available':agent is not None,
                   'agent_audio_frame_index':len(audio)-1 if agent is not None else None})
    if i<3 or (i+1)%50==0: print(i+1,'/',codes.shape[1],frames[-1]['seconds'],'seconds;', ''.join(text)[-180:],flush=True)
elapsed=time.monotonic()-start
np.savez_compressed(out/'agent_codes.npz',agent_codes=np.stack(audio,axis=1))
(out/'transcript.txt').write_text(''.join(text).strip()+'\n')
report={'input':args.input,'adapter':args.adapter,'seed':args.seed,'input_seconds':codes.shape[1]*.08,
        'original_input_frames':original_frames,'explicit_frame_limit':args.max_frames,
        'input_sha256':hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
        'model_load_seconds':load_seconds,'warmup_seconds':warmup_seconds,
        'wired_limit_bytes':wired_limit,'prior_wired_limit_bytes':prior_wired_limit,
        'text_sampler':asdict(text_sampler),'audio_sampler':asdict(audio_sampler),
        'generation_seconds':elapsed,'generation_realtime_factor':elapsed/(codes.shape[1]*.08),
        'scope':'Model generation only; codec preprocessing and decoding timed separately',
        'sampling_seed_policy':'Reset after loading, adapter initialization and warmup; identical seed per baseline/candidate trial',
        'peak_model_memory_bytes':mx.metal.get_peak_memory(),'frames':frames,
        'waveform_timing_note':'Generated codes are in original frame order. Decoder/stream scheduling latency must be calibrated before claiming response latency.'}
(out/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='frames'},indent=2),flush=True)
