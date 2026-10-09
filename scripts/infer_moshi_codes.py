"""Run native duplex inference on prerecorded user Mimi codes on this Mac.

Input must be a NumPy archive with user_codes shaped [8, frames]. The other
speaker remains a continuously sampled native Moshi stream. This is not an
ASR/text/TTS cascade. Decode returned agent codes separately to avoid loading
the codec alongside the full speech model on an 8 GB machine.
"""
import argparse
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
args=p.parse_args()
out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
codes=np.load(args.input)['user_codes']
assert codes.ndim==2 and codes.shape[0]==8 and np.all((codes>=0)&(codes<2048))
weight=Path('models/moshi/model.q4.safetensors')
with weight.open('rb') as f:
    assert hashlib.file_digest(f,'sha256').hexdigest()=='7959d590e23c1ebc78cfa3501344a6ff331561aa0cadc4429b733b890bbc919c'
mx.metal.set_memory_limit(5_700_000_000);mx.metal.set_cache_limit(64_000_000)
mx.random.seed(args.seed)
model=Lm(config_v0_1());model.set_dtype(mx.bfloat16)
nn.quantize(model,bits=4,group_size=32);model.load_weights(str(weight),strict=True)
if args.adapter:
    cfg=json.loads(Path(args.adapter_config).read_text())
    attach_lora(model,layers=cfg['layers'],rank=cfg['rank'],scale=cfg['scale'])
    model.load_weights(args.adapter,strict=False)
mx.eval(model.parameters());model.eval();model.warmup()
# Adapter initialization and warmup must not shift the matched sampling stream.
mx.random.seed(args.seed)
tok=spm.SentencePieceProcessor(model_file='models/moshi/tokenizer_spm_32k_3.model')
gen=LmGen(model,max_steps=codes.shape[1]+5,text_sampler=Sampler(),audio_sampler=Sampler())
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
    if (i+1)%50==0: print(i+1,'/',codes.shape[1],''.join(text)[-180:],flush=True)
elapsed=time.monotonic()-start
np.savez_compressed(out/'agent_codes.npz',agent_codes=np.stack(audio,axis=1))
(out/'transcript.txt').write_text(''.join(text).strip()+'\n')
report={'input':args.input,'adapter':args.adapter,'seed':args.seed,'input_seconds':codes.shape[1]*.08,
        'generation_seconds':elapsed,'generation_realtime_factor':elapsed/(codes.shape[1]*.08),
        'scope':'Model generation only; codec preprocessing and decoding timed separately',
        'sampling_seed_policy':'Reset after loading, adapter initialization and warmup; identical seed per baseline/candidate trial',
        'peak_model_memory_bytes':mx.metal.get_peak_memory(),'frames':frames,
        'waveform_timing_note':'Generated codes are in original frame order. Decoder/stream scheduling latency must be calibrated before claiming response latency.'}
(out/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='frames'},indent=2),flush=True)
