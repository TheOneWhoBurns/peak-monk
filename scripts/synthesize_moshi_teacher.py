"""Experimental native-voice training synthesis with forced authored text.

This is teacher data generation, NEVER evidence of learned character behavior.
Only training episodes are allowed by this pilot. No model adapter is loaded.
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

class ForcedText:
    def __init__(self, tokens):
        self.tokens=tokens;self.index=0
    def __call__(self, logits):
        if self.index>=len(self.tokens):raise ValueError("Forced text exhausted")
        token=mx.full(logits.shape[:-1],int(self.tokens[self.index]),dtype=mx.int32)
        self.index+=1
        return token, None

p=argparse.ArgumentParser()
p.add_argument('--episodes',nargs='+',required=True)
p.add_argument('--output',required=True)
p.add_argument('--seed',type=int,default=41)
args=p.parse_args()
root=Path(args.output)
if root.exists():raise ValueError('Use a fresh synthesis directory')
inputs=[]
for name in args.episodes:
    if Path(name).name!=name:raise ValueError('Simple episode IDs only')
    path=Path('data/speech_codes')/(name+'.npz')
    meta=json.loads(path.with_suffix('.json').read_text())
    assert meta['split']=='train','Pilot cannot consume held-out episodes'
    raw=np.load(path)['codes']
    assert raw.shape[0]==17 and np.all((raw[0]>=0)&(raw[0]<32000))
    inputs.append((name,path,meta,raw))
weight=Path('models/moshi/model.q4.safetensors')
with weight.open('rb') as f:
    sha=hashlib.file_digest(f,'sha256').hexdigest()
assert sha=='7959d590e23c1ebc78cfa3501344a6ff331561aa0cadc4429b733b890bbc919c'
mx.set_memory_limit(5_700_000_000);mx.set_cache_limit(64_000_000)
mx.set_wired_limit(mx.metal.device_info()['max_recommended_working_set_size'])
model=Lm(config_v0_1());model.set_dtype(mx.bfloat16)
nn.quantize(model,bits=4,group_size=32);model.load_weights(str(weight),strict=True)
mx.eval(model.parameters());model.eval()
tok=spm.SentencePieceProcessor(model_file='models/moshi/tokenizer_spm_32k_3.model')
root.mkdir(parents=True)
for name,path,meta,raw in inputs:
    dest=root/name;dest.mkdir()
    model.warmup()  # Also resets temporal caches between independent episodes.
    for cache in model.depformer_cache:cache.reset()
    mx.random.seed(args.seed)
    delay=max(model.cfg.audio_delays)
    # These source timelines end in explicit silence. Repeat the final user
    # frame only to flush delayed agent codes, then retain exactly T frames.
    users=np.pad(raw[9:],((0,0),(0,delay)),mode='edge')
    forced=np.pad(raw[0],(0,delay),constant_values=3)
    sampler=ForcedText(forced)
    gen=LmGen(model,max_steps=len(forced)+5,text_sampler=sampler,audio_sampler=Sampler())
    audio=[];start=time.monotonic()
    for i in range(len(forced)):
        result=gen.step(mx.array(users[:,i][None,:]))
        token=result[0] if isinstance(result,tuple) else result
        assert int(token.item())==int(forced[i])
        codes=gen.last_audio_tokens()
        if codes is not None:audio.append(np.array(codes)[0])
    native=np.stack(audio,axis=1).astype(np.uint32)
    assert native.shape==(8,raw.shape[1]),native.shape
    np.savez_compressed(dest/'agent_codes.npz',agent_codes=native)
    combined=np.concatenate([raw[:1],native.astype(np.int32),raw[9:]],axis=0)
    np.savez_compressed(dest/'teacher_codes.npz',codes=combined)
    text=''.join(tok.id_to_piece(int(t)).replace('▁',' ') for t in raw[0] if t not in (0,3)).strip()
    (dest/'transcript.txt').write_text(text+'\n')
    report=dict(mode='FORCED TEXT SYNTHETIC TEACHER; NOT LEARNED CHARACTER INFERENCE',episode=name,
        source_codes_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),model_sha256=sha,seed=args.seed,
        forced_text=True,unmodified_user_codes=True,source_split=meta['split'],
        frames=raw.shape[1],seconds=time.monotonic()-start,peak_memory_bytes=mx.get_peak_memory(),
        flush_frames=delay,admission='pending unprompted decoded-audio ASR and author review',
        inherited_sources=meta['sources'])
    (dest/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    print(name,text,flush=True)
