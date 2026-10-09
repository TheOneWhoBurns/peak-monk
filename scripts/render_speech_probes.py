"""Render fixed user-side timelines for native speech evaluations.

These synthetic user recordings are evaluation inputs, never training data.
Actual PCM placement is recorded; no timing success is inferred from the script.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import onnxruntime as ort
import soundfile as sf
from kokoro_onnx import Kokoro
import rustymimi

p=argparse.ArgumentParser()
p.add_argument('--suite',default='evaluation/speech-roleplay-v2.json')
p.add_argument('--output',default='evaluation/audio-roleplay-v2')
args=p.parse_args()
out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
model_file=Path('models/kokoro-hf/onnx/model_quantized.onnx')
assert hashlib.sha256(model_file.read_bytes()).hexdigest()=='fbae9257e1e05ffc727e951ef9b9c98418e6d79f1c9b6b13bd59f5c9028a1478'
options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
session=ort.InferenceSession(str(model_file),sess_options=options,providers=['CPUExecutionProvider'])
tts=Kokoro.from_session(session,'models/kokoro-hf/voices.npz')
weight=Path('models/moshi/tokenizer-e351c8d8-checkpoint125.safetensors')
with weight.open('rb') as f:
 assert hashlib.file_digest(f,'sha256').hexdigest()=='09b782f0629851a271227fb9d36db65c041790365f11bbe5d3d59369cf863f50'
for case in json.loads(Path(args.suite).read_text()):
 dest=out/(case['id']+'.json')
 if dest.exists():continue
 audio=np.zeros(round(case['duration_seconds']*24000),dtype=np.float32);placements=[]
 for i,turn in enumerate(case['user_turns']):
  pcm,sr=tts.create(turn['text'],voice='am_michael',speed=1.,lang='en-us')
  assert sr==24000 and np.isfinite(pcm).all()
  start=round(turn['at_seconds']*sr);end=start+len(pcm)
  assert end<=len(audio) and not np.any(audio[start:end]),'Invalid probe timeline'
  audio[start:end]=pcm
  placements.append(dict(text=turn['text'],start_sample=start,end_sample=end,start_seconds=start/sr,end_seconds=end/sr))
 audio=np.clip(audio,-.95,.95);wav=out/(case['id']+'.wav');sf.write(wav,audio,24000,subtype='PCM_16')
 # Encode the saved waveform so the file hash identifies the exact codec input.
 audio,_=sf.read(wav,dtype='float32');audio=np.pad(audio,(0,(-len(audio))%1920))
 mimi=rustymimi.Tokenizer(str(weight),num_codebooks=8)
 encoded=[];tick=time.monotonic()
 for i in range(0,len(audio),1920):
  chunk=mimi.encode_step(audio[i:i+1920][None,None,:])
  if chunk is not None:encoded.append(chunk)
 codes=np.concatenate(encoded,axis=-1)[0];assert codes.shape==(8,len(audio)//1920)
 del mimi
 path=out/(case['id']+'.npz');np.savez_compressed(path,user_codes=codes)
 result=dict(case=case,voice='am_michael',voice_seen_in_training=True,synthetic_user=True,codec_state='fresh instance per probe',sample_rate=24000,placements=placements,codec_seconds=time.monotonic()-tick,suite_sha256=hashlib.sha256(Path(args.suite).read_bytes()).hexdigest(),wav_sha256=hashlib.sha256(wav.read_bytes()).hexdigest(),codes_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),use='Held-out evaluation input, excluded from speech_codes and prefix training cache')
 dest.write_text(json.dumps(result,indent=2)+'\n');print(case['id'],codes.shape,flush=True)
