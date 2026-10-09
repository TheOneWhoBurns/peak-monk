"""Pinned PersonaPlex native inference, with explicit experimental task transfer.

No character role prompt is supplied. A stock voice prefix is used identically
for base and transferred-adapter controls. Releasing unused depth weights requires
a successful real trace-parity report for this exact model.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import mlx.core as mx
from mlx.utils import tree_flatten
from moshi_training import attach_lora
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'vendor/personaplex-edge-source/src'))
from personaplex_edge.runtime import NativeRuntime

p=argparse.ArgumentParser()
p.add_argument('input',type=Path);p.add_argument('--output',type=Path,required=True)
p.add_argument('--parity-report',type=Path,required=True)
p.add_argument('--adapter',type=Path);p.add_argument('--adapter-config',type=Path)
p.add_argument('--allow-task-transfer',action='store_true')
p.add_argument('--voice',default='NATM0');p.add_argument('--seed',type=int,default=41)
p.add_argument('--max-frames',type=int)
a=p.parse_args()
if a.output.exists():raise ValueError('Use a fresh output directory')
if (a.adapter is None)!=(a.adapter_config is None):raise ValueError('Adapter and config must be provided together')
if a.adapter and not a.allow_task_transfer:raise ValueError('This cross-base experiment requires --allow-task-transfer')
parity=json.loads(a.parity_report.read_text())
expected='9840e8618cdff1b1251341c5970ea25c5288c794e470fc1f9087aba15967771f'
if parity.get('status')!='pass' or parity.get('model_sha256')!=expected or parity.get('retained_depth_slices')!=8:
    raise ValueError('Exact-model real trace parity has not passed')
codes=np.load(a.input)['user_codes'];original_frames=codes.shape[1]
if a.max_frames is not None:
    if a.max_frames<3:raise ValueError('At least three frames required')
    codes=codes[:,:a.max_frames]
assert codes.ndim==2 and codes.shape[0]==8 and np.all((codes>=0)&(codes<2048))
mx.set_memory_limit(5_700_000_000);mx.set_cache_limit(32_000_000)
mx.set_wired_limit(mx.metal.device_info()['max_recommended_working_set_size'])
start=time.monotonic();runtime=NativeRuntime('models/personaplex-edge',verify=True)
assert runtime.artifact.manifest['files']['model.q4.safetensors']['sha256']==expected
runtime.generator.reset();runtime.model.depformer.slices=runtime.model.depformer.slices[:8]
config=None
if a.adapter:
    config=json.loads(a.adapter_config.read_text())
    if config['model_sha256']!='7959d590e23c1ebc78cfa3501344a6ff331561aa0cadc4429b733b890bbc919c':
        raise ValueError('Unexpected adapter origin')
    attach_lora(runtime.model,layers=config['layers'],rank=config['rank'],scale=config['scale'])
    weights=mx.load(str(a.adapter));parameters=dict(tree_flatten(runtime.model.trainable_parameters()))
    if set(weights)!=set(parameters) or any(weights[k].shape!=parameters[k].shape for k in weights):
        raise ValueError('Transferred adapter does not cover the exact expected trainable tensors')
    runtime.model.load_weights(list(weights.items()),strict=False);del weights,parameters
runtime.model.eval();mx.eval(runtime.model.parameters());gc.collect();mx.clear_cache()
load_seconds=time.monotonic()-start
runtime.generator.reset();mx.random.seed(a.seed)
start=time.monotonic();runtime.generator.prefill(runtime.artifact.voice_path(a.voice),runtime.text_tokenizer,'')
prefill_seconds=time.monotonic()-start
text=[];audio=[];frames=[];start=time.monotonic()
for i in range(codes.shape[1]):
    tick=time.monotonic();result=runtime.generator.step(mx.array(codes[:,i][None,:]))
    token=None;piece=''
    if result is not None:
        token=result.text_token
        frame=np.asarray(result.audio_tokens).astype(np.uint32)[0]
        if np.any(frame>=2048):raise ValueError('Unfilled audio frame after voice prefill')
        audio.append(frame)
        if token not in (0,3) and 0<=token<runtime.text_tokenizer.get_piece_size():
            piece=runtime.text_tokenizer.id_to_piece(token).replace('▁',' ')
        text.append(piece)
    frames.append(dict(input_frame=i,text_token=token,piece=piece,seconds=time.monotonic()-tick,
                       agent_frame_available=result is not None,
                       agent_audio_frame_index=len(audio)-1 if result is not None else None))
    if (i+1)%50==0:print(i+1,''.join(text)[-140:],flush=True)
elapsed=time.monotonic()-start
if not audio:raise ValueError('No generated frames')
a.output.mkdir(parents=True)
np.savez_compressed(a.output/'agent_codes.npz',agent_codes=np.stack(audio,axis=1))
(a.output/'transcript.txt').write_text(''.join(text).strip()+'\n')
r=dict(model='models/personaplex-edge',model_sha256=expected,
       adapter=str(a.adapter) if a.adapter else None,adapter_sha256=hashlib.sha256(a.adapter.read_bytes()).hexdigest() if a.adapter else None,
       adapter_origin_model_sha256=config['model_sha256'] if config else None,
       transfer_scope='Explicit cross-base task-delta hypothesis; not a checkpoint trained directly on PersonaPlex' if a.adapter else 'Pretrained PersonaPlex control',
       input=str(a.input),input_sha256=hashlib.sha256(a.input.read_bytes()).hexdigest(),input_seconds=codes.shape[1]*.08,
       original_input_frames=original_frames,explicit_frame_limit=a.max_frames,
       voice=a.voice,role_prompt='',seed=a.seed,source_generator='Pinned personaplex-edge generator',
       sampling=runtime.config['lm_gen_config'],parity_report=str(a.parity_report),
       model_load_seconds=load_seconds,voice_and_empty_prompt_prefill_seconds=prefill_seconds,
       generation_seconds=elapsed,generation_realtime_factor=elapsed/(codes.shape[1]*.08),
       peak_model_memory_bytes=mx.get_peak_memory(),frames=frames,
       scope='Native model generation; separate codec decode and ASR required. No live end-to-end latency claim.')
(a.output/'metrics.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps({k:v for k,v in r.items() if k!='frames'},indent=2),flush=True)
