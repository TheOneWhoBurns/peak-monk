"""Precompute the frozen temporal prefix, then exit to release its weights."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import mlx.core as mx
from moshi_mlx.models.lm import config_v0_1
from moshi_split import load_part, delay_streams

p=argparse.ArgumentParser()
p.add_argument('--suffix-layers',type=int,default=4)
p.add_argument('--max-frames',type=int,default=1500)
args=p.parse_args()
weight=Path('models/moshi/model.q4.safetensors')
with weight.open('rb') as f:
    assert hashlib.file_digest(f,'sha256').hexdigest()=='7959d590e23c1ebc78cfa3501344a6ff331561aa0cadc4429b733b890bbc919c'
mx.metal.set_memory_limit(5_500_000_000)
mx.metal.set_cache_limit(64_000_000)
mx.metal.set_wired_limit(mx.metal.device_info()['max_recommended_working_set_size'])
prefix=load_part(weight,args.suffix_layers,'prefix')
out=Path(f'data/speech_cache/suffix-{args.suffix_layers}');out.mkdir(parents=True,exist_ok=True)
rows=[]
for path in sorted(Path('data/speech_codes').glob('*.npz')):
    info=json.loads(path.with_suffix('.json').read_text())
    if info['frames']>args.max_frames:
        raise ValueError(f'{path}: longer than predeclared context limit; do not silently truncate')
    target=out/(path.stem+'.safetensors')
    meta=out/(path.stem+'.json')
    code_sha=hashlib.sha256(path.read_bytes()).hexdigest()
    if target.exists() and meta.exists():
        cached=json.loads(meta.read_text())
        if cached.get('codes_sha256')!=code_sha or cached.get('suffix_layers')!=args.suffix_layers:
            raise ValueError(f'{target}: stale prefix cache; use a fresh cache directory')
        rows.append(cached);continue
    start=time.monotonic()
    raw=mx.array(np.load(path)['codes'][None,:,:])
    streams=delay_streams(raw,config_v0_1())
    hidden=prefix(streams[:,:,:-1],materialize=True)
    mx.eval(hidden)
    mx.save_safetensors(str(target),{'hidden':hidden,'targets':streams[:,:,1:]})
    row=dict(info,suffix_layers=args.suffix_layers,path=str(target),
             codes_sha256=code_sha,
             prefix_seconds=time.monotonic()-start,peak_memory_bytes=mx.metal.get_peak_memory(),
             model_sha256='7959d590e23c1ebc78cfa3501344a6ff331561aa0cadc4429b733b890bbc919c',
             cached_prefix='Frozen embeddings and first '+str(32-args.suffix_layers)+' temporal layers')
    meta.write_text(json.dumps(row,indent=2)+'\n');rows.append(row)
    print(path.stem,round(row['prefix_seconds'],1),row['peak_memory_bytes'],flush=True)
    del hidden,streams,raw
    mx.metal.clear_cache()
(out/'manifest.json').write_text(json.dumps(rows,indent=2)+'\n')
