"""Actual all-layer adapter diagnostics, with isolated spoken-text and codebook scores."""
import argparse,json
from pathlib import Path
from bounded_speech_runtime import BoundedSpeechRuntime
from bounded_text_runtime import atomic_json
from sample_dialogue_paths import sha
p=argparse.ArgumentParser();p.add_argument('--adapter',default='adapters/speech-run07/candidate.safetensors');p.add_argument('--output',type=Path,required=True);p.add_argument('--episodes',nargs='+',default=['name','teacher','identity_transparency']);p.add_argument('--include-valid',action='store_true');a=p.parse_args()
if a.output.exists():raise ValueError('Preserve prior audit')
a.output.mkdir()
rows=[]
for path in sorted(Path('data/speech_codes').glob('*.json')):
 meta=json.loads(path.read_text())
 if meta['id'] in a.episodes or (a.include_valid and meta['split']=='valid'):
  source=path.with_suffix('.npz');assert sha(source)==meta['file_sha256'];rows.append((source,meta))
assert set(a.episodes)<=set(m['id'] for _,m in rows)
report=dict(adapter=a.adapter,adapter_sha256=sha(a.adapter) if a.adapter!='base' else None,results=[],scope='Teacher-forced full actual temporal path. Every previous agent token is ground truth; not free-running conversation or audio quality.')
atomic_json(a.output/'report.json',report)
with BoundedSpeechRuntime(a.output,memory_limit=2_800_000_000,max_seconds=1200) as runtime:
 import numpy as np
 import mlx.core as mx
 from moshi_mlx.models import config_v0_1
 from moshi_weight_stream import MoshiParts
 from moshi_split import delay_streams
 from moshi_disk_metrics import disk_metrics
 parts=MoshiParts('models/moshi/model.q4.safetensors',config_v0_1(),None if a.adapter=='base' else a.adapter)
 def boundary(phase,**kw):
  metrics=runtime.response_done()
  if phase=='diagnostic_complete':runtime.event(phase,**metrics)
 for path,meta in rows:
  runtime.event('episode_begin',episode=meta['id']);streams=delay_streams(mx.array(np.load(path)['codes'][None]),parts.cfg)
  result=disk_metrics(parts,streams,boundary=boundary)
  record=dict(episode=meta['id'],split=meta['split'],input_sha256=meta['file_sha256'],**result)
  report['results'].append(record);atomic_json(a.output/'report.json',report)
  print(json.dumps(record),flush=True)
 report.update(status='complete',metrics=runtime.response_done());atomic_json(a.output/'report.json',report)
