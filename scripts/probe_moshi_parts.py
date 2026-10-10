"""Measure selective actual-Moshi loading without a full backbone allocation."""
import argparse,gc,json,time
from pathlib import Path
from bounded_speech_runtime import BoundedSpeechRuntime
from bounded_text_runtime import atomic_json
from sample_dialogue_paths import sha
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.output.exists():raise ValueError('Preserve previous probe')
a.output.mkdir()
weight=Path('models/moshi/model.q4.safetensors');adapter=Path('adapters/speech-run07/candidate.safetensors')
assert sha(weight)=='7959d590e23c1ebc78cfa3501344a6ff331561aa0cadc4429b733b890bbc919c'
assert sha(adapter)=='2e2b8363b1c7350ccc55ca97c238ea054a6ae2fa573da92c7c401e0f9083156c'
with BoundedSpeechRuntime(a.output,memory_limit=2_800_000_000,max_seconds=180) as runtime:
 import mlx.core as mx
 from mlx.utils import tree_flatten
 from moshi_mlx.models import config_v0_1
 from moshi_weight_stream import MoshiParts
 parts=MoshiParts(weight,config_v0_1(),adapter)
 records=[]
 for part in ['inputs',0,'heads']:
  runtime.event('load_begin',part=part);tick=time.monotonic();module=parts.load(part)
  record=dict(part=part,load_seconds=time.monotonic()-tick,parameter_bytes=sum(v.nbytes for _,v in tree_flatten(module.parameters())),metrics=runtime.response_done())
  records.append(record);runtime.event('load_complete',**record);print(json.dumps(record),flush=True)
  del module;gc.collect();runtime.response_done()
 atomic_json(a.output/'result.json',dict(status='pass',parts=records,scope='Actual pretrained selective-part loads only; no full backbone resident and no gradient/quality evidence.'))
