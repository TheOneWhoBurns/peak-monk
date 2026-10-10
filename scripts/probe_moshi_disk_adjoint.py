"""One disposable real full-episode update, with model parts streamed from disk."""
import argparse,json,math,tempfile,time
from pathlib import Path
from bounded_speech_runtime import BoundedSpeechRuntime
from bounded_text_runtime import atomic_json
from sample_dialogue_paths import sha
p=argparse.ArgumentParser();p.add_argument('--episode',default='anger');p.add_argument('--output',type=Path,required=True);p.add_argument('--adapter-output',type=Path,required=True);a=p.parse_args()
if a.output.exists() or a.adapter_output.exists():raise ValueError('Preserve earlier evidence')
a.output.mkdir();a.adapter_output.mkdir(parents=True)
weight=Path('models/moshi/model.q4.safetensors');parent=Path('adapters/speech-run07/candidate.safetensors')
weight_sha=sha(weight);parent_sha=sha(parent)
assert weight_sha=='7959d590e23c1ebc78cfa3501344a6ff331561aa0cadc4429b733b890bbc919c'
assert parent_sha=='2e2b8363b1c7350ccc55ca97c238ea054a6ae2fa573da92c7c401e0f9083156c'
path=Path('data/speech_codes')/(a.episode+'.npz');meta=json.loads(path.with_suffix('.json').read_text())
assert meta['split']=='train' and sha(path)==meta['file_sha256']
atomic_json(a.output/'config.json',dict(episode=meta,model_sha256=weight_sha,parent_sha256=parent_sha,head_frames=8,
 source_sha256={n:sha(Path(__file__).with_name(n)) for n in ['moshi_disk_adjoint.py','moshi_weight_stream.py','moshi_layerwise.py','moshi_staged.py','bounded_speech_runtime.py','bounded_text_runtime.py','probe_moshi_disk_adjoint.py']},
 scope='Disposable full-episode gradient and one update. No time truncation or quality claim. No wiring or full-model residency.'))
with BoundedSpeechRuntime(a.output,memory_limit=2_800_000_000,max_seconds=1800) as runtime:
 import numpy as np
 import mlx.core as mx
 from moshi_mlx.models import config_v0_1
 from moshi_weight_stream import MoshiParts
 from moshi_split import delay_streams
 from moshi_disk_adjoint import disk_episode_grad,disk_adamw_update
 parts=MoshiParts(weight,config_v0_1(),parent)
 streams=delay_streams(mx.array(np.load(path)['codes'][None]),parts.cfg)
 def boundary(phase,**kw):
  metrics=runtime.response_done();runtime.event(phase,**kw,**metrics)
  if 'layer' not in kw or kw['layer']%8==0:print(json.dumps(dict(phase=phase,**kw,**metrics)),flush=True)
 with tempfile.TemporaryDirectory(prefix='jun-disk-adjoint-') as tmp:
  start=time.monotonic();result=disk_episode_grad(parts,streams,Path(tmp)/'gradient',head_frames=8,boundary=boundary)
  if not math.isfinite(result['loss']) or not math.isfinite(result['gradient_norm']) or result['gradient_norm']<=0:raise ValueError('Nonfinite or zero gradient')
  output=a.adapter_output/'candidate.safetensors'
  update=disk_adamw_update(parts,result,Path(tmp)/'optimizer',output,clip_norm=1.,expected_step=1)
  boundary('optimizer_update_complete')
  report=dict(status='pass',episode=a.episode,frames=meta['frames'],loss=result['loss'],gradient_norm=result['gradient_norm'],update=update,
   seconds=time.monotonic()-start,metrics=runtime.response_done(),adapter=str(output),adapter_sha256=sha(output),parent_sha256=parent_sha,
   scope='Actual pretrained Moshi plus run07parent, complete episode and all32temporal adapter gradients, one finite disposable update. Gradient math validated separately on tiny models; no native quality evidence.')
  atomic_json(a.output/'result.json',report);atomic_json(a.adapter_output/'config.json',report);print(json.dumps(report),flush=True)
assert sha(parent)==parent_sha
