"""Verify a standard export using hashed frozen inputs and unchanged prefix factors."""
import argparse,gc,hashlib,json
from pathlib import Path
from bounded_text_runtime import BoundedTextRuntime,atomic_json
from sample_dialogue_paths import sha
p=argparse.ArgumentParser()
p.add_argument('--run',type=Path,required=True)
p.add_argument('--adapter',type=Path,required=True)
p.add_argument('--reference',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
if a.output.exists(): raise ValueError('Preserve prior results')
a.output.mkdir()
ref=json.loads(a.reference.read_text()); meta=json.loads((a.run/'metadata.json').read_text())
manifest=json.loads((a.run/'frozen_prefix_manifest.json').read_text())
entry=next(e for e in manifest['entries'] if e['split']=='repairs' and e['index']==ref['row_index'])
assert sha(a.adapter/'adapters.safetensors')==ref['adapter_sha256']
assert sha(ref['data'])==ref['data_sha256'] and sha(entry['path'])==entry['sha256']
with BoundedTextRuntime(a.output,memory_limit=3_400_000_000,max_seconds=180) as runtime:
 import mlx.core as mx
 import numpy as np
 from mlx_lm import load
 from conversation_objective import encode_conversation
 from repair_export_score import selected_score
 factors=mx.load(str(a.adapter/'adapters.safetensors'))
 digest=hashlib.sha256(); frozen_count=0
 for name,value in sorted(factors.items()):
  if name not in meta['trainable_names']:
   digest.update(name.encode());digest.update(np.asarray(value).tobytes());frozen_count+=1
 assert digest.hexdigest()==meta['frozen_adapter_sha256']==manifest['frozen_adapter_sha256']
 del factors,value
 model,tok=load(ref['model'],adapter_path=str(a.adapter))
 model.eval();model.model.layers=model.layers[manifest['frozen_layers']:]
 gc.collect();runtime.response_done()
 row=[json.loads(s) for s in Path(ref['data']).read_text().splitlines()][ref['row_index']]
 tokens,mask=encode_conversation(row,tok)
 assert hashlib.sha256(json.dumps(tokens[:-1]).encode()).hexdigest()==entry['input_tokens_sha256']
 hidden=mx.load(entry['path'])['hidden']
 actual=selected_score(model,tokens,mask,input_embeddings=hidden)
 error=max(abs(x-y) for x,y in zip(actual['target_log_probs'],ref['score']['target_log_probs'],strict=True))
 assert actual['target_tokens']==ref['score']['target_tokens']
 report=dict(status='pass' if error==0 else 'fail',maximum_logp_difference=error,
  frozen_adapter_tensors=frozen_count,frozen_adapter_sha256=digest.hexdigest(),
  actual=actual,reference=ref,cached_input=entry,metrics=runtime.response_done(),
  scope='Reloaded standard adapter: every frozen prefix/head factor byte-identical to training; hashed cached input; exact suffix score parity. Does not replace an independent full-backbone forward comparison.')
 atomic_json(a.output/'result.json',report);print(json.dumps(report),flush=True)
 assert error==0
