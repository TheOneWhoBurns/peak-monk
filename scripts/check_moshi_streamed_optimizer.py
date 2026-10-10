"""Two-step global-clipped AdamW parity with per-block state saved and reloaded."""
import argparse,json,tempfile
from pathlib import Path
import mlx.core as mx
import mlx.optimizers as optim
from mlx.utils import tree_flatten,tree_unflatten
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
mx.set_default_device(mx.cpu);mx.random.seed(67)
initial={f'block{i}':{'a':mx.random.normal((7,3)),'b':mx.random.normal((3,9))} for i in range(3)}
full=initial;streamed=initial
optimizer=optim.AdamW(learning_rate=1e-4,weight_decay=.01)
reports=[]
with tempfile.TemporaryDirectory() as tmp:
 for step in range(2):
  grads={k:{n:mx.random.normal(v.shape) for n,v in params.items()} for k,params in initial.items()}
  norm=mx.sqrt(sum(mx.sum(g*g) for _,g in tree_flatten(grads)));scale=mx.minimum(1.,.3/(norm+1e-6))
  clipped={k:{n:g*scale for n,g in gs.items()} for k,gs in grads.items()}
  full=optimizer.apply_gradients(clipped,full);mx.eval(full,optimizer.state)
  updated={};states={}
  for name,params in streamed.items():
   local=optim.AdamW(learning_rate=1e-4,weight_decay=.01)
   path=Path(tmp)/(name+'.safetensors')
   if step:local.state=tree_unflatten(list(mx.load(str(path)).items()))
   updated[name]=local.apply_gradients(clipped[name],params);mx.eval(updated[name],local.state)
   mx.save_safetensors(str(path),dict(tree_flatten(local.state)))
   states[name]=tree_unflatten(list(mx.load(str(path)).items()))
  streamed=updated
  error=max(mx.max(mx.abs(v-dict(tree_flatten(streamed))[k])).item() for k,v in tree_flatten(full))
  state_error=0.
  for name,state in states.items():
   assert int(state['step'].item())==step+1
   for k,v in tree_flatten(state):
    if k in ('step','learning_rate'): expected=optimizer.state[k]
    else:expected=dict(tree_flatten(optimizer.state[name]))[k]
    state_error=max(state_error,mx.max(mx.abs(v-expected)).item())
  assert error<1e-7 and state_error<1e-7
  reports.append(dict(step=step+1,gradient_norm=norm.item(),clip_scale=scale.item(),parameter_error=error,state_error=state_error))
r=dict(status='pass',steps=reports,scope='Actual pinned MLX AdamW: one common global clipping scale, separate block optimizers with serialized state and equal global steps. Synthetic gradients; does not verify pretrained gradients or memory.')
a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
