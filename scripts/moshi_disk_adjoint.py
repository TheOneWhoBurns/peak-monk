"""Full-episode temporal gradients with one frozen model part resident at a time."""
import gc
from pathlib import Path
import mlx.core as mx
import mlx.optimizers as optim
from mlx.utils import tree_flatten,tree_unflatten
from moshi_training import layer_forward
from moshi_staged import staged_head_grad
from moshi_layerwise import temporal_layer_vjp,norm_vjp


def disk_episode_grad(parts,streams,scratch,head_frames=8,padding_weight=.05,boundary=lambda **kw:None):
    scratch=Path(scratch);scratch.mkdir(exist_ok=False)
    def save_hidden(index,h):
        mx.eval(h);mx.save_safetensors(str(scratch/f'hidden-{index:02d}.safetensors'),{'hidden':h})
    def release(phase,**kw):
        gc.collect();boundary(phase=phase,**kw)
    inputs=parts.load('inputs');h=mx.stop_gradient(inputs(streams[:,:,:-1]));save_hidden(0,h)
    del inputs;release('input_embeddings_complete')
    for i in range(parts.cfg.transformer.num_layers):
        layer=parts.load(i);h=mx.stop_gradient(layer_forward(layer,h)[0]);save_hidden(i+1,h)
        del layer;release('temporal_forward_complete',layer=i)
    heads=parts.load('heads');normalized=mx.stop_gradient(heads.out_norm(h));mx.eval(normalized)
    value,grad,cotangent=staged_head_grad(heads,normalized,streams[:,:,1:],head_frames,padding_weight,boundary=boundary)
    cotangent=norm_vjp(heads.out_norm,h,cotangent)
    value=float(value.item());gradients=[];squared_norm=0.
    def save_gradient(part,gradient):
        nonlocal squared_norm
        flat=dict(tree_flatten(gradient));mx.eval(flat)
        if not all(mx.all(mx.isfinite(v)).item() for v in flat.values()):raise ValueError('Nonfinite gradient')
        squared_norm+=sum(float(mx.sum(v.astype(mx.float32)**2).item()) for v in flat.values())
        path=scratch/f'gradient-{part}.safetensors';mx.save_safetensors(str(path),flat)
        gradients.append(dict(part=part,path=str(path)))
    save_gradient('heads',grad)
    del heads,normalized,h,grad;release('heads_gradient_complete',loss=value)
    for i in reversed(range(parts.cfg.transformer.num_layers)):
        h=mx.load(str(scratch/f'hidden-{i:02d}.safetensors'))['hidden']
        layer=parts.load(i);grad,cotangent=temporal_layer_vjp(layer,h,cotangent)
        save_gradient(i,grad)
        del layer,h,grad;release('temporal_backward_complete',layer=i)
    del cotangent;release('episode_gradient_complete')
    return dict(loss=value,gradient_norm=squared_norm**.5,gradients=gradients)


def disk_adamw_update(parts,gradient_result,state_dir,output,lr=1e-4,weight_decay=.01,clip_norm=1.,expected_step=1):
    """All gradients already complete; one common scale and equal block steps."""
    state_dir=Path(state_dir);state_dir.mkdir(exist_ok=True)
    if Path(output).exists():raise ValueError('Preserve prior complete export')
    scale=min(1.,clip_norm/(gradient_result['gradient_norm']+1e-8))
    exported={}
    for record in gradient_result['gradients']:
        part=record['part'];prefix='text_linear.' if part=='heads' else f'transformer.layers.{part}.'
        params=parts.adapter.load([prefix],strip=prefix)
        grads=mx.load(record['path'])
        if set(params)!=set(grads):raise ValueError('Gradient mapping mismatch')
        grads=tree_unflatten([(k,v*scale) for k,v in grads.items()])
        params=tree_unflatten(list(params.items()))
        optimizer=optim.AdamW(learning_rate=lr,weight_decay=weight_decay)
        state_path=state_dir/f'{part}.safetensors'
        if state_path.exists():optimizer.state=tree_unflatten(list(mx.load(str(state_path)).items()))
        updated=optimizer.apply_gradients(grads,params);mx.eval(updated,optimizer.state)
        if int(optimizer.state['step'].item())!=expected_step:raise ValueError('Optimizer step mismatch')
        if not all(mx.all(mx.isfinite(v)).item() for _,v in tree_flatten(updated)):raise ValueError('Nonfinite update')
        temporary=state_path.with_suffix('.tmp.safetensors')
        mx.save_safetensors(str(temporary),dict(tree_flatten(optimizer.state)));temporary.replace(state_path)
        exported.update({prefix+k:v for k,v in tree_flatten(updated)})
        del optimizer,grads,params,updated
    expected={k for k in parts.adapter.header if k!='__metadata__'}
    if set(exported)!=expected:raise ValueError('Incomplete adapter export')
    temporary=Path(str(output)+'.tmp.safetensors');mx.save_safetensors(str(temporary),exported);temporary.replace(output)
    return dict(clip_scale=scale,adapter_tensors=len(exported),step=expected_step)
