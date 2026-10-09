"""Check frozen SFT factors, fresh gradients and standard-adapter export math."""
import json
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.tuner.lora import LoRALinear, LoRAEmbedding
from character_delta import CharacterDelta

mx.set_default_device(mx.cpu);mx.random.seed(61)
checks=[]
for kind in ['linear','embedding','tied_head']:
    if kind=='linear':
        base=LoRALinear.from_base(nn.Linear(32,64,bias=False),r=3,scale=2,dropout=.15)
        x=mx.random.normal((2,7,32));call=lambda m:m(x)
    else:
        base=LoRAEmbedding.from_base(nn.Embedding(64,32),r=3,scale=2,dropout=.15)
        if kind=='embedding': x=mx.array([[1,2,3],[4,5,6]]);call=lambda m:m(x)
        else: x=mx.random.normal((2,7,32));call=lambda m:m.as_linear(x)
    base.lora_b=mx.random.normal(base.lora_b.shape)*.04;base.eval()
    reference=call(base);mx.eval(reference)
    old_a,old_b=base.lora_a,base.lora_b
    delta=CharacterDelta(base,rank=2,dropout=.1);delta.eval()
    zero_error=float(mx.max(mx.abs(call(delta)-reference)).item());assert zero_error==0
    delta.lora_b=mx.random.normal(delta.lora_b.shape)*.03
    leaves=dict(tree_flatten(delta.trainable_parameters()));assert set(leaves)=={'lora_a','lora_b'}
    value,grad=nn.value_and_grad(delta,lambda m:mx.mean(call(m)**2))(delta);mx.eval(value,grad)
    assert all(bool(mx.all(mx.isfinite(v)).item()) for _,v in tree_flatten(grad))
    assert any(float(mx.max(mx.abs(v)).item())>0 for _,v in tree_flatten(grad))
    # Simulate an update and prove the parent factors are untouched.
    delta.lora_a=delta.lora_a-.001*grad['lora_a'];delta.lora_b=delta.lora_b-.001*grad['lora_b']
    assert bool(mx.array_equal(old_a,delta.base.lora_a).item()) and bool(mx.array_equal(old_b,delta.base.lora_b).item())
    a,b=delta.exported_factors()
    if kind=='linear': merged=LoRALinear.from_base(base.linear,r=5,scale=2,dropout=0)
    else: merged=LoRAEmbedding.from_base(base.embedding,r=5,scale=2,dropout=0)
    merged.lora_a=a;merged.lora_b=b;merged.eval()
    export_error=float(mx.max(mx.abs(call(delta)-call(merged))).item());assert export_error<1e-6
    checks.append(dict(kind=kind,zero_delta_error=zero_error,export_error=export_error,frozen_parent='unchanged',trainable_keys=sorted(leaves)))
report=dict(status='pass',checks=checks,scope='Small FP32 adapter algebra and gradient checks only. No trained character quality or BF16 export equivalence claim.')
Path('results/character-delta-check.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
