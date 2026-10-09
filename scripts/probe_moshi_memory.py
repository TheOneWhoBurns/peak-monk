"""Bounded memory/gradient probe on actual frozen quantized Moshi weights.

Random tokens measure implementation feasibility only; this is NOT a character
or speech training run. Run separately from all other model processes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from moshi_mlx.models.lm import Lm, config_v0_1
from moshi_training import attach_lora, loss

p = argparse.ArgumentParser()
p.add_argument('--frames', type=int, default=16)
p.add_argument('--layers', type=int, default=2)
p.add_argument('--rank', type=int, default=4)
p.add_argument('--output', default='results/moshi-memory-probe.json')
args = p.parse_args()
weight = Path('models/moshi/model.q4.safetensors')
with weight.open('rb') as f:
    assert hashlib.file_digest(f, 'sha256').hexdigest() == '7959d590e23c1ebc78cfa3501344a6ff331561aa0cadc4429b733b890bbc919c', 'Incomplete or incorrect weights'
metal = mx.metal
metal.set_memory_limit(6_000_000_000)
metal.set_cache_limit(128_000_000)
start = time.monotonic()
model = Lm(config_v0_1())
model.set_dtype(mx.bfloat16)
nn.quantize(model, bits=4, group_size=32)
model.load_weights(str(weight), strict=True)
mx.eval(model.parameters())
loaded = time.monotonic() - start
loaded_memory = metal.get_active_memory()
attach_lora(model, layers=args.layers, rank=args.rank)
mx.eval(model.parameters())
params = sum(x.size for _, x in tree_flatten(model.trainable_parameters()))
mx.random.seed(41)
tokens = mx.random.randint(0, 2048, (1, 17, args.frames + 1))
optimizer = optim.Adam(learning_rate=1e-5)
grad_fn = nn.value_and_grad(model, loss)
steps = []
for i in range(2):
    start = time.monotonic()
    value, grad = grad_fn(model, tokens)
    optimizer.update(model, grad)
    mx.eval(value, model.parameters(), optimizer.state)
    steps.append({'loss': value.item(), 'seconds': time.monotonic() - start,
                  'peak_bytes': metal.get_peak_memory(), 'active_bytes': metal.get_active_memory()})
    print(json.dumps(steps[-1]), flush=True)
report = {'scope':'Random-code feasibility probe, NOT speech/character training',
          'frames': args.frames, 'layers': args.layers, 'rank': args.rank,
          'trainable_parameters': params, 'load_seconds': loaded,
          'loaded_active_bytes': loaded_memory, 'steps': steps}
Path(args.output).write_text(json.dumps(report, indent=2)+'\n')
