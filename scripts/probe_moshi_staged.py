"""Disposable actual-model full-episode gradient and optimizer memory probe."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map
from moshi_mlx.models import Lm, config_v0_1
from moshi_training import attach_lora, loss
from moshi_split import delay_streams
from moshi_staged import staged_episode_grad
from moshi_windowed import offload_optimizer, restore_optimizer

p = argparse.ArgumentParser()
p.add_argument('--episodes', nargs='+', default=['name', 'anger'])
p.add_argument('--head-frames', type=int, default=16)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
if a.output.exists():
    raise ValueError('Preserve previous probe evidence')
weight = Path('models/moshi/model.q4.safetensors')
with weight.open('rb') as f:
    weight_hash = hashlib.file_digest(f, 'sha256').hexdigest()
assert weight_hash == '7959d590e23c1ebc78cfa3501344a6ff331561aa0cadc4429b733b890bbc919c'
mx.set_memory_limit(5_600_000_000)
mx.set_cache_limit(32_000_000)
mx.set_wired_limit(mx.metal.device_info()['max_recommended_working_set_size'])
mx.random.seed(41)
model = Lm(config_v0_1())
model.set_dtype(mx.bfloat16)
nn.quantize(model, bits=4, group_size=32)
model.load_weights(str(weight), strict=True)
attach_lora(model, layers=32, rank=8, scale=2.)
model.train()
mx.eval(model.parameters())
optimizer = optim.AdamW(learning_rate=1e-4, weight_decay=.01)
records = []
start = time.monotonic()
with tempfile.TemporaryDirectory(prefix='jun-staged-optimizer-') as temporary:
    optimizer_file = Path(temporary) / 'state.safetensors'
    for ident in a.episodes:
        path = Path('data/speech_codes') / (ident + '.npz')
        meta = json.loads(path.with_suffix('.json').read_text())
        assert meta['split'] == 'train'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == meta['file_sha256']
        streams = delay_streams(mx.array(np.load(path)['codes'][None]), model.cfg)
        tick = time.monotonic()
        print('Staged full-episode update', ident, meta['frames'], flush=True)
        value, gradient = staged_episode_grad(model, streams, head_frames=a.head_frames)
        norm = mx.sqrt(sum(mx.sum(g.astype(mx.float32)**2) for _, g in tree_flatten(gradient)))
        mx.eval(value, norm)
        assert mx.isfinite(value).item() and mx.isfinite(norm).item()
        gradient = tree_map(lambda g: g * mx.minimum(1., 1./(norm+1e-8)), gradient)
        if optimizer_file.exists():
            restore_optimizer(optimizer, optimizer_file)
        optimizer.update(model, gradient)
        mx.eval(model.parameters(), optimizer.state)
        assert all(mx.all(mx.isfinite(v)).item() for _, v in tree_flatten(model.trainable_parameters()))
        records.append(dict(episode=ident, input_sha256=meta['file_sha256'], frames=meta['frames'],
            loss=value.item(), gradient_norm=norm.item(), seconds=time.monotonic()-tick,
            peak_memory_bytes=mx.get_peak_memory()))
        print(json.dumps(records[-1]), flush=True)
        del streams, value, gradient, norm
        mx.clear_cache()
        offload_optimizer(optimizer, optimizer_file)
report = dict(status='pass', model_sha256=weight_hash, head_frames=a.head_frames,
    seconds=time.monotonic()-start, peak_memory_bytes=mx.get_peak_memory(), updates=records,
    scope='Actual pretrained quantized Moshi, full temporal gradients, staged joint text/audio heads, finite disposable optimizer updates; no adapter saved or quality claim.')
a.output.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
