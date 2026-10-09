"""Full-context local SFT with bounded vocabulary-head memory and RNG replay.

One optimizer step uses the token mean over an entire accumulation group.
Exports are ordinary MLX LoRA adapters; no prompt is embedded in inference.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time

import mlx.core as mx
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map
from mlx_lm import load
from conversation_objective import ConversationDataset
from qwen_rng_checkpoint import qwen_rng_checkpoint
from staged_character_grad import staged_character_grad, restore_rng


def capture_checkpoint(model, tokenizer, cfg, out, step, checkpoint):
    """Audit the same resident policy, restoring the training RNG afterward."""
    from mlx_lm.tuner.datasets import CacheDataset, ChatDataset
    from mlx_lm.tuner.trainer import evaluate
    from assistant_objective import ThroughEOS, assistant_loss
    from evaluate import capture_conversations
    snapshot = out.with_name(out.name + f'-step{step:04d}')
    snapshot.mkdir(exist_ok=False)
    os.link(checkpoint, snapshot / 'adapters.safetensors')
    (snapshot / 'adapter_config.json').write_bytes((out / 'adapter_config.json').read_bytes())
    suite = 'evaluation/dev-roleplay-v2.json'
    target = Path('results') / f'{out.name}-step{step:04d}.json'
    if target.exists():
        raise ValueError('Never overwrite a previous evaluation')
    fingerprint = hashlib.sha256()
    for name in ('config.json', 'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja'):
        path = Path(cfg['model']) / name
        if path.exists():
            fingerprint.update(name.encode())
            fingerprint.update(path.read_bytes())
    rng = list(mx.random.state)
    mx.eval(rng)
    model.eval()
    try:
        rows = [json.loads(line) for line in Path('data/prepared/valid.jsonl').read_text().splitlines()]
        dataset = CacheDataset(ThroughEOS(ChatDataset(rows, tokenizer, mask_prompt=True), tokenizer.eos_token_id))
        loss = evaluate(model, dataset, batch_size=1, num_batches=-1,
                        max_seq_length=768, loss=assistant_loss)
        result = dict(model=cfg['model'], adapter=str(snapshot), adapter_sha256=sha(checkpoint),
            prompted=False, serialization_sha256=fingerprint.hexdigest(), validation_loss=loss,
            loss_version='assistant_eos_v2', training_loss_version='staged_group_token_mean_v1',
            suite=suite, suite_sha256=sha(suite), temperature=0., top_p=1., presence_penalty=0.,
            max_tokens=160, seed=41, quality_scoring='Unscored development capture; explicit review required',
            conversations=[], capture_mode='Same resident policy at completed optimizer step; exact standard adapter exported first')
        target.write_text(json.dumps(result, indent=2) + '\n')
        args = argparse.Namespace(suite=suite, prompted=False, temperature=0., top_p=1.,
                                  presence_penalty=0., max_tokens=160, seed=41)
        capture_conversations(model, tokenizer, args, result, target)
    finally:
        model.train()
        restore_rng(rng)
        mx.clear_cache()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    out = Path(cfg['output'])
    if out.exists():
        raise ValueError('A fresh experiment directory is required')
    parent = Path(cfg['parent'])
    if sha(parent / 'adapters.safetensors') != cfg['parent_sha256']:
        raise ValueError('Parent checkpoint differs from the declared experiment')
    data = Path(cfg['data']) / 'train.jsonl'
    if sha(data) != cfg['train_sha256']:
        raise ValueError('Training data differs from the declared experiment')
    if cfg['epochs'] < 1 or cfg['accumulation'] < 1:
        raise ValueError('Positive epoch and accumulation counts required')
    out.mkdir(parents=True)
    mx.set_memory_limit(5_600_000_000)
    mx.set_cache_limit(32_000_000)
    mx.set_wired_limit(mx.device_info()['max_recommended_working_set_size'])
    mx.random.seed(cfg['seed'])
    model, tok = load(cfg['model'], adapter_path=str(parent))
    adapter_cfg = json.loads((parent / 'adapter_config.json').read_text())
    if Path(adapter_cfg['model']).resolve() != Path(cfg['model']).resolve():
        raise ValueError('Parent adapter belongs to another base model')
    model.freeze()
    for _, module in model.named_modules():
        if 'lora_a' in module and 'lora_b' in module:
            module.unfreeze(recurse=False, keys=['lora_a', 'lora_b'], strict=True)
    leaves = tree_flatten(model.trainable_parameters())
    if not leaves or not all(k.endswith(('.lora_a', '.lora_b')) for k, _ in leaves):
        raise ValueError('Only declared adapter factors may be trained')
    qwen_rng_checkpoint(model.layers[0])
    model.train()
    mx.eval(model.parameters())
    rows = [json.loads(line) for line in data.read_text().splitlines()]
    dataset = ConversationDataset(rows, tok, cfg['max_length'])
    steps = math.ceil(len(dataset) / cfg['accumulation']) * cfg['epochs']
    warmup = max(1, round(steps * cfg['warmup_fraction']))
    metadata = dict(config=cfg, config_sha256=sha(args.config),
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        source_sha256={f: sha(Path(__file__).with_name(f)) for f in
                      ['train_staged_conversations.py', 'staged_character_grad.py',
                       'qwen_rng_checkpoint.py', 'conversation_objective.py']},
        windows=len(dataset), optimizer_steps=steps,
        assistant_targets_per_epoch=sum(sum(m[1:]) for _, m in dataset.items),
        trainable_parameters=sum(v.size for _, v in leaves),
        objective='Sum assistant-through-EOS cross entropy / supervised tokens in accumulation group',
        checkpointing='Staged tied head plus full-context transformer RNG-correct rematerialization')
    del leaves
    (out / 'training_config.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (out / 'adapter_config.json').write_text(json.dumps(adapter_cfg, indent=2) + '\n')
    optimizer = optim.Adam(learning_rate=cfg['learning_rate'])
    rng = random.Random(cfg['seed'])
    step = 0
    start = time.monotonic()
    for epoch in range(cfg['epochs']):
        order = list(range(len(dataset)))
        rng.shuffle(order)
        for offset in range(0, len(order), cfg['accumulation']):
            tick = time.monotonic()
            group = [dataset[i] for i in order[offset:offset + cfg['accumulation']]]
            denominator = sum(sum(mask[1:]) for _, mask in group)
            if denominator < 1:
                raise ValueError('Empty supervision group')
            gradient = None
            value = mx.array(0., dtype=mx.float32)
            for tokens, mask in group:
                (loss, count), part = staged_character_grad(model,
                    mx.array([tokens], dtype=mx.int32), mx.array([mask], dtype=mx.bool_),
                    chunk_size=cfg['head_chunk_size'], denominator=denominator)
                gradient = part if gradient is None else tree_map(lambda a, b: a + b, gradient, part)
                value = value + loss
                mx.eval(value, gradient)
                del part, loss, count
                mx.clear_cache()
            norm = mx.sqrt(sum(mx.sum(g.astype(mx.float32) ** 2) for _, g in tree_flatten(gradient)))
            mx.eval(value, norm)
            if not mx.isfinite(value).item() or not mx.isfinite(norm).item():
                raise FloatingPointError('Nonfinite loss or gradients before optimizer update')
            gradient = tree_map(lambda g: g * mx.minimum(1., cfg['clip_norm'] / (norm + 1e-8)), gradient)
            step += 1
            if step <= warmup:
                lr = cfg['learning_rate'] * (.1 + .9 * step / warmup)
            else:
                phase = (step - warmup) / max(steps - warmup, 1)
                lr = cfg['minimum_lr'] + .5 * (cfg['learning_rate'] - cfg['minimum_lr']) * (1 + math.cos(math.pi * phase))
            optimizer.learning_rate = lr
            optimizer.update(model, gradient)
            mx.eval(model.parameters(), optimizer.state)
            if not all(mx.all(mx.isfinite(v)).item() for _, v in tree_flatten(model.trainable_parameters())):
                raise FloatingPointError('Nonfinite adapter after optimizer update')
            record = dict(step=step, epoch=epoch + 1, microbatches=len(group),
                loss=value.item(), target_tokens=denominator, gradient_norm=norm.item(),
                learning_rate=lr, seconds=time.monotonic() - tick, peak_bytes=mx.get_peak_memory())
            with (out / 'metrics.jsonl').open('a') as f:
                f.write(json.dumps(record) + '\n')
            print(json.dumps(record), flush=True)
            del gradient, value, norm
            mx.clear_cache()
            if step in cfg['checkpoints'] or step == steps:
                checkpoint = out / f'{step:07d}_adapters.safetensors'
                mx.save_safetensors(str(checkpoint),
                                    dict(tree_flatten(model.trainable_parameters())))
                if cfg.get('resident_dev_capture', False):
                    capture_checkpoint(model, tok, cfg, out, step, checkpoint)
    mx.save_safetensors(str(out / 'adapters.safetensors'), dict(tree_flatten(model.trainable_parameters())))
    completion = dict(status='Training complete, unreviewed candidate', steps=step,
        seconds_including_resident_audits=time.monotonic() - start, peak_bytes=mx.get_peak_memory(),
        adapter_sha256=sha(out / 'adapters.safetensors'))
    (out / 'completion.json').write_text(json.dumps(completion, indent=2) + '\n')


if __name__ == '__main__':
    main()
