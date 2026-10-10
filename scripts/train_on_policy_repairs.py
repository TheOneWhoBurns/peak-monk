"""Full-history selective SFT repairs plus separate clean conversation replay.

Only the last declared transformer layers are updated. Earlier learned adapters
and the tied embedding stay frozen; exports retain every original LoRA factor.
No prefix truncation, positive loss on erroneous history, or reward-model claim.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import re
import time

from bounded_text_runtime import BoundedTextRuntime, atomic_json
from sample_dialogue_paths import sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    cli = parser.parse_args()
    cfg = json.loads(cli.config.read_text())
    out = Path(cfg['output'])
    if out.exists():
        raise ValueError('New experiment output required')
    for path, expected in [(Path(cfg['parent'])/'adapters.safetensors', cfg['parent_sha256']),
                           (Path(cfg['repairs']), cfg['repairs_sha256']),
                           (Path(cfg['replay']), cfg['replay_sha256'])]:
        if sha(path) != expected:
            raise ValueError(f'Source hash changed: {path}')
    repairs = [json.loads(s) for s in Path(cfg['repairs']).read_text().splitlines()]
    replay = [json.loads(s) for s in Path(cfg['replay']).read_text().splitlines()]
    if not repairs or not replay:
        raise ValueError('Repairs and retention examples required')
    if any(r.get('split') != 'train' or r.get('review') != 'author-approved' or
           r.get('supervised_message_indices') != [len(r['messages'])-1] for r in repairs):
        raise ValueError('Every repair must be explicitly approved, training-only and last-reply supervised')
    if any(r.get('split', 'train') != 'train' for r in replay):
        raise ValueError('Retention data must be training-only')
    if not 1 <= cfg['train_last_layers'] <= 36 or not 1 <= cfg['steps'] <= 128:
        raise ValueError('Invalid bounded training configuration')
    if not 0 < cfg['max_length'] <= 2048 or cfg['memory_limit'] > 4_200_000_000:
        raise ValueError('This trainer requires a new protocol for larger memory/context')
    out.mkdir(parents=True)
    atomic_json(out/'plan.json', cfg)
    with BoundedTextRuntime(out, memory_limit=cfg['memory_limit'], max_seconds=cfg['max_seconds']) as runtime:
        import mlx.core as mx
        import mlx.optimizers as optim
        import numpy as np
        from mlx.utils import tree_flatten, tree_map
        from mlx_lm import load
        from conversation_objective import encode_conversation
        from staged_character_grad import staged_character_grad
        from qwen_rng_checkpoint import qwen_rng_checkpoint

        runtime.event('load_begin')
        model, tok = load(cfg['model'], adapter_path=cfg['parent'])
        model.freeze()
        first = len(model.layers) - cfg['train_last_layers']
        for name, module in model.named_modules():
            match = re.match(r'model\.layers\.(\d+)\.', name)
            if match and int(match[1]) >= first and 'lora_a' in module and 'lora_b' in module:
                module.unfreeze(recurse=False, keys=['lora_a', 'lora_b'], strict=True)
        trainable = dict(tree_flatten(model.trainable_parameters()))
        if not trainable or any(not k.endswith(('.lora_a', '.lora_b')) for k in trainable):
            raise ValueError('Unexpected trainable parameter selection')
        trainable_names = set(trainable)
        def all_adapters():
            return {k: v for k, v in tree_flatten(model.parameters()) if k.endswith(('.lora_a', '.lora_b'))}
        def frozen_digest():
            digest = hashlib.sha256()
            for name, value in sorted(all_adapters().items()):
                if name not in trainable_names:
                    digest.update(name.encode())
                    digest.update(np.asarray(value).tobytes())
            return digest.hexdigest()
        frozen_before = frozen_digest()
        qwen_rng_checkpoint(model.layers[0])
        model.train()
        model.model.embed_tokens.eval()
        for layer in model.layers[:first]:
            layer.eval()
        mx.eval(model.parameters())
        runtime.event('load_end', **runtime.check())
        prepared, clean = [], []
        for rows, destination in [(repairs, prepared), (replay, clean)]:
            for row in rows:
                tokens, mask = encode_conversation(row, tok)
                if len(tokens) > cfg['max_length']:
                    raise ValueError(f"Full history exceeds declared length: {row.get('id',row.get('family'))}")
                destination.append((tokens, mask))
        source_names = ['train_on_policy_repairs.py', 'bounded_text_runtime.py', 'conversation_objective.py',
                        'staged_character_grad.py', 'qwen_rng_checkpoint.py']
        atomic_json(out/'metadata.json', dict(config_sha256=sha(cli.config),
            source_sha256={n: sha(Path(__file__).with_name(n)) for n in source_names},
            trainable_parameters=sum(v.size for v in trainable.values()), trainable_names=sorted(trainable_names),
            frozen_adapter_sha256=frozen_before, repair_count=len(prepared), replay_count=len(clean),
            repair_lengths=[len(x) for x, _ in prepared], replay_lengths=[len(x) for x, _ in clean],
            objective='repair_weight times next-reply token-mean CE plus replay_weight times clean-conversation token-mean CE',
            context='Complete histories, no truncated gradient within the trainable layers; lower layers and embedding intentionally frozen'))
        del trainable
        adapter_config = json.loads((Path(cfg['parent'])/'adapter_config.json').read_text())
        atomic_json(out/'adapter_config.json', adapter_config)
        optimizer = optim.AdamW(learning_rate=cfg['learning_rate'], weight_decay=0.)
        rng = random.Random(cfg['seed'])
        order, retention_order = list(range(len(prepared))), list(range(len(clean)))
        mx.random.seed(cfg['seed'])
        start = time.monotonic()
        for step in range(cfg['steps']):
            if step % len(order) == 0:
                rng.shuffle(order)
            if step % len(retention_order) == 0:
                rng.shuffle(retention_order)
            ri, ci = order[step % len(order)], retention_order[step % len(retention_order)]
            runtime.check()
            runtime.event('update_begin', step=step+1, repair_index=ri, replay_index=ci)
            total, gradient = mx.array(0.), None
            losses = []
            for (tokens, mask), weight in [(prepared[ri], cfg['repair_weight']), (clean[ci], cfg['replay_weight'])]:
                (loss, _), part = staged_character_grad(model, mx.array([tokens]), mx.array([mask]),
                                                       chunk_size=cfg['head_chunk_size'])
                scaled = tree_map(lambda g: weight*g, part)
                gradient = scaled if gradient is None else tree_map(lambda a,b:a+b, gradient, scaled)
                total = total + weight*loss
                mx.eval(total, gradient)
                losses.append(loss.item())
                del loss, part, scaled
                runtime.response_done()
            norm = mx.sqrt(sum(mx.sum(g.astype(mx.float32)**2) for _,g in tree_flatten(gradient)))
            mx.eval(total, norm)
            if not mx.isfinite(total).item() or not mx.isfinite(norm).item():
                raise FloatingPointError('Nonfinite update')
            gradient = tree_map(lambda g:g*mx.minimum(1.,1./(norm+1e-8)), gradient)
            optimizer.update(model, gradient)
            mx.eval(model.parameters(), optimizer.state)
            if not all(mx.all(mx.isfinite(v)).item() for _,v in tree_flatten(model.trainable_parameters())):
                raise FloatingPointError('Nonfinite adapter')
            row = dict(step=step+1, repair_id=repairs[ri]['id'], replay_family=replay[ci]['family'],
                       loss=total.item(), repair_loss=losses[0], replay_loss=losses[1],
                       gradient_norm=norm.item(), training_elapsed_seconds=time.monotonic()-start)
            del gradient, total, norm
            row.update(runtime.response_done())
            runtime.event('update_end', **row)
            print(json.dumps(row), flush=True)
            if step+1 in cfg['checkpoints'] or step+1 == cfg['steps']:
                if frozen_digest() != frozen_before:
                    raise RuntimeError('Frozen adapter factors changed')
                path = out/f'{step+1:07d}_adapters.safetensors'
                temporary = path.with_suffix('.tmp.safetensors')
                mx.save_safetensors(str(temporary), all_adapters())
                with temporary.open('rb') as f:
                    os.fsync(f.fileno())
                os.replace(temporary, path)
                atomic_json(out/f'step{step+1:04d}.json', dict(step=step+1, adapter_sha256=sha(path),
                    frozen_adapter_unchanged=True, status='Unreviewed research export'))
                if step+1 == cfg['steps']:
                    os.link(path, out/'adapters.safetensors')
        atomic_json(out/'completion.json', dict(steps=cfg['steps'], status='Training complete; no acceptance claim',
                                                seconds=time.monotonic()-start))


if __name__ == '__main__':
    main()
