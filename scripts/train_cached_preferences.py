"""Full-history offline DPO, chosen-reply NLL and clean conversation replay.

Only the last declared transformer layers are updated. Earlier learned adapters
and the tied embedding stay frozen; exports retain every original LoRA factor.
No prefix truncation, positive loss on erroneous history, or reward-model claim.
"""
import argparse
import gc
import hashlib
import inspect
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
    if not cfg.get('stream_frozen_prefix') or not cfg.get('cache_frozen_prefix'):
        raise ValueError('Preference trainer requires the streamed cached prefix')
    if cfg['preference_beta'] <= 0 or cfg['chosen_nll_weight'] < 0:
        raise ValueError('Invalid preference weights')
    stream_prefix = bool(cfg.get('stream_frozen_prefix'))
    if stream_prefix and not cfg.get('cache_frozen_prefix'):
        raise ValueError('Frozen-prefix streaming requires cached-prefix training')
    out = Path(cfg['output'])
    if out.exists():
        raise ValueError('New experiment output required')
    for path, expected in [(Path(cfg['parent'])/'adapters.safetensors', cfg['parent_sha256']),
                           (Path(cfg['repairs']), cfg['repairs_sha256']),
                           (Path(cfg['replay']), cfg['replay_sha256'])]:
        if sha(path) != expected:
            raise ValueError(f'Source hash changed: {path}')
    repairs = [json.loads(s) for s in Path(cfg['repairs']).read_text().splitlines()]
    if any(not isinstance(r.get('rejected'), str) or not r['rejected'].strip() for r in repairs):
        raise ValueError('Each preference requires its reviewed sampled alternative')
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
        from repair_preference_grad import repair_preference_grad, masked_reply_score
        from mlx_lm.tuner.trainer import grad_checkpoint

        runtime.event('load_begin')
        model, tok = load(cfg['model'], adapter_path=cfg['parent'], lazy=stream_prefix)
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
        suffix_offset, parent_factors = 0, None
        def all_adapters():
            values = {} if parent_factors is None else dict(parent_factors)
            for key, value in tree_flatten(model.parameters()):
                if not key.endswith(('.lora_a', '.lora_b')):
                    continue
                if suffix_offset:
                    key = re.sub(r'^model\.layers\.(\d+)\.',
                        lambda m:f'model.layers.{int(m[1])+suffix_offset}.', key)
                values[key] = value
            return values
        def frozen_digest():
            digest = hashlib.sha256()
            for name, value in sorted(all_adapters().items()):
                if name not in trainable_names:
                    digest.update(name.encode())
                    digest.update(np.asarray(value).tobytes())
            return digest.hexdigest()
        frozen_before = frozen_digest()
        # All dropout is disabled below. The upstream deterministic checkpoint
        # remains active in eval mode, unlike the stochastic-training wrapper.
        grad_checkpoint(model.layers[0])
        if cfg.get('query_chunk_size'):
            from qwen_chunked_attention import chunk_qwen_attention
            chunk_qwen_attention(model, cfg['query_chunk_size'])
        model.eval()  # Reference and policy probabilities both disable dropout.
        model.model.embed_tokens.eval()
        for layer in model.layers[:first]:
            layer.eval()
        if stream_prefix:
            if first == 0:
                raise ValueError('No frozen prefix to stream')
            # Keep every original adapter factor for frozen hashing and export;
            # the inference-only prefix wrappers deliberately hide base tensors.
            parent_factors = all_adapters()
            from qwen_streamed_inference import install_frozen_prefix
            install_frozen_prefix(model,cfg['model'],first,lambda index:runtime.check())
        mx.eval(model.parameters())
        runtime.event('load_end', **runtime.settle_after_load())
        import copy
        rejected_rows = copy.deepcopy(repairs)
        for row in rejected_rows:
            row['messages'][-1]['content'] = row['rejected']
        prepared, rejected, clean = [], [], []
        for rows, destination in [(repairs, prepared), (rejected_rows, rejected), (replay, clean)]:
            for row in rows:
                tokens, mask = encode_conversation(row, tok)
                if len(tokens) > cfg['max_length']:
                    raise ValueError(f"Full history exceeds declared length: {row.get('id',row.get('family'))}")
                destination.append((tokens, mask))
        source_names = ['train_cached_preferences.py', 'repair_preference_grad.py', 'bounded_text_runtime.py', 'conversation_objective.py',
                        'staged_character_grad.py', 'qwen_chunked_attention.py',
                        'repair_export_score.py', 'qwen_streamed_inference.py']
        atomic_json(out/'metadata.json', dict(config_sha256=sha(cli.config),
            source_sha256={n: sha(Path(__file__).with_name(n)) for n in source_names},
            checkpoint_implementation_sha256=sha(Path(inspect.getfile(grad_checkpoint))),
            trainable_parameters=sum(v.size for v in trainable.values()), trainable_names=sorted(trainable_names),
            frozen_adapter_sha256=frozen_before, repair_count=len(prepared), replay_count=len(clean),
            repair_lengths=[len(x) for x, _ in prepared], replay_lengths=[len(x) for x, _ in clean],
            objective='repair_weight times (DPO plus chosen_nll_weight times chosen token-mean CE) plus replay_weight times clean token-mean CE; dropout disabled',
            context='Complete histories, no truncated gradient within the trainable layers; lower layers and embedding intentionally frozen'))
        del trainable
        adapter_config = json.loads((Path(cfg['parent'])/'adapter_config.json').read_text())
        atomic_json(out/'adapter_config.json', adapter_config)
        cached = bool(cfg.get('cache_frozen_prefix'))
        if cached:
            if first == 0:
                raise ValueError('No frozen prefix to cache')
            from mlx_lm.models.base import create_attention_mask
            cache_dir = out/'frozen_prefix'
            cache_dir.mkdir()
            cache_manifest = []
            for split, items in [('repairs', prepared), ('rejected', rejected), ('replay', clean)]:
                for index, (tokens, _) in enumerate(items):
                    runtime.check()
                    runtime.event('prefix_cache_begin', split=split, index=index, tokens=len(tokens))
                    batch = mx.array([tokens[:-1]])
                    hidden = model.model.embed_tokens(batch)
                    mask = create_attention_mask(hidden, None)
                    mx.eval(hidden)
                    for layer in model.layers[:first]:
                        hidden = layer(hidden, mask)
                        mx.eval(hidden)
                    path = cache_dir/f'{split}-{index:04d}.safetensors'
                    mx.save_safetensors(str(path), {'hidden': hidden})
                    cache_manifest.append(dict(split=split, index=index, path=str(path),
                        sha256=sha(path), shape=list(hidden.shape), dtype=str(hidden.dtype),
                        input_tokens_sha256=hashlib.sha256(json.dumps(tokens[:-1]).encode()).hexdigest()))
                    del hidden, batch, mask, layer
                    runtime.event('prefix_cache_end', split=split, index=index, **runtime.response_done())
            atomic_json(out/'frozen_prefix_manifest.json', dict(parent_sha256=cfg['parent_sha256'],
                frozen_layers=first, frozen_adapter_sha256=frozen_before, entries=cache_manifest,
                scope='Deterministic frozen embedding and lower layers only; no final norm, no target loss, no context truncation'))
            # Release the large frozen backbone before any backward pass.
            model.model.layers = model.layers[first:]
            suffix_offset = first
            gc.collect()
            runtime.response_done()
            parent_factors = mx.load(str(Path(cfg['parent'])/'adapters.safetensors'))
            mx.eval(parent_factors)
            if frozen_digest() != frozen_before:
                raise RuntimeError('Suffix export remapping changed frozen factors')
            runtime.event('frozen_backbone_unloaded', **runtime.response_done())
            mx.reset_peak_memory()
            runtime.event('peak_counter_reset', scope='Subsequent MLX peaks cover suffix training; earlier prefix-computation peaks remain in runtime log')
        references = []
        for index, (chosen_item, rejected_item) in enumerate(zip(prepared, rejected)):
            scores = []
            for split, item in [('repairs', chosen_item), ('rejected', rejected_item)]:
                features = mx.load(str(cache_dir/f'{split}-{index:04d}.safetensors'))['hidden']
                score = masked_reply_score(model, mx.array([item[0]]), mx.array([item[1]]),
                    features=features, chunk_size=cfg['head_chunk_size'])
                scores.append(score)
                del features
                runtime.response_done()
            references.append(dict(id=repairs[index]['id'], chosen=scores[0], rejected=scores[1],
                margin=scores[0]['logp']-scores[1]['logp']))
            runtime.event('reference_pair_complete', index=index, **runtime.response_done())
        atomic_json(out/'reference_scores.json', dict(parent=cfg['parent'], rows=references,
            scope='Frozen parent probabilities before updates; identical masks, head chunks and disabled dropout.'))
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
            cfeatures = mx.load(str(cache_dir/f'repairs-{ri:04d}.safetensors'))['hidden']
            rfeatures = mx.load(str(cache_dir/f'rejected-{ri:04d}.safetensors'))['hidden']
            chosen = tuple(mx.array([v]) for v in prepared[ri])
            negative = tuple(mx.array([v]) for v in rejected[ri])
            loss, gradient, diagnostic = repair_preference_grad(model, chosen, negative, references[ri]['margin'],
                beta=cfg['preference_beta'], nll_weight=cfg['chosen_nll_weight'],
                chosen_features=cfeatures, rejected_features=rfeatures, chunk_size=cfg['head_chunk_size'])
            if step == 0 and abs(diagnostic['relative_margin']) > 1e-3:
                raise ValueError('Initial policy/reference probabilities disagree')
            total = cfg['repair_weight']*loss
            gradient = tree_map(lambda g: cfg['repair_weight']*g, gradient)
            mx.eval(total, gradient)
            del cfeatures, rfeatures, chosen, negative, loss
            runtime.response_done()
            tokens, mask = clean[ci]
            features = mx.load(str(cache_dir/f'replay-{ci:04d}.safetensors'))['hidden']
            (replay_loss, _), part = staged_character_grad(model, mx.array([tokens]), mx.array([mask]),
                chunk_size=cfg['head_chunk_size'], input_embeddings=features)
            total = total + cfg['replay_weight']*replay_loss
            gradient = tree_map(lambda g, r: g+cfg['replay_weight']*r, gradient, part)
            mx.eval(total, gradient, replay_loss)
            replay_value = replay_loss.item()
            del part, features, replay_loss
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
                       loss=total.item(), preference=diagnostic, replay_loss=replay_value,
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
                from repair_export_score import selected_score
                model.eval()
                features = mx.load(str(cache_dir/'repairs-0000.safetensors'))['hidden'] if cached else None
                try:
                    score = selected_score(model, *prepared[0], input_embeddings=features)
                    atomic_json(out/f'step{step+1:04d}-export-score.json', dict(model=cfg['model'],
                        adapter_sha256=sha(path), data=cfg['repairs'], data_sha256=cfg['repairs_sha256'],
                        row_index=0, score=score, cached_prefix=cached))
                finally:
                    del features
                    model.eval()
                    model.model.embed_tokens.eval()
                    if not cached:
                        for layer in model.layers[:first]:
                            layer.eval()
                runtime.response_done()
                if step+1 == cfg['steps']:
                    os.link(path, out/'adapters.safetensors')
        atomic_json(out/'completion.json', dict(steps=cfg['steps'], status='Training complete; no acceptance claim',
                                                seconds=time.monotonic()-start))


if __name__ == '__main__':
    main()
