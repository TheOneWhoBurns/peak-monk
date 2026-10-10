"""Resumable, token-accounted training trajectories with explicit ancestry.

Manifests can continue saved spines or branch from an earlier exchange. Callers
are reviewed between batches; this is not an autonomous partner-model judge.
Dry-run uses only the standard library and does not initialize Metal.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

from bounded_text_runtime import BoundedTextRuntime, atomic_json


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def prefix_for(job):
    if not job.get('prefix'):
        return [], None
    source = job['prefix']
    path = Path(source['path'])
    if sha(path) != source['sha256']:
        raise ValueError('Prefix source changed')
    record = json.loads(path.read_text())
    if record['root_id'] != job['root_id']:
        raise ValueError('Branch must retain root identity')
    count = source['exchanges'] * 2
    messages = record['messages'][:count]
    if count < 2 or len(messages) != count:
        raise ValueError('Prefix exchange unavailable')
    return messages, source


def validate(manifest):
    jobs = manifest['jobs']
    if manifest['split'] != 'train' or not jobs:
        raise ValueError('Only explicit nonempty training manifests are supported')
    if len({job['id'] for job in jobs}) != len(jobs):
        raise ValueError('Duplicate job IDs')
    for job in jobs:
        if not job['id'].replace('_', '').replace('-', '').isalnum():
            raise ValueError('Unsafe job filename')
        if not job['caller_turns'] or any(not isinstance(t, str) or not t.strip() for t in job['caller_turns']):
            raise ValueError('Caller turns must be nonempty text')
        messages, _ = prefix_for(job)
        if any(m['role'] != ('user' if i % 2 == 0 else 'assistant') for i, m in enumerate(messages)):
            raise ValueError('Expected alternating zero-system conversation')
    return jobs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    manifest = json.loads(a.manifest.read_text())
    jobs = validate(manifest)
    cfg = manifest['generation']
    if not 0 < cfg['max_input_tokens'] <= 2048 or not 0 < cfg['max_output_tokens'] <= 160:
        raise ValueError('Pilot token bounds exceeded; expansion requires a declared protocol change')
    if a.dry_run:
        print(json.dumps(dict(jobs=len(jobs), replies=sum(len(j['caller_turns']) for j in jobs),
                              status='Manifest and ancestry valid; no MLX import or model load')))
        return
    a.output.mkdir(parents=True, exist_ok=True)
    config = dict(manifest=manifest, manifest_sha256=sha(a.manifest),
                  source_sha256={n: sha(Path(__file__).with_name(n)) for n in
                                 ['sample_dialogue_paths.py', 'bounded_text_runtime.py']})
    if sha(Path(manifest['adapter'])/'adapters.safetensors') != manifest['adapter_sha256']:
        raise ValueError('Research checkpoint hash differs')
    previous = a.output/'config.json'
    if previous.exists() and json.loads(previous.read_text()) != config:
        raise ValueError('Resume requires identical manifest and sampler source')
    atomic_json(previous, config)
    with BoundedTextRuntime(a.output, max_seconds=cfg.get('max_seconds', 600)) as runtime:
        import mlx.core as mx
        from mlx_lm import load, stream_generate
        from mlx_lm.sample_utils import make_sampler
        runtime.event('load_begin')
        model, tok = load(manifest['model'], adapter_path=manifest['adapter'])
        model.eval()
        mx.eval(model.parameters())
        runtime.event('load_end', **runtime.check())
        for job in jobs:
            path = a.output/(job['id']+'.json')
            prefix, source = prefix_for(job)
            if path.exists():
                record = json.loads(path.read_text())
                if record['status'] == 'complete':
                    continue
                if record['status'] == 'context_censored':
                    continue
            else:
                record = dict(id=job['id'], root_id=job['root_id'], family=job['family'],
                              kind=job['kind'], prefix_source=source, split='train',
                              messages=prefix, turns=[], status='running', review='pending')
            for index in range(len(record['turns']), len(job['caller_turns'])):
                messages = record['messages'] + [dict(role='user', content=job['caller_turns'][index])]
                tokens = tok.apply_chat_template(messages, tokenize=True,
                                                add_generation_prompt=True, enable_thinking=False)
                if len(tokens) > cfg['max_input_tokens']:
                    record.update(status='context_censored', next_input_tokens=len(tokens))
                    atomic_json(path, record)
                    break
                seed = int(hashlib.sha256(f"{job['seed']}:{job['id']}:{index}".encode()).hexdigest()[:8], 16) % 2**31
                mx.random.seed(seed)
                runtime.check()
                runtime.event('response_begin', job=job['id'], index=index, input_tokens=len(tokens), seed=seed)
                start = time.monotonic()
                text, last = '', None
                generator = stream_generate(model, tok, tokens, max_tokens=cfg['max_output_tokens'],
                    prefill_step_size=128, sampler=make_sampler(temp=cfg['temperature'], top_p=cfg['top_p']))
                try:
                    for chunk in generator:
                        text += chunk.text
                        last = chunk
                        if chunk.generation_tokens % 16 == 0:
                            runtime.check()
                finally:
                    generator.close()
                if last is None:
                    raise RuntimeError('Empty generation stream')
                metrics = runtime.response_done()
                record['turns'].append(dict(caller=job['caller_turns'][index], reply=text,
                    seed=seed, input_tokens=len(tokens), output_tokens=last.generation_tokens,
                    finish_reason=last.finish_reason, seconds=time.monotonic()-start, **metrics))
                record['messages'] = messages + [dict(role='assistant', content=text)]
                record['status'] = 'complete' if index+1 == len(job['caller_turns']) else 'running'
                atomic_json(path, record)
                runtime.event('response_saved', job=job['id'], index=index, **metrics)
                print(job['id'], index, text.replace('\n', ' ')[:220], flush=True)
    atomic_json(a.output/'completion.json', dict(status='batch_finished', jobs=len(jobs),
        statuses={job['id']: json.loads((a.output/(job['id']+'.json')).read_text())['status'] for job in jobs}))


if __name__ == '__main__':
    main()
