"""Reproducible loss and unedited conversation capture. No automatic quality claims."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tuner.datasets import ChatDataset, CacheDataset
from mlx_lm.tuner.trainer import evaluate

ROOT = Path(__file__).resolve().parents[1]

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--adapter')
    p.add_argument('--suite', default='evaluation/dev.json')
    p.add_argument('--output', required=True)
    p.add_argument('--loss-only', action='store_true')
    p.add_argument('--prompted', action='store_true')
    p.add_argument('--max-tokens', type=int, default=160)
    args = p.parse_args()
    mx.random.seed(41)
    model, tok = load(str(ROOT/'models/base'), adapter_path=args.adapter)
    rows = [json.loads(l) for l in (ROOT/'data/prepared/valid.jsonl').read_text().splitlines()]
    start = time.monotonic()
    loss = evaluate(model, CacheDataset(ChatDataset(rows,tok,mask_prompt=True)),
                    batch_size=1,num_batches=-1,max_seq_length=768)
    result = {'adapter': args.adapter, 'prompted': args.prompted,
              'validation_loss': loss, 'loss_seconds': time.monotonic()-start,
              'seed':41, 'temperature':0.0, 'max_tokens':args.max_tokens,
              'quality_scoring': 'Not yet scored; raw outputs for explicit rubric review.',
              'suite': args.suite, 'suite_sha256': hashlib.sha256((ROOT/args.suite).read_bytes()).hexdigest(),
              'conversations':[]}
    out = ROOT/args.output
    out.parent.mkdir(parents=True,exist_ok=True)
    def save():
        out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    save()
    print(f'validation_loss={loss:.6f}',flush=True)
    if args.loss_only:
        return
    for case in json.loads((ROOT/args.suite).read_text()):
        messages = []
        if args.prompted:
            messages.append({'role':'system','content':(ROOT/'character/core.md').read_text()})
        captured = dict(case, responses=[], messages=[])
        for user in case['turns']:
            messages.append({'role':'user','content':user})
            prompt = tok.apply_chat_template(messages,add_generation_prompt=True,tokenize=False)
            start = time.monotonic()
            first = None
            text = ''
            last = None
            for chunk in stream_generate(model,tok,prompt,max_tokens=args.max_tokens,sampler=make_sampler(temp=0.0)):
                if first is None:
                    first = time.monotonic()-start
                text += chunk.text
                last = chunk
            messages.append({'role':'assistant','content':text})
            captured['responses'].append({'text':text,'first_token_seconds':first,
                'elapsed_seconds':time.monotonic()-start,
                'tokens':last.generation_tokens,'tokens_per_second':last.generation_tps,
                'peak_memory_gb':last.peak_memory,'finish_reason':last.finish_reason})
            mx.clear_cache()
        captured['messages'] = messages
        result['conversations'].append(captured)
        save()
        print(case['id']+': '+captured['responses'][-1]['text'][:180].replace('\n',' '),flush=True)
    print('saved',out,flush=True)

if __name__ == '__main__':
    main()
