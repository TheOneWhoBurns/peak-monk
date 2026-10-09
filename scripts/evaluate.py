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
from assistant_objective import ThroughEOS, assistant_loss

ROOT = Path(__file__).resolve().parents[1]

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--adapter')
    p.add_argument('--model', default='models/base')
    p.add_argument('--suite', default='evaluation/dev.json')
    p.add_argument('--output', required=True)
    p.add_argument('--loss-only', action='store_true')
    p.add_argument('--prompted', action='store_true')
    p.add_argument('--persona-file', default='character/core.md')
    p.add_argument('--loss-version', choices=['legacy', 'assistant_eos_v2'], default='legacy')
    p.add_argument('--max-tokens', type=int, default=160)
    args = p.parse_args()
    mx.random.seed(41)
    model, tok = load(str(ROOT/args.model), adapter_path=args.adapter)
    serialization = hashlib.sha256()
    for name in ('config.json','tokenizer.json','tokenizer_config.json','chat_template.jinja'):
        path=ROOT/args.model/name
        if path.exists():
            serialization.update(name.encode()); serialization.update(path.read_bytes())
    rows = [json.loads(l) for l in (ROOT/'data/prepared/valid.jsonl').read_text().splitlines()]
    start = time.monotonic()
    dataset = ChatDataset(rows,tok,mask_prompt=True)
    kwargs = {}
    if args.loss_version == 'assistant_eos_v2':
        dataset = ThroughEOS(dataset, tok.eos_token_id)
        kwargs['loss'] = assistant_loss
    loss = evaluate(model, CacheDataset(dataset),
                    batch_size=1,num_batches=-1,max_seq_length=768, **kwargs)
    result = {'model':args.model, 'adapter': args.adapter, 'prompted': args.prompted,
              'serialization_sha256':serialization.hexdigest(),
              'persona_file':args.persona_file if args.prompted else None,
              'validation_loss': loss, 'loss_version':args.loss_version, 'loss_seconds': time.monotonic()-start,
              'loss_comparison_note':'Compare loss only between identical base/tokenizer/template runs. Reference models may serialize differently; their conversation audit is the comparison.',
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
            messages.append({'role':'system','content':(ROOT/args.persona_file).read_text()})
        captured = dict(case, responses=[], messages=[])
        for user in case['turns']:
            messages.append({'role':'user','content':user})
            prompt = tok.apply_chat_template(messages,add_generation_prompt=True,tokenize=False,enable_thinking=False)
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
