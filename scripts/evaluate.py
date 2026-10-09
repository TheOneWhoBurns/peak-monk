"""Reproducible loss and unedited conversation capture. No automatic quality claims."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors
from mlx_lm.tuner.datasets import ChatDataset, CacheDataset
from mlx_lm.tuner.trainer import evaluate
from assistant_objective import ThroughEOS, assistant_loss

ROOT = Path(__file__).resolve().parents[1]

def capture_conversations(model, tok, args, result, out):
    def save():
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    for case in json.loads((ROOT/args.suite).read_text()):
        messages = []
        if args.prompted:
            messages.append({'role':'system','content':(ROOT/args.persona_file).read_text()})
        captured = dict(case, responses=[], messages=[])
        for turn_index,user in enumerate(case['turns']):
            messages.append({'role':'user','content':user})
            prompt = tok.apply_chat_template(messages,add_generation_prompt=True,tokenize=False,enable_thinking=False)
            start = time.monotonic()
            first = None
            text = ''
            last = None
            if args.temperature>0:
                case_seed=int(hashlib.sha256(case['id'].encode()).hexdigest()[:8],16)
                mx.random.seed((args.seed+case_seed+997*turn_index)%(2**31))
            processors=make_logits_processors(presence_penalty=args.presence_penalty if args.presence_penalty else None,presence_context_size=128)
            for chunk in stream_generate(model,tok,prompt,max_tokens=args.max_tokens,
                    sampler=make_sampler(temp=args.temperature,top_p=args.top_p,top_k=getattr(args,'top_k',0)),logits_processors=processors):
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
    p.add_argument('--temperature',type=float,default=0.0)
    p.add_argument('--top-p',type=float,default=1.0)
    p.add_argument('--top-k',type=int,default=0)
    p.add_argument('--presence-penalty',type=float,default=0.0)
    p.add_argument('--seed',type=int,default=41)
    args = p.parse_args()
    mx.random.seed(args.seed)
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
              'seed':args.seed, 'temperature':args.temperature, 'top_p':args.top_p, 'top_k':args.top_k,
              'presence_penalty':args.presence_penalty,'presence_context_size':128,
              'sampling_seed_policy':'Per case/turn deterministic seed when temperature > 0, reset after loading weights; historical greedy mode unchanged.',
              'max_tokens':args.max_tokens,
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
    capture_conversations(model, tok, args, result, out)

if __name__ == '__main__':
    main()
