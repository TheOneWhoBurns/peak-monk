"""Interactive local character inference. Intentionally no persona system prompt."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def resolve_inputs(adapter, model=None, root=ROOT):
    def local(value):
        path = Path(value)
        return (path if path.is_absolute() else root / path).resolve()

    adapter_path = local(adapter)
    config_path = adapter_path / 'adapter_config.json'
    if not config_path.is_file():
        raise ValueError('Adapter directory needs its saved adapter_config.json')
    trained_base = json.loads(config_path.read_text()).get('model')
    if not trained_base:
        raise ValueError('Adapter configuration does not identify its training base')
    expected = local(trained_base)
    selected = local(model) if model else expected
    if selected != expected:
        raise ValueError(f'Adapter was trained on {expected}, not {selected}')
    if not selected.is_dir() or not (adapter_path / 'adapters.safetensors').is_file():
        raise ValueError('The recorded model and saved adapter weights must exist locally')
    return selected, adapter_path

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',help='Defaults to the adapter training base; must match it')
    p.add_argument('--adapter',required=True,
                   help='Explicit experimental adapter directory; no accepted release exists yet')
    p.add_argument('--temperature',type=float,default=0.0,
                   help='Default matches greedy development evaluation')
    p.add_argument('--max-tokens',type=int,default=180)
    args=p.parse_args()
    try:
        model_path, adapter_path = resolve_inputs(args.adapter, args.model)
    except (ValueError, OSError) as error:
        p.error(str(error))
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler
    model,tok=load(str(model_path),adapter_path=str(adapter_path))
    messages=[]
    print('Jun — fictional monk, running locally. /reset clears context; /quit exits.')
    while True:
        try:
            user=input('You: ').strip()
        except (EOFError,KeyboardInterrupt):
            break
        if user=='/quit': break
        if user=='/reset': messages=[]; continue
        if not user: continue
        messages.append({'role':'user','content':user})
        # Keep complete recent exchanges under a modest local context budget.
        while len(tok.apply_chat_template(messages,add_generation_prompt=True,return_dict=False))>2048 and len(messages)>1:
            messages=messages[2:]
        prompt=tok.apply_chat_template(messages,add_generation_prompt=True,tokenize=False)
        print('Jun: ',end='',flush=True)
        text=''
        for chunk in stream_generate(model,tok,prompt,max_tokens=args.max_tokens,
                                     sampler=make_sampler(temp=args.temperature,top_p=0.9)):
            text+=chunk.text
            print(chunk.text,end='',flush=True)
        print()
        messages.append({'role':'assistant','content':text})
        mx.clear_cache()

if __name__=='__main__': main()
