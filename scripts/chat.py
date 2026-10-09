"""Interactive local character inference. Intentionally no persona system prompt."""
import argparse
from pathlib import Path
import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',default='models/stronger-clean')
    p.add_argument('--adapter',required=True,
                   help='Explicit experimental adapter directory; no accepted release exists yet')
    p.add_argument('--temperature',type=float,default=0.6)
    p.add_argument('--max-tokens',type=int,default=180)
    args=p.parse_args()
    if not Path(args.model).is_dir() or not Path(args.adapter).is_dir():
        p.error('Model and adapter must be existing local directories')
    model,tok=load(args.model,adapter_path=args.adapter)
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
