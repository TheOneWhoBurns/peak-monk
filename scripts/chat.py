"""Interactive local character inference. Intentionally no persona system prompt."""
import argparse
import json
from pathlib import Path
import tempfile

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
    p.add_argument('--max-tokens',type=int,default=160)
    p.add_argument('--seed',type=int,default=41)
    p.add_argument('--runtime-log',type=Path,
                   help='New directory for resource/provenance logs; defaults under results')
    args=p.parse_args()
    try:
        model_path, adapter_path = resolve_inputs(args.adapter, args.model)
    except (ValueError, OSError) as error:
        p.error(str(error))
    if not 1 <= args.max_tokens <= 160 or args.temperature < 0:
        p.error('Use1–160 output tokens and nonnegative temperature')
    from bounded_text_runtime import BoundedTextRuntime, atomic_json
    from sample_dialogue_paths import sha
    output=args.runtime_log
    if output is None:
        output=Path(tempfile.mkdtemp(prefix='chat-',dir=ROOT/'results'))
    else:
        output.mkdir(parents=True,exist_ok=False)
    atomic_json(output/'config.json',dict(model=str(model_path),adapter=str(adapter_path),
        adapter_sha256=sha(adapter_path/'adapters.safetensors'),
        source_sha256={name:sha(Path(__file__).with_name(name)) for name in
                       ['chat.py','bounded_text_runtime.py','qwen_streamed_inference.py']},
        persona_system_prompt=False,resident_layers=8,max_input_tokens=2048,
        max_output_tokens=args.max_tokens,temperature=args.temperature,seed=args.seed,
        context_policy='Stop before exceeding input budget; never silently discard history.'))
    with BoundedTextRuntime(output,memory_limit=1_800_000_000,max_seconds=3600) as runtime:
        import mlx.core as mx
        from mlx_lm import load, stream_generate
        from mlx_lm.sample_utils import make_sampler
        from qwen_streamed_inference import install_streaming
        model,tok=load(str(model_path),adapter_path=str(adapter_path),lazy=True)
        model.eval()

        def boundary(index):
            metrics=runtime.check()
            if metrics['mlx_peak_bytes']>2_000_000_000:
                raise RuntimeError('Interactive text inference exceeded2GB peak plan')

        install_streaming(model,model_path,resident_layers=8,boundary=boundary)
        mx.eval(model.parameters())
        runtime.event('load_end',**runtime.settle_after_load())
        mx.random.seed(args.seed)
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
            proposed=messages+[{'role':'user','content':user}]
            tokens=tok.apply_chat_template(proposed,add_generation_prompt=True,
                                          tokenize=True,enable_thinking=False)
            if len(tokens)>2048:
                print('Context limit reached. Use /reset to begin again, or shorten this message.')
                continue
            runtime.check()
            print('Jun: ',end='',flush=True)
            text=''
            generator=stream_generate(model,tok,tokens,max_tokens=args.max_tokens,
                                      sampler=make_sampler(temp=args.temperature,top_p=1.0))
            try:
                for chunk in generator:
                    text+=chunk.text
                    print(chunk.text,end='',flush=True)
            finally:
                generator.close()
            print()
            messages=proposed+[{'role':'assistant','content':text}]
            runtime.event('response_complete',input_tokens=len(tokens),**runtime.response_done())

if __name__=='__main__': main()
