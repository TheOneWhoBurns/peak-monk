"""Tiny real Qwen3 parity: base, nonzero LoRA, prefill, KV, and sampled tokens."""
import argparse
import json
import tempfile
from dataclasses import asdict
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.models.qwen3 import Model,ModelArgs
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler
from mlx_lm.utils import load_model
from mlx_lm.tuner.utils import linear_to_lora_layers,load_adapters
from qwen_streamed_inference import install_streaming,install_frozen_prefix


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise ValueError('Preserve previous proof')
    mx.random.seed(307)
    cfg=ModelArgs(model_type='qwen3',hidden_size=64,num_hidden_layers=4,
        intermediate_size=128,num_attention_heads=4,rms_norm_eps=1e-6,
        vocab_size=256,num_key_value_heads=2,max_position_embeddings=4096,
        head_dim=16,tie_word_embeddings=True,rope_theta=1e6)
    model=Model(cfg);model.set_dtype(mx.bfloat16);nn.quantize(model,group_size=64,bits=4)
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp);base=root/'base';adapter=root/'adapter';base.mkdir();adapter.mkdir()
        (base/'config.json').write_text(json.dumps(dict(asdict(cfg),quantization=dict(group_size=64,bits=4))))
        mx.save_safetensors(str(base/'model.safetensors'),dict(tree_flatten(model.parameters())))
        lora=dict(rank=2,scale=2.,dropout=.05,keys=['self_attn.q_proj','self_attn.k_proj',
            'self_attn.v_proj','self_attn.o_proj','mlp.gate_proj','mlp.up_proj','mlp.down_proj','model.embed_tokens'])
        linear_to_lora_layers(model,4,lora)
        for key,value in tree_flatten(model.parameters()):
            if key.endswith('.lora_b'):
                model.load_weights([(key,mx.random.normal(value.shape).astype(value.dtype)*.02)],strict=False)
        mx.save_safetensors(str(adapter/'adapters.safetensors'),{k:v for k,v in tree_flatten(model.parameters()) if k.endswith(('.lora_a','.lora_b'))})
        (adapter/'adapter_config.json').write_text(json.dumps(dict(fine_tune_type='lora',num_layers=4,lora_parameters=lora)))
        model.eval();mx.eval(model.parameters())
        prompt=mx.random.randint(0,256,(1,17));mx.eval(prompt)
        def trace(current):
            cache=make_prompt_cache(current);scores=[];sampled=[]
            for start in range(0,17,5):
                logits=current(prompt[:,start:start+5],cache=cache);mx.eval(logits)
                scores.append(logits)
            mx.random.seed(401);sampler=make_sampler(temp=.7,top_p=.9)
            for _ in range(12):
                last=logits[:,-1,:]
                token=sampler(last-mx.logsumexp(last,axis=-1,keepdims=True));mx.eval(token)
                sampled.append(token.item());logits=current(token[:,None],cache=cache)
                mx.eval(logits);scores.append(logits)
            return scores,sampled,[c.state for c in cache]
        expected,tokens,caches=trace(model)
        reports=[]
        for resident in [0,1,3]:
            loaded,_=load_model(base,lazy=True);loaded=load_adapters(loaded,str(adapter));loaded.eval()
            install_streaming(loaded,base,resident)
            actual,generated,states=trace(loaded)
            max_error=max(mx.max(mx.abs(a.astype(mx.float32)-b.astype(mx.float32))).item() for a,b in zip(expected,actual))
            differences=[]
            for aa,bb in zip(caches,states,strict=True):
                for a,b in zip(aa,bb,strict=True):
                    if isinstance(a,mx.array):
                        differences.append(mx.max(mx.abs(a.astype(mx.float32)-b.astype(mx.float32))).item())
                    else:
                        assert a==b
            cache_error=max(differences)
            assert max_error==0 and cache_error==0 and tokens==generated
            reports.append(dict(resident_layers=resident,max_logit_error=max_error,max_kv_error=cache_error,matching_sampled_tokens=len(tokens)))
        loaded,_=load_model(base,lazy=True);loaded=load_adapters(loaded,str(adapter));loaded.freeze();loaded.train()
        loaded.model.embed_tokens.eval()
        for layer in loaded.layers[:3]:layer.eval()
        for name,module in loaded.layers[3].named_modules():
            if 'lora_a' in module and 'lora_b' in module:module.unfreeze(recurse=False,keys=['lora_a','lora_b'])
        trainable_before=set(dict(tree_flatten(loaded.trainable_parameters())))
        assert trainable_before
        install_frozen_prefix(loaded,base,3)
        assert set(dict(tree_flatten(loaded.trainable_parameters())))==trainable_before
        from mlx_lm.models.base import create_attention_mask
        def prefix(current):
            h=current.model.embed_tokens(prompt);mask=create_attention_mask(h,None)
            for layer in current.layers[:3]:h=layer(h,mask);mx.eval(h)
            return h
        prefix_error=mx.max(mx.abs(prefix(model).astype(mx.float32)-prefix(loaded).astype(mx.float32))).item()
        assert prefix_error==0
        report=dict(status='pass',cases=reports,frozen_prefix_max_error=prefix_error,suffix_trainable_parameters_unchanged=True,scope='Tiny Qwen3 BF16/4bit with nonzero attention, MLP and tied-embedding LoRA. Exact prefill, incremental logits, full KV state, sampled-token and frozen-prefix parity. Actual checkpoint memory/performance requires a separate probe.')
        args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
