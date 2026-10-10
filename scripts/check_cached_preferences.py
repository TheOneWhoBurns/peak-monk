"""Numerical proof for frozen-prefix DPO, masked EOS targets, and retention."""
import argparse
import json
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map
from mlx_lm.models.qwen3 import Model, ModelArgs
from mlx_lm.models.base import create_attention_mask
from mlx_lm.tuner.lora import LoRAEmbedding, LoRALinear
from repair_preference_grad import repair_preference_grad, masked_reply_score
from staged_character_grad import staged_character_grad
from qwen_rng_checkpoint import qwen_rng_checkpoint


def make_model():
    model = Model(ModelArgs(model_type='qwen3', hidden_size=32, num_hidden_layers=2,
        intermediate_size=64, num_attention_heads=4, rms_norm_eps=1e-6, vocab_size=64,
        num_key_value_heads=2, max_position_embeddings=256, head_dim=8,
        tie_word_embeddings=True, rope_theta=1e6))
    nn.quantize(model, bits=4, group_size=32)
    model.model.embed_tokens = LoRAEmbedding.from_base(model.model.embed_tokens, r=2, scale=2., dropout=.2)
    for layer in model.layers:
        for parent, names in [(layer.self_attn, ['q_proj','k_proj','v_proj','o_proj']),
                              (layer.mlp, ['gate_proj','up_proj','down_proj'])]:
            for name in names:
                setattr(parent, name, LoRALinear.from_base(getattr(parent,name), r=2, scale=2., dropout=.2))
    for name, value in tree_flatten(model.trainable_parameters()):
        if name.endswith('lora_b'):
            model.load_weights([(name,mx.random.normal(value.shape)*.02)], strict=False)
    model.freeze()
    for name, module in model.layers[1].named_modules():
        if 'lora_a' in module:
            module.unfreeze(recurse=False, keys=['lora_a','lora_b'], strict=True)
    model.eval()
    return model


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    mx.set_default_device(mx.cpu);mx.random.seed(277)
    full=make_model();suffix=make_model()
    suffix.load_weights(tree_flatten(full.parameters()),strict=True)
    chosen=(mx.array([[1,2,3,4,5,6,7,8]]),mx.array([[False]*3+[True]*4+[False]]))
    rejected=(mx.array([[1,2,3,9,10,11]]),mx.array([[False]*3+[True]*2+[False]]))
    replay=(mx.array([[12,13,14,15,16,17,18,19,20]]),
            mx.array([[False,False,True,True,False,False,True,True,False]]))
    features=[]
    for tokens,mask in [chosen,rejected,replay]:
        h=suffix.model.embed_tokens(tokens[:,:-1]);h=suffix.layers[0](h,create_attention_mask(h,None))
        mx.eval(h);features.append(mx.stop_gradient(h))
    suffix.model.layers=suffix.layers[1:]
    reference=masked_reply_score(full,*chosen,chunk_size=3)['logp']-masked_reply_score(full,*rejected,chunk_size=3)['logp']
    score_errors=[]
    for pair,feat in zip([chosen,rejected,replay],features):
        a=masked_reply_score(full,*pair,chunk_size=3);b=masked_reply_score(suffix,*pair,features=feat,chunk_size=3)
        score_errors.append(abs(a['logp']-b['logp']))
    qwen_rng_checkpoint(suffix.layers[0])

    def nll(current,pair):
        tokens,mask=pair
        logits=current(tokens[:,:-1]).astype(mx.float32)
        ce=nn.losses.cross_entropy(logits,tokens[:,1:])
        return mx.where(mask[:,1:],ce,0).sum()

    def direct(current):
        c=nll(current,chosen);r=nll(current,rejected)
        return mx.logaddexp(mx.array(0.),.1*(c-r+reference))+.2*c/chosen[1][:,1:].sum()+.7*nll(current,replay)/replay[1][:,1:].sum()

    fo=optim.AdamW(learning_rate=1e-4,weight_decay=0.);so=optim.AdamW(learning_rate=1e-4,weight_decay=0.)
    reports=[]
    for step in (1,2):
        expected,fg=nn.value_and_grad(full,direct)(full)
        value,sg,diagnostic=repair_preference_grad(suffix,chosen,rejected,reference,
            chosen_features=features[0],rejected_features=features[1],chunk_size=3)
        (retention,_),rg=staged_character_grad(suffix,*replay,input_embeddings=features[2],chunk_size=3)
        value=value+.7*retention;sg=tree_map(lambda a,b:a+.7*b,sg,rg)
        mx.eval(expected,fg,value,sg)
        expected_grad=dict(tree_flatten(fg))
        error=max(mx.max(mx.abs(g-expected_grad[k.replace('layers.0.','layers.1.')])).item() for k,g in tree_flatten(sg))
        def clip(gradient):
            norm=mx.sqrt(sum(mx.sum(v.astype(mx.float32)**2) for _,v in tree_flatten(gradient)))
            return tree_map(lambda g:g*mx.minimum(1.,1./(norm+1e-8)),gradient)
        fo.update(full,clip(fg));so.update(suffix,clip(sg));mx.eval(full.parameters(),suffix.parameters(),fo.state,so.state)
        expected_params=dict(tree_flatten(full.trainable_parameters()))
        adapter_error=max(mx.max(mx.abs(v-expected_params[k.replace('layers.0.','layers.1.')])).item() for k,v in tree_flatten(suffix.trainable_parameters()))
        row=dict(step=step,loss_error=abs(expected.item()-value.item()),gradient_error=error,adapter_error=adapter_error,diagnostic=diagnostic)
        assert row['loss_error']<1e-5 and error<1e-5 and adapter_error<1e-6,row
        reports.append(row)
    assert max(score_errors)<1e-5,score_errors
    report=dict(status='pass',score_errors=score_errors,updates=reports,
        scope='Tiny quantized Qwen3, nonzero LoRA, frozen embedding/lower layer, unequal candidate lengths, explicit trailing-token exclusion, ragged head chunks, retention and two clipped AdamW updates. Full model direct objective agrees with cached suffix. No actual-model resource or quality claim.')
    args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
