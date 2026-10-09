"""Single-resident-model DPO plus chosen-response NLL on Apple Silicon.

Frozen reference log probabilities are cached before any update. This is offline
preference optimization, not a reproduction of Zero's GRPO or an LLM judge.
Only a reviewed SFT checkpoint and authored preference pairs are accepted inputs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map
from mlx_lm import load
from mlx_lm.tuner.trainer import grad_checkpoint


def serialize(tok, messages, answer):
    prefix=tok.apply_chat_template(messages,add_generation_prompt=True,return_dict=False)
    full=tok.apply_chat_template(messages+[{'role':'assistant','content':answer}],return_dict=False)
    assert full[:len(prefix)]==prefix,'Training/inference prefix mismatch'
    end=next((i+1 for i in range(len(prefix),len(full)) if full[i]==tok.eos_token_id),len(full))
    return mx.array(full[:end])[None,:],len(prefix)


def log_probability(model,tokens,offset):
    logits=model(tokens[:,:-1]).astype(mx.float32)
    ce=nn.losses.cross_entropy(logits,tokens[:,1:])
    mask=mx.arange(1,tokens.shape[1])>=offset
    return -mx.where(mask,ce,0).sum(),mask.sum()


def objective(model,chosen,rejected,offset,ref_margin,beta,nll_weight):
    cp,count=log_probability(model,chosen,offset)
    rp,_=log_probability(model,rejected,offset)
    margin=cp-rp-ref_margin
    return mx.logaddexp(mx.array(0.),-beta*margin)-nll_weight*cp/count


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True)
    p.add_argument('--sft-adapter',required=True)
    p.add_argument('--pairs',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--steps',type=int,default=96)
    p.add_argument('--lr',type=float,default=2e-6)
    p.add_argument('--beta',type=float,default=.1)
    p.add_argument('--nll-weight',type=float,default=.2)
    p.add_argument('--seed',type=int,default=41)
    p.add_argument('--fresh-rank',type=int,default=0,
                   help='Positive rank trains a fresh delta around frozen SFT factors')
    args=p.parse_args()
    out=Path(args.output)
    if out.exists():raise ValueError('Fresh experiment directory required')
    out.mkdir(parents=True)
    records=[json.loads(l) for l in Path(args.pairs).read_text().splitlines()]
    assert records and all(r['review']=='author-approved' for r in records)
    assert all(not any(m['role']=='system' for m in r['messages']) for r in records)
    cfg=json.loads((Path(args.sft_adapter)/'adapter_config.json').read_text())
    if Path(cfg['model']).resolve()!=Path(args.model).resolve():
        raise ValueError('Preference optimization must use the recorded SFT base')
    if args.fresh_rank<0:raise ValueError('Fresh rank must be nonnegative')
    mx.set_memory_limit(5_600_000_000);mx.set_cache_limit(32_000_000)
    mx.set_wired_limit(mx.metal.device_info()['max_recommended_working_set_size'])
    mx.random.seed(args.seed)
    model,tok=load(args.model,adapter_path=args.sft_adapter)
    model.freeze()
    if args.fresh_rank:
        from character_delta import attach_character_delta,export_character_delta
        attach_character_delta(model,args.fresh_rank,cfg['lora_parameters']['dropout'])
        cfg['character_delta']={'frozen_parent':args.sft_adapter,
            'parent_rank':cfg['lora_parameters']['rank'],'trained_delta_rank':args.fresh_rank,
            'export':'Concatenated factors; evaluate the exported standard adapter'}
        cfg['lora_parameters']['rank']+=args.fresh_rank
        export=lambda:export_character_delta(model)
    else:
        for _,module in model.named_modules():
            if 'lora_a' in module and 'lora_b' in module:
                module.unfreeze(recurse=False,keys=['lora_a','lora_b'],strict=True)
        export=lambda:dict(tree_flatten(model.trainable_parameters()))
    leaves=tree_flatten(model.trainable_parameters())
    assert leaves and all(k.endswith(('.lora_a','.lora_b')) for k,v in leaves)
    checkpointed_types=set()
    for layer in model.layers:
        if type(layer) not in checkpointed_types:
            grad_checkpoint(layer)
            checkpointed_types.add(type(layer))
    mx.random.seed(args.seed);rng=random.Random(args.seed)
    (out/'adapter_config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    metadata=dict(vars(args),source_adapter_sha256=hashlib.sha256((Path(args.sft_adapter)/'adapters.safetensors').read_bytes()).hexdigest(),pairs_sha256=hashlib.sha256(Path(args.pairs).read_bytes()).hexdigest(),trainable_parameters=sum(v.size for _,v in leaves),objective='softplus(-beta * (policy chosen-rejected logp - frozen reference margin)) + chosen token NLL weight',reference='Same SFT checkpoint, evaluated before any preference updates. Cached probabilities; no second resident model.',reviewer='Author, not independent')
    (out/'preference_config.json').write_text(json.dumps(metadata,indent=2)+'\n')
    prepared=[];reference=[];model.eval()
    for row in records:
        chosen,offset=serialize(tok,row['messages'],row['chosen'])
        rejected,other=serialize(tok,row['messages'],row['rejected']);assert offset==other
        if max(chosen.shape[1],rejected.shape[1])>1024:raise ValueError('Pair exceeds declared context')
        cp,_=log_probability(model,chosen,offset);rp,_=log_probability(model,rejected,offset);mx.eval(cp,rp)
        prepared.append((chosen,rejected,offset,cp.item()-rp.item()))
        reference.append(dict(id=row['id'],chosen_logp=cp.item(),rejected_logp=rp.item()))
        mx.clear_cache()
    (out/'reference_logp.json').write_text(json.dumps(reference,indent=2)+'\n')
    optimizer=optim.AdamW(learning_rate=args.lr,weight_decay=.01)
    vg=nn.value_and_grad(model,objective)
    order=list(range(len(records)));start=time.monotonic()
    log=out/'preference_metrics.jsonl'
    for step in range(args.steps):
        if step%len(order)==0:rng.shuffle(order)
        index=order[step%len(order)];chosen,rejected,offset,margin=prepared[index]
        model.train();loss,grad=vg(model,chosen,rejected,offset,margin,args.beta,args.nll_weight)
        norm=mx.sqrt(sum(mx.sum(v.astype(mx.float32)**2) for _,v in tree_flatten(grad)))
        grad=tree_map(lambda v:v*mx.minimum(1.,1./(norm+1e-8)),grad)
        mx.eval(loss,norm)
        if not mx.isfinite(loss).item() or not mx.isfinite(norm).item():
            raise FloatingPointError('Nonfinite preference loss/gradient before update')
        optimizer.update(model,grad);mx.eval(model.parameters(),optimizer.state,loss,norm)
        if not mx.isfinite(loss).item():raise FloatingPointError('Nonfinite preference loss')
        row=dict(step=step+1,id=records[index]['id'],loss=loss.item(),gradient_norm=norm.item(),peak_memory_bytes=mx.get_peak_memory())
        with log.open('a') as f:f.write(json.dumps(row)+'\n')
        if (step+1)%8==0:print(json.dumps(row),flush=True)
        del grad,loss,norm;mx.clear_cache()
        if (step+1)%32==0 or step+1==args.steps:
            mx.save_safetensors(str(out/f'{step+1:07d}_adapters.safetensors'),export())
    mx.save_safetensors(str(out/'adapters.safetensors'),export())
    (out/'completion.json').write_text(json.dumps(dict(seconds=time.monotonic()-start,steps=args.steps,status='Unreviewed preference-trained candidate; evaluate unprompted behavior before acceptance'),indent=2)+'\n')

if __name__=='__main__':main()
