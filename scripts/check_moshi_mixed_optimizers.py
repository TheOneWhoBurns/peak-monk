"""Verify shared native adapters with independent text/audio AdamW histories."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten,tree_map
from moshi_mlx.models import Lm,config_v0_1
from moshi_training import attach_lora,loss,layer_forward
from moshi_weight_stream import MoshiParts
from moshi_text_adjoint import TextMoshiParts,disk_text_grad
from moshi_disk_adjoint import disk_episode_grad,disk_adamw_update


def directory_hash(path):
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(path.glob('*.safetensors'))}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    mx.random.seed(311)
    cfg=config_v0_1()
    cfg=replace(cfg,transformer=replace(cfg.transformer,d_model=64,num_heads=4,num_layers=2,dim_feedforward=192),
        depformer=replace(cfg.depformer,transformer=replace(cfg.depformer.transformer,d_model=32,num_heads=4,num_layers=2,dim_feedforward=96)),
        text_in_vocab_size=33,text_out_vocab_size=32,audio_vocab_size=33)
    model=Lm(cfg)
    nn.quantize(model,bits=4,group_size=32)
    reports=[]
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory)
        base=root/'base.safetensors'
        mx.save_safetensors(str(base),dict(tree_flatten(model.parameters())))
        attach_lora(model,layers=2,rank=2,scale=2.)
        for name,value in tree_flatten(model.trainable_parameters()):
            if name.endswith('.b'):
                model.load_weights([(name,mx.random.normal(value.shape)*.02)],strict=False)
        adapter=root/'adapter0.safetensors'
        mx.save_safetensors(str(adapter),dict(tree_flatten(model.trainable_parameters())))
        text=mx.concatenate([mx.array([[32]]),mx.random.randint(4,32,(1,17))],axis=1)
        mask=mx.array([[0]*3+[1]*7+[0]*3+[1]*5])
        audio=mx.random.randint(0,31,(1,17,14))
        audio[:,0,1:3]=mx.array([0,3])
        audio[:,1:4,-1]=-1
        optimizers={kind:optim.AdamW(learning_rate=1e-4,weight_decay=.01) for kind in ('text','audio')}
        counters=dict(text=0,audio=0)
        for kind in counters:(root/kind).mkdir()

        def text_loss(current):
            h=current.text_emb(text[:,:-1])
            for layer in current.transformer.layers:h=layer_forward(layer,h)[0]
            h=current.out_norm(h)
            total=mx.array(0.,dtype=mx.float32)
            for offset in range(0,h.shape[1],4):
                end=min(offset+4,h.shape[1])
                logits=current.text_linear(h[:,offset:end]).astype(mx.float32)
                ce=nn.losses.cross_entropy(logits,text[:,offset+1:end+1])
                total=total+(ce*mask[:,offset+1:end+1]).sum()/mask[:,1:].sum()
            return total

        for step,kind in enumerate(('text','audio','text'),1):
            objective=text_loss if kind=='text' else lambda m:loss(m,audio,text_padding_weight=.05,checkpoint_layers=False)
            reference,gradient=nn.value_and_grad(model,objective)(model)
            mx.eval(reference,gradient)
            if kind=='text':
                parts=TextMoshiParts(base,cfg,adapter,rank=2,scale=2.)
                actual=disk_text_grad(parts,text,mask,root/f'gradient{step}',head_tokens=4)
            else:
                parts=MoshiParts(base,cfg,adapter,rank=2,scale=2.)
                actual=disk_episode_grad(parts,audio,root/f'gradient{step}',head_frames=4,padding_weight=.05)
            expected=dict(tree_flatten(gradient));flat={}
            for row in actual['gradients']:
                prefix='text_linear.' if row['part']=='heads' else f"transformer.layers.{row['part']}."
                flat.update({prefix+k:v for k,v in mx.load(row['path']).items()})
            assert set(flat)==set(expected)
            gradient_error=max(mx.max(mx.abs(v-flat[k])).item() for k,v in expected.items())
            norm=mx.sqrt(sum(mx.sum(v.astype(mx.float32)**2) for v in expected.values()))
            scale=mx.minimum(1.,.3/(norm+1e-8))
            optimizers[kind].update(model,tree_map(lambda g:g*scale,gradient))
            mx.eval(model.parameters(),optimizers[kind].state)
            counters[kind]+=1
            other='audio' if kind=='text' else 'text'
            before=directory_hash(root/other)
            output=root/f'adapter{step}.safetensors'
            disk_adamw_update(parts,actual,root/kind,output,clip_norm=.3,expected_step=counters[kind])
            assert directory_hash(root/other)==before
            saved=mx.load(str(output))
            parameter_error=max(mx.max(mx.abs(v-saved[k])).item() for k,v in tree_flatten(model.trainable_parameters()))
            row=dict(step=step,modality=kind,loss_error=abs(reference.item()-actual['loss']),
                gradient_error=gradient_error,adapter_error=parameter_error,optimizer_steps=dict(counters),inactive_optimizer_unchanged=True)
            assert row['loss_error']<1e-5 and gradient_error<1e-5 and parameter_error<1e-6,row
            reports.append(row);adapter=output
    report=dict(status='pass',cases=reports,
        scope='Tiny FP32/quantized4-bit model, alternating text/audio/text objectives, nonzero shared adapters, native initial input, masked text, audio pad targets, separate AdamW states and clipped updates. No actual-model feasibility or transfer-quality claim.')
    args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
