"""Teacher-forced diagnostics with one temporal model part resident at a time."""
import gc
import mlx.core as mx
import mlx.nn as nn
from moshi_training import layer_forward,depth_forward

def disk_metrics(parts,streams,head_frames=8,boundary=lambda **kw:None):
    inputs=parts.load('inputs');h=inputs(streams[:,:,:-1]);mx.eval(h);del inputs;gc.collect();boundary(phase='diagnostic_inputs')
    for i in range(parts.cfg.transformer.num_layers):
        layer=parts.load(i);h=layer_forward(layer,h)[0];mx.eval(h);del layer;gc.collect();boundary(phase='diagnostic_layer',layer=i)
    heads=parts.load('heads');h=heads.out_norm(h);mx.eval(h)
    targets=streams[:,:,1:]
    groups={k:dict(tokens=0,ce_sum=0.,top1_matches=0) for k in ['spoken','silence_pad','end_pad']}
    audio=[dict(tokens=0,ce_sum=0.,top1_matches=0) for _ in range(heads.dep_q)]
    def accumulate(dst,ce,prediction,target,mask):
        dst['tokens']+=int(mask.sum().item());dst['ce_sum']+=float(mx.where(mask,ce,0).sum().item())
        dst['top1_matches']+=int(((prediction==target)&mask).sum().item())
    for start in range(0,h.shape[1],head_frames):
        end=min(start+head_frames,h.shape[1]);y=targets[:,:,start:end];hidden=h[:,start:end]
        target=y[:,0];valid=(target>=0)&(target<parts.cfg.text_out_vocab_size)
        logits=heads.text_linear(hidden).astype(mx.float32)
        ce=nn.losses.cross_entropy(logits,mx.clip(target,0,parts.cfg.text_out_vocab_size-1));pred=mx.argmax(logits,axis=-1)
        for name,mask in [('spoken',valid&(target!=0)&(target!=3)),('silence_pad',valid&(target==3)),('end_pad',valid&(target==0))]:accumulate(groups[name],ce,pred,target,mask)
        logits=depth_forward(heads,hidden,y).astype(mx.float32)
        target=y[:,1:1+heads.dep_q];valid=(target>=0)&(target<parts.cfg.audio_vocab_size-1)
        ce=nn.losses.cross_entropy(logits,mx.clip(target,0,parts.cfg.audio_vocab_size-2));pred=mx.argmax(logits,axis=-1)
        for i in range(heads.dep_q):accumulate(audio[i],ce[:,i],pred[:,i],target[:,i],valid[:,i])
        del logits,ce,pred,target,valid,hidden,y,mask
        boundary(phase='diagnostic_heads',frame=end)
    def finish(g):
        return dict(g,mean_ce=g['ce_sum']/g['tokens'] if g['tokens'] else None,top1_accuracy=g['top1_matches']/g['tokens'] if g['tokens'] else None)
    groups={k:finish(v) for k,v in groups.items()};audio=[finish(v) for v in audio]
    audio_ce=sum(g['ce_sum']*(100 if i==0 else 1) for i,g in enumerate(audio))/max(1,sum(g['tokens']*(100 if i==0 else 1) for i,g in enumerate(audio)))
    joint={}
    for weight in [.05,.5]:
        numerator=sum(g['ce_sum']*(1 if k=='spoken' else weight) for k,g in groups.items())
        denominator=sum(g['tokens']*(1 if k=='spoken' else weight) for k,g in groups.items())
        joint[str(weight)]=numerator/max(1,denominator)+audio_ce
    del heads,h;gc.collect();boundary(phase='diagnostic_complete')
    return dict(text_groups=groups,audio_codebooks=audio,semantic_weighted_audio_ce=audio_ce,joint_ce_by_padding_weight=joint)
