"""Complete causal forward context with truncated gradients between windows.

Carried attention keys/values are detached. Weights remain unchanged throughout
an episode, gradients accumulate, and one optimizer update follows the episode.
The summed loss matches full teacher forcing; its gradient deliberately does NOT
include credit through earlier windows. This is truncated BPTT, not exact full
sequence backpropagation.
"""
from functools import partial
import mlx.core as mx
import mlx.nn as nn
from mlx.nn.utils import checkpoint
from mlx.utils import tree_map
from moshi_training import embed,layer_forward,depth_forward


def temporal_chunk(model, inputs, previous=None, checkpoint_layers=True):
    xs=embed(model.text_emb,inputs[:,0])
    for i,embedding in enumerate(model.audio_embs):
        xs=xs+embed(embedding,inputs[:,i+1])
    if previous is None:previous=[None]*len(model.transformer.layers)
    state=[]
    for layer,past in zip(model.transformer.layers,previous):
        fn=checkpoint(layer,partial(layer_forward,layer)) if checkpoint_layers else partial(layer_forward,layer)
        xs,kv=fn(xs,past);state.append(kv)
    xs=model.out_norm(xs)
    return xs,model.text_linear(xs),state


def normalizers(model, targets, padding_weight):
    text=targets[:,0]
    valid=(text>=0)&(text<model.cfg.text_out_vocab_size)
    tw=mx.where((text==0)|(text==3),padding_weight,1.)*valid
    audio=targets[:,1:1+model.dep_q]
    valid=(audio>=0)&(audio<model.cfg.audio_vocab_size-1)
    aw=mx.array([100.]+[1.]*(model.dep_q-1))[None,:,None]*valid
    return max(tw.sum().item(),1.),max(aw.sum().item(),1.)


def chunk_loss(model, inputs, targets, previous, denominators, padding_weight=.05):
    hidden,logits,state=temporal_chunk(model,inputs,previous)
    target=targets[:,0];valid=(target>=0)&(target<model.cfg.text_out_vocab_size)
    weight=mx.where((target==0)|(target==3),padding_weight,1.)*valid
    ce=nn.losses.cross_entropy(logits.astype(mx.float32),mx.clip(target,0,model.cfg.text_out_vocab_size-1))
    value=(ce*weight).sum()/denominators[0]
    logits=depth_forward(model,hidden,targets,checkpoint_layers=True)
    target=targets[:,1:1+model.dep_q];valid=(target>=0)&(target<model.cfg.audio_vocab_size-1)
    weight=mx.array([100.]+[1.]*(model.dep_q-1))[None,:,None]*valid
    ce=nn.losses.cross_entropy(logits.astype(mx.float32),mx.clip(target,0,model.cfg.audio_vocab_size-2))
    value=value+(ce*weight).sum()/denominators[1]
    return value,state


def episode_grad(model, streams, window=64, padding_weight=.05):
    denominator=normalizers(model,streams[:,:,1:],padding_weight)
    value_grad=nn.value_and_grad(model,partial(chunk_loss,padding_weight=padding_weight))
    state=None;accumulated=None;total=0.;frames=streams.shape[-1]-1
    for start in range(0,frames,window):
        end=min(start+window,frames)
        (value,state),grad=value_grad(model,streams[:,:,start:end],streams[:,:,start+1:end+1],state,denominator)
        accumulated=grad if accumulated is None else tree_map(lambda a,b:a+b,accumulated,grad)
        state=tree_map(mx.stop_gradient,state)
        mx.eval(value,state,accumulated)
        total+=value.item()
        del value,grad
        mx.clear_cache()
    return mx.array(total),accumulated
