"""Exact frozen-prefix/suffix split for memory-efficient Moshi adaptation.

Cache inputs to the trainable suffix, not final model outputs. With identical
teacher-forced streams and frozen prefix weights, this preserves the training
objective while allowing the 28-30 frozen temporal layers to be unloaded.
"""
import re
import mlx.core as mx
import mlx.nn as nn
from moshi_mlx.models.lm import Lm, config_v0_1
from moshi_training import embed, layer_forward, depth_forward


class Prefix(nn.Module):
    def __init__(self, model, suffix_layers):
        super().__init__()
        self.text_emb = model.text_emb
        self.audio_embs = model.audio_embs
        self.layers = model.transformer.layers[:-suffix_layers]

    def __call__(self, tokens, materialize=False):
        xs = embed(self.text_emb, tokens[:, 0])
        for i, embedding in enumerate(self.audio_embs):
            xs = xs + embed(embedding, tokens[:, i+1])
        for layer in self.layers:
            xs, _ = layer_forward(layer, xs)
            if materialize:
                mx.eval(xs)
                mx.metal.clear_cache()
        return xs


class Suffix(nn.Module):
    def __init__(self, model, suffix_layers):
        super().__init__()
        self.cfg = model.cfg
        self.transformer = model.transformer
        self.transformer.layers = self.transformer.layers[-suffix_layers:]
        self.depformer = model.depformer
        self.out_norm = model.out_norm
        self.text_linear = model.text_linear

    @property
    def dep_q(self):
        return self.cfg.depformer.num_slices

    def __call__(self, hidden):
        xs = hidden
        for layer in self.transformer.layers:
            xs, _ = layer_forward(layer, xs)
        xs = self.out_norm(xs)
        return xs, self.text_linear(xs)


def load_part(weights, suffix_layers, part):
    if not 1 <= suffix_layers < 32:
        raise ValueError('Expected 1-31 trainable suffix layers')
    model = Lm(config_v0_1())
    model.set_dtype(mx.bfloat16)
    selected = Prefix(model, suffix_layers) if part=='prefix' else Suffix(model, suffix_layers)
    del model
    nn.quantize(selected, bits=4, group_size=32)
    weights = mx.load(str(weights))
    mapped = []
    start = 32-suffix_layers
    for key, value in weights.items():
        match = re.match(r'transformer\.layers\.(\d+)\.(.*)', key)
        if part=='prefix':
            if key.startswith(('text_emb.', 'audio_embs.')):
                mapped.append((key,value))
            elif match and int(match[1])<start:
                mapped.append((f'layers.{match[1]}.{match[2]}',value))
        else:
            if key.startswith(('depformer.', 'out_norm.', 'text_linear.')):
                mapped.append((key,value))
            elif match and int(match[1])>=start:
                mapped.append((f'transformer.layers.{int(match[1])-start}.{match[2]}',value))
    selected.load_weights(mapped, strict=True)
    del weights, mapped
    mx.eval(selected.parameters())
    selected.freeze()
    return selected


def cached_loss(model, hidden, targets, audio_weight=1.0, text_padding_weight=0.5):
    out, text_logits = model(hidden)
    text_target = targets[:, 0]
    valid = (text_target>=0)&(text_target<model.cfg.text_out_vocab_size)
    tw = mx.where((text_target==0)|(text_target==3),text_padding_weight,1.0)*valid
    ce = nn.losses.cross_entropy(text_logits.astype(mx.float32),mx.clip(text_target,0,model.cfg.text_out_vocab_size-1))
    text_loss = (ce*tw).sum()/mx.maximum(tw.sum(),1)
    if audio_weight==0: return text_loss
    logits = depth_forward(model,out,targets)
    target = targets[:,1:1+model.dep_q]
    valid = (target>=0)&(target<model.cfg.audio_vocab_size-1)
    weights = mx.array([100.0]+[1.0]*(model.dep_q-1))[None,:,None]*valid
    ce = nn.losses.cross_entropy(logits.astype(mx.float32),mx.clip(target,0,model.cfg.audio_vocab_size-2))
    return text_loss+audio_weight*(ce*weights).sum()/mx.maximum(weights.sum(),1)


def delay_streams(raw, cfg):
    """raw=[B,17,T] -> delayed streams with leading initial frame, as upstream."""
    b, k, t = raw.shape
    assert k==1+cfg.audio_codebooks
    output=[]
    for i, delay in enumerate([0]+cfg.audio_delays):
        initial=cfg.text_out_vocab_size if i==0 else cfg.audio_padding_token
        prefix=mx.full((b,1+delay),initial,dtype=mx.int32)
        output.append(mx.concatenate([prefix,raw[:,i,:max(0,t-delay)]],axis=-1)[:,:t+1])
    return mx.stack(output,axis=1)


def export_adapters(model, suffix_layers, path):
    from mlx.utils import tree_flatten
    mapped={}
    for key,value in tree_flatten(model.trainable_parameters()):
        match=re.match(r'transformer\.layers\.(\d+)\.(.*)',key)
        if match: key=f'transformer.layers.{32-suffix_layers+int(match[1])}.{match[2]}'
        mapped[key]=value
    mx.save_safetensors(str(path),mapped)
