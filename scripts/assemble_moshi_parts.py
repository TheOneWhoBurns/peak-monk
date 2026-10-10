"""Assemble inference weights without materializing a second full weight copy."""
import gc
import mlx.core as mx
from moshi_mlx.models import Lm

def assemble(parts,boundary=lambda **kw:None):
    if parts.cfg.extra_heads_num_heads or parts.cfg.conditioners:
        raise ValueError('Assembly only supports the audited unconditional Moshi model')
    model=Lm(parts.cfg)
    # Release unevaluated initializers before reading any pretrained tensor.
    model.transformer.layers=[];model.depformer=None;model.out_norm=None
    model.text_linear=None;model.text_emb=None;model.audio_embs=[]
    gc.collect()
    inputs=parts.load('inputs');model.text_emb=inputs.text_emb;model.audio_embs=inputs.audio_embs
    del inputs;gc.collect();boundary(phase='inference_inputs_loaded')
    for i in range(parts.cfg.transformer.num_layers):
        model.transformer.layers.append(parts.load(i))
        boundary(phase='inference_layer_loaded',layer=i)
    heads=parts.load('heads');model.depformer=heads.depformer;model.out_norm=heads.out_norm;model.text_linear=heads.text_linear
    del heads;gc.collect();model.eval();mx.eval(model.parameters());boundary(phase='inference_heads_loaded')
    return model

import mlx.nn as nn

class StreamingTemporal(nn.Module):
    def __init__(self,parts,boundary):
        super().__init__();self.cfg=parts.cfg.transformer;self._parts=parts;self._boundary=boundary
    def __call__(self,xs,cache,cross_attention_src=None):
        if cross_attention_src is not None:raise ValueError('Unconditional model only')
        for i in range(self.cfg.num_layers):
            # Module initializers consume random keys even though their weights
            # are immediately replaced. Preserve the native sampling stream.
            state=list(mx.random.state);mx.eval(state)
            try:layer=self._parts.load(i)
            finally:mx.random.state.clear();mx.random.state.extend(state)
            xs=layer(xs,cache=cache[i]);mx.eval(xs)
            del layer
            if i%8==7 or i+1==self.cfg.num_layers:
                self._boundary(phase='streamed_inference_layer',layer=i)
        return xs


def assemble_streamed(parts,boundary=lambda **kw:None):
    if parts.cfg.extra_heads_num_heads or parts.cfg.conditioners:raise ValueError('Unconditional model only')
    model=Lm(parts.cfg);model.transformer=StreamingTemporal(parts,boundary)
    model.depformer=None;model.out_norm=None;model.text_linear=None;model.text_emb=None;model.audio_embs=[]
    gc.collect()
    inputs=parts.load('inputs');model.text_emb=inputs.text_emb;model.audio_embs=inputs.audio_embs
    del inputs;gc.collect();boundary(phase='inference_inputs_loaded')
    heads=parts.load('heads');model.depformer=heads.depformer;model.out_norm=heads.out_norm;model.text_linear=heads.text_linear
    del heads;gc.collect();model.eval();mx.eval(model.parameters());boundary(phase='inference_heads_loaded')
    return model
