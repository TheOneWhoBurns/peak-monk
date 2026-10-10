"""Read only selected safetensor ranges; never load the full Moshi backbone."""
import json
import math
import struct
from pathlib import Path
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from moshi_mlx.models.lm import ScaledEmbedding,DepFormer
from moshi_mlx.modules.transformer import TransformerLayer
from moshi_training import LoRALinear,embed

class TensorRanges:
    def __init__(self,path):
        self.path=Path(path)
        with self.path.open('rb') as f:
            size=struct.unpack('<Q',f.read(8))[0]
            if size>16_000_000:raise ValueError('Unexpected safetensor header')
            self.header=json.loads(f.read(size))
        self.offset=8+size
    def load(self,prefixes,strip=''):
        arrays={}
        types={'BF16':(np.uint16,mx.bfloat16),'F32':(np.float32,mx.float32),'U32':(np.uint32,mx.uint32)}
        for key,meta in self.header.items():
            if key=='__metadata__' or not key.startswith(tuple(prefixes)):continue
            if not key.startswith(strip):raise ValueError('Incorrect prefix remapping')
            dtype,target=types[meta['dtype']]
            begin,end=meta['data_offsets'];size=math.prod(meta['shape'])*np.dtype(dtype).itemsize
            if end-begin!=size or self.offset+end>self.path.stat().st_size:raise ValueError('Invalid tensor range')
            raw=np.memmap(self.path,mode='r',dtype=dtype,offset=self.offset+begin,shape=tuple(meta['shape']))
            value=mx.array(raw)
            if meta['dtype']=='BF16':value=value.view(target)
            mx.eval(value);del raw
            arrays[key[len(strip):]]=value
        if not arrays:raise ValueError('No matching tensors')
        return arrays

class Inputs(nn.Module):
    def __init__(self,cfg):
        super().__init__();dim=cfg.transformer.d_model
        self.text_emb=ScaledEmbedding(cfg.text_in_vocab_size,dim,demux_second_stream=cfg.demux_second_stream)
        self.audio_embs=[ScaledEmbedding(cfg.audio_vocab_size,dim) for _ in range(cfg.audio_codebooks)]
    def __call__(self,tokens):
        h=embed(self.text_emb,tokens[:,0])
        for i,e in enumerate(self.audio_embs):h=h+embed(e,tokens[:,i+1])
        return h

class Heads(nn.Module):
    def __init__(self,cfg):
        super().__init__();self.cfg=cfg;dim=cfg.transformer.d_model
        self.depformer=DepFormer(cfg)
        self.out_norm=nn.RMSNorm(dim,1e-8) if cfg.transformer.norm=='rms_norm' else nn.LayerNorm(dim,1e-5)
        self.text_linear=nn.Linear(dim,cfg.text_out_vocab_size,bias=False)
    @property
    def dep_q(self):return self.cfg.depformer.num_slices

class MoshiParts:
    def __init__(self,weights,cfg,adapter=None,rank=8,scale=2.):
        self.weights=TensorRanges(weights);self.cfg=cfg
        self.adapter=TensorRanges(adapter) if adapter else None
        self.rank=rank;self.scale=scale
    def load(self,part):
        if part=='inputs':module=Inputs(self.cfg);prefixes=['text_emb.','audio_embs.'];strip=''
        elif part=='heads':module=Heads(self.cfg);prefixes=['depformer.','out_norm.','text_linear.'];strip=''
        elif isinstance(part,int) and 0<=part<self.cfg.transformer.num_layers:
            module=TransformerLayer(self.cfg.transformer);strip=f'transformer.layers.{part}.';prefixes=[strip]
        else:raise ValueError('Unknown model part')
        module.set_dtype(mx.bfloat16);nn.quantize(module,bits=4,group_size=32)
        weights=self.weights.load(prefixes,strip=strip)
        module.load_weights(list(weights.items()),strict=True);del weights
        module.freeze()
        if self.adapter and part!='inputs':
            if part=='heads':module.text_linear=LoRALinear(module.text_linear,self.rank,self.scale);prefixes=['text_linear.']
            else:
                module.self_attn.out_proj=LoRALinear(module.self_attn.out_proj,self.rank,self.scale)
                module.gating.linear_out=LoRALinear(module.gating.linear_out,self.rank,self.scale)
            weights=self.adapter.load(prefixes,strip=strip)
            from mlx.utils import tree_flatten
            if set(weights)!=set(dict(tree_flatten(module.trainable_parameters()))):raise ValueError('Adapter part mismatch')
            module.load_weights(list(weights.items()),strict=False);del weights
        mx.eval(module.parameters())
        return module
