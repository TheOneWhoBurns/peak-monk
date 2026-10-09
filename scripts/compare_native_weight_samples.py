"""CPU-only fixed-row comparison of two verified, aligned quantized backbones.

This diagnoses weight-space proximity; it does not prove adapter transfer quality.
No full matrix or model is dequantized or modified.
"""
import json
from pathlib import Path
import struct
import numpy as np


def header(path):
    with path.open('rb') as f:
        n=struct.unpack('<Q',f.read(8))[0];data=json.loads(f.read(n))
    return data,n+8


def selected(path,metadata,base,name,rows):
    entry=metadata[name];kind=entry['dtype']
    dtype={'U32':'<u4','BF16':'<u2','F16':'<f2','F32':'<f4'}[kind]
    mapped=np.memmap(path,mode='r',dtype=dtype,offset=base+entry['data_offsets'][0],shape=tuple(entry['shape']))
    value=np.array(mapped[rows]);del mapped
    if kind=='BF16':value=(value.astype(np.uint32)<<16).view(np.float32)
    return value


def matrix(path,metadata,base,prefix,rows):
    packed=selected(path,metadata,base,prefix+'.weight',rows)
    scale=selected(path,metadata,base,prefix+'.scales',rows).astype(np.float32)
    bias=selected(path,metadata,base,prefix+'.biases',rows).astype(np.float32)
    integers=((packed[:,:,None]>>np.arange(0,32,4,dtype=np.uint32))&15).reshape(len(rows),-1).astype(np.float32)
    return integers*np.repeat(scale,32,axis=1)+np.repeat(bias,32,axis=1)


def main():
    paths=[Path('models/moshi/model.q4.safetensors'),Path('models/personaplex-edge/model.q4.safetensors')]
    headers=[header(p) for p in paths]
    prefixes=['transformer.layers.0.self_attn.in_proj','transformer.layers.15.self_attn.in_proj',
              'transformer.layers.31.gating.linear_out','text_linear']
    checks=[]
    for prefix in prefixes:
        n=headers[0][0][prefix+'.weight']['shape'][0]
        rows=np.linspace(0,n-1,64,dtype=int)
        a,b=[matrix(path,h,offset,prefix,rows).astype(np.float64) for path,(h,offset) in zip(paths,headers)]
        cosine=float(np.sum(a*b)/(np.linalg.norm(a)*np.linalg.norm(b)))
        error=float(np.linalg.norm(a-b)/np.linalg.norm(a))
        checks.append({'matrix':prefix,'rows':rows.tolist(),'shape':list(a.shape),'cosine_similarity':cosine,'relative_l2_difference':error})
    report={'scope':'Fixed evenly spaced64rows in four matrices; CPU-only affine INT4 decoding, not full-model similarity or quality evaluation',
            'models':[str(p) for p in paths],'quantization':{'bits':4,'group_size':32},'checks':checks,
            'interpretation':'High proximity supports investigating task-delta transfer, but behavior and speech quality must be measured on the actual transferred candidate.'}
    Path('results/native-weight-sample-comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps([{k:v for k,v in c.items() if k!='rows'} for c in checks],indent=2))


if __name__=='__main__':main()
