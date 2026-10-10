"""Read saved mixed-training weights and optimizer states without loading MLX."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import numpy as np


def read(path):
    with path.open('rb') as f:
        size=struct.unpack('<Q',f.read(8))[0]
        if size>16000000:raise ValueError('Unexpected header size')
        header=json.loads(f.read(size));raw=f.read()
    values={}
    types={'F32':'<f4','F16':'<f2','I32':'<i4','I64':'<i8','U32':'<u4','U64':'<u8','BF16':'<u2'}
    for key,item in header.items():
        if key=='__metadata__':continue
        lo,hi=item['data_offsets']
        if not 0<=lo<=hi<=len(raw):raise ValueError('Invalid tensor range')
        value=np.frombuffer(raw[lo:hi],dtype=types[item['dtype']]).reshape(item['shape'])
        if item['dtype']=='BF16':value=(value.astype(np.uint32)<<16).view(np.float32)
        values[key]=value
    return values


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--preserve',type=Path);args=p.parse_args()
    root=args.root;metadata=json.loads((root/'metadata.json').read_text());cfg=metadata['config'];latest=json.loads((root/'latest.json').read_text());folder=root/latest['directory'];candidate=folder/'candidate.safetensors'
    digest=hashlib.sha256(candidate.read_bytes()).hexdigest()
    assert digest==latest['adapter_sha256']
    weights=read(candidate);parent=read(Path(cfg['parent']))
    assert len(weights)==130 and set(weights)==set(parent)
    assert all(np.isfinite(a).all() and a.shape==parent[k].shape for k,a in weights.items())
    changed=sum(not np.array_equal(weights[k],parent[k]) for k in weights)
    states={}
    for mode in ('text','audio'):
        files=sorted((folder/('optimizer-'+mode)).glob('*.safetensors'))
        assert len(files)==(33 if latest['optimizer_steps'][mode] else 0)
        steps=[]
        for path in files:
            values=read(path);assert all(np.isfinite(a).all() for a in values.values())
            actual=int(values['step']);assert actual==latest['optimizer_steps'][mode];steps.append(actual)
        states[mode]=dict(files=len(files),steps=sorted(set(steps)),all_finite=True)
    if args.preserve:
        if args.preserve.exists():assert hashlib.sha256(args.preserve.read_bytes()).hexdigest()==digest
        else:os.link(candidate,args.preserve)
    events=[json.loads(x) for x in (root/'runtime.jsonl').read_text().splitlines()]
    updates=[x for x in events if x['phase']=='mixed_update_committed']
    assert len(updates)==latest['step'] and [x['step'] for x in updates]==list(range(1,latest['step']+1))
    assert all(x['metrics']['memory_pressure_level']==1 for x in updates)
    assert all(x['modality']==metadata['schedule'][i]['modality'] and x['id']==metadata['schedule'][i]['id'] for i,x in enumerate(updates))
    report=dict(status='pass',scope='Actual saved state/export inspection; no generation or character-transfer quality claim.',latest=latest,
        saved_adapter=str(args.preserve or candidate),adapter_tensors=len(weights),changed_tensors=changed,all_finite=True,
        optimizer_states=states,updates=updates,parent_validation=json.loads((root/'validation-0000.json').read_text()),
        segment=json.loads((root/'segment-completion.json').read_text()),inspector_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('status','latest','adapter_tensors','changed_tensors','all_finite','optimizer_states')},indent=2))


if __name__=='__main__':main()
