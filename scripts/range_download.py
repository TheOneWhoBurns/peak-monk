"""Resume this pinned model over four HTTP ranges; verify upstream LFS SHA256."""
import concurrent.futures
import hashlib
import os
from pathlib import Path
import shutil
import urllib.request
import argparse

p=argparse.ArgumentParser()
p.add_argument('--repo',default='johnlakness/Qwen3-1.7B-Base-mlx-4Bit')
p.add_argument('--revision',default='ccbac499d70a93105c0e96618dd40564f577afa2')
p.add_argument('--sha256',default='0dfefefebd85dd75e4c42330e27da8b628f36c9198ca873b628710e9d4139db0')
p.add_argument('--size',type=int,default=968080210)
p.add_argument('--destination',default='models/base')
p.add_argument('--filename',default='model.safetensors')
args=p.parse_args()
root=Path(__file__).resolve().parents[1]/args.destination
root.mkdir(parents=True,exist_ok=True)
sha=args.sha256
size=args.size
url=f'https://huggingface.co/{args.repo}/resolve/{args.revision}/{args.filename}'
dest=root/args.filename
if dest.exists():
    raise SystemExit('Destination exists; refusing overwrite')
partials=list((root/'.cache/huggingface/download').glob(f'*.{sha}.*.incomplete'))
prefix=max(partials,key=lambda p:p.stat().st_size) if partials else None
if prefix:
    shutil.copyfile(prefix,dest)
else:
    dest.touch()
start=dest.stat().st_size
fd=os.open(dest,os.O_RDWR)
def fetch(bounds):
    a,b=bounds
    for attempt in range(3):
        try:
            req=urllib.request.Request(url,headers={'Range':f'bytes={a}-{b}'})
            with urllib.request.urlopen(req,timeout=60) as r:
                assert r.status==206 and r.headers['Content-Range']==f'bytes {a}-{b}/{size}'
                pos=a
                while pos<=b:
                    block=r.read(min(1024*1024,b-pos+1))
                    if not block: raise IOError('Truncated range')
                    wrote=os.pwrite(fd,block,pos)
                    if wrote!=len(block): raise IOError('Short local write')
                    pos+=len(block)
            print(f'completed range {a}-{b}',flush=True)
            return
        except Exception:
            if attempt==2: raise
chunk=32*1024*1024
try:
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(fetch,[(a,min(a+chunk-1,size-1)) for a in range(start,size,chunk)]))
finally:
    os.close(fd)
with dest.open('rb') as f:
    actual=hashlib.file_digest(f,'sha256').hexdigest()
assert actual==sha, (actual,sha)
# Teach HF's local cache that the independently verified artifact is complete.
metadata=root/f'.cache/huggingface/download/{args.filename}.metadata'
metadata.parent.mkdir(parents=True,exist_ok=True)
metadata.write_text(args.revision+'\n'+sha+'\n'+str(__import__('time').time())+'\n')
print('Verified upstream SHA256:',actual,flush=True)
