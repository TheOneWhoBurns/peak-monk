"""Resume this pinned model over four HTTP ranges; verify upstream LFS SHA256."""
import concurrent.futures
import hashlib
import os
from pathlib import Path
import shutil
import urllib.request
import urllib.parse
import argparse
import re
import threading
import time
import urllib.error

p=argparse.ArgumentParser()
p.add_argument('--repo',default='johnlakness/Qwen3-1.7B-Base-mlx-4Bit')
p.add_argument('--revision',default='ccbac499d70a93105c0e96618dd40564f577afa2')
p.add_argument('--sha256',default='0dfefefebd85dd75e4c42330e27da8b628f36c9198ca873b628710e9d4139db0')
p.add_argument('--size',type=int,default=968080210)
p.add_argument('--destination',default='models/base')
p.add_argument('--filename',default='model.safetensors')
p.add_argument('--resume-log',type=Path)
p.add_argument('--workers',type=int,default=4)
p.add_argument('--chunk-mib',type=int,default=32)
p.add_argument('--reuse-signed-url',action='store_true',help='Resolve the public artifact CDN once instead of requesting a new redirect per range')
args=p.parse_args()
if not re.fullmatch(r'[0-9a-f]{64}',args.sha256):
    raise SystemExit('Expected a 64-character lowercase SHA256 digest')
root=Path(__file__).resolve().parents[1]/args.destination
root.mkdir(parents=True,exist_ok=True)
sha=args.sha256
size=args.size
origin_url=f'https://huggingface.co/{args.repo}/resolve/{args.revision}/{args.filename}'
url=origin_url
url_lock=threading.Lock()
if args.reuse_signed_url:
    with urllib.request.urlopen(urllib.request.Request(url,method='HEAD'),timeout=60) as response:
        url=response.url
    # The signed URL is an ephemeral artifact capability; keep it out of logs.
    print('Resolved artifact host:',urllib.parse.urlparse(url).hostname,flush=True)
dest=root/args.filename
dest.parent.mkdir(parents=True,exist_ok=True)
if dest.exists() and not args.resume_log:
    raise SystemExit('Destination exists; refusing overwrite')
partials=list((root/'.cache/huggingface/download').glob(f'*.{sha}.*.incomplete'))
prefix=max(partials,key=lambda p:p.stat().st_size) if partials else None
completed=[]
if args.resume_log:
    if not dest.exists(): raise SystemExit('Resume requires existing partial destination')
    completed=[tuple(map(int,m)) for m in re.findall(r'completed range (\d+)-(\d+)',args.resume_log.read_text())]
elif prefix:
    shutil.copyfile(prefix,dest)
else:
    dest.touch()
start=0 if args.resume_log else dest.stat().st_size
fd=os.open(dest,os.O_RDWR)
def fetch(bounds):
    global url
    a,b=bounds
    for attempt in range(8):
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
        except Exception as error:
            print(f'retry range {a}-{b}: {type(error).__name__}, attempt {attempt+1}',flush=True)
            if attempt==7: raise
            if args.reuse_signed_url and isinstance(error,urllib.error.HTTPError) and error.code in (401,403):
                with url_lock:
                    with urllib.request.urlopen(urllib.request.Request(origin_url,method='HEAD'),timeout=60) as response:
                        url=response.url
            time.sleep(min(2**attempt,15))
chunk=args.chunk_mib*1024*1024
jobs=[(a,min(a+chunk-1,size-1)) for a in range(start,size,chunk)
      if not any(lo<=a and min(a+chunk-1,size-1)<=hi for lo,hi in completed)]
try:
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(fetch,jobs))
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
