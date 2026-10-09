"""Fetch a predeclared eight-case official benchmark subset via ZIP ranges.

Select the first two input.wav paths in lexical order from each complete archive
index before model inference. ZIP CRCs check extracted bytes; local SHA256s fingerprint them.
No archive member is executed. This is a small convenience subset, not the full
benchmark; local ASR and author content review also differ from its full protocol.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import stat
import time
import urllib.request
import zipfile
import zlib


class RemoteZip(io.RawIOBase):
    def __init__(self, ident, size):
        self.url=f'https://drive.usercontent.google.com/download?id={ident}&export=download&confirm=t'
        self.size=size;self.position=0;self.etag=None;self.downloaded=0
    def seekable(self):return True
    def readable(self):return True
    def tell(self):return self.position
    def seek(self,offset,whence=0):
        position=offset if whence==0 else self.position+offset if whence==1 else self.size+offset
        if not 0<=position<=self.size:raise ValueError('ZIP seek outside archive')
        self.position=position;return position
    def read(self,n=-1):
        n=min(self.size-self.position,n if n>=0 else self.size-self.position)
        if not n:return b''
        if n>16_000_000:raise ValueError('Unexpectedly large individual range')
        start=self.position;end=start+n-1
        for attempt in range(3):
            try:
                headers={'Range':f'bytes={start}-{end}'}
                if self.etag:headers['If-Match']=self.etag
                request=urllib.request.Request(self.url,headers=headers)
                with urllib.request.urlopen(request,timeout=25) as response:
                    expected=f'bytes {start}-{end}/{self.size}'
                    if response.status!=206 or response.headers.get('Content-Range')!=expected:
                        raise ValueError('Server did not return the exact requested ZIP range')
                    etag=response.headers.get('ETag')
                    if self.etag and etag!=self.etag:raise ValueError('Archive changed while reading')
                    self.etag=etag
                    data=response.read(n+1)
                    if len(data)!=n:raise ValueError('Incomplete ZIP range')
                self.position+=n;self.downloaded+=n;return data
            except Exception:
                if attempt==2:raise
                time.sleep(1)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('evaluation/full-duplex-bench-v1-subset'))
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    probes=json.loads(Path('sources/full-duplex-bench-download-probes.json').read_text())
    report={'status':'selecting and fetching','selection':'First two lexical input.wav paths per complete archive directory; fixed before model inference',
            'scope':'Eight-case convenience subset, not representative full-benchmark scores',
            'source':'https://github.com/DanielLin94144/Full-Duplex-Bench',
            'license_note':'Synthetic subsets MIT; natural Candor/ICC subsets CC BY-NC4 and upstream terms',
            'archives':[]}
    manifest=Path('sources/full-duplex-bench-subset.json')
    allowed={'input.wav','context.wav','interrupt.wav','pause.json','turn_taking.json','interrupt.json','transcription.json'}
    for row in sorted(probes,key=lambda r:r['name']):
        if row.get('status')!=206 or not row.get('zip_signature'):raise ValueError('Verified range probe required')
        size=int(row['content_range'].split('/')[-1]);remote=RemoteZip(row['id'],size)
        category=row['name'].removesuffix('.zip')
        with zipfile.ZipFile(remote) as archive:
            infos=archive.infolist()
            for info in infos:
                path=PurePosixPath(info.filename)
                if path.is_absolute() or '..' in path.parts or stat.S_ISLNK(info.external_attr>>16):
                    raise ValueError('Unsafe archive member')
            inputs=sorted(i.filename for i in infos if PurePosixPath(i.filename).name=='input.wav' and '__MACOSX' not in i.filename)
            if len(inputs)<2:raise ValueError('Insufficient sample folders')
            selected=inputs[:2];parents={str(PurePosixPath(n).parent) for n in selected}
            record=dict(name=row['name'],google_file_id=row['id'],archive_bytes=size,
                        total_inputs=len(inputs),selected_inputs=selected,etag=remote.etag,files=[])
            report['archives'].append(record);manifest.write_text(json.dumps(report,indent=2)+'\n')
            print('Selected',category,selected,flush=True)
            for info in infos:
                path=PurePosixPath(info.filename)
                if str(path.parent) not in parents or path.name not in allowed:continue
                if info.file_size>30_000_000:raise ValueError('Oversized selected member')
                target=a.output/category/Path(*path.parts)
                target.parent.mkdir(parents=True,exist_ok=True)
                if not target.resolve().is_relative_to(a.output.resolve()) or target.is_symlink():
                    raise ValueError('Destination escapes subset directory')
                if target.exists():
                    content=target.read_bytes()
                    if len(content)!=info.file_size or zlib.crc32(content)!=info.CRC:
                        raise ValueError('Preserve differing local benchmark member')
                else:
                    content=archive.read(info)
                    temporary=target.with_suffix(target.suffix+'.part');temporary.write_bytes(content);temporary.replace(target)
                record['files'].append({'source_member':info.filename,'local_path':str(target),'bytes':len(content),
                                        'zip_crc32':info.CRC,'sha256':hashlib.sha256(content).hexdigest()})
                manifest.write_text(json.dumps(report,indent=2)+'\n')
                print('Verified',target,flush=True)
            record['downloaded_range_bytes']=remote.downloaded
    report['status']='Eight fixed official cases downloaded, ZIP-CRC checked and SHA-fingerprinted; no inference or benchmark scores'
    manifest.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
