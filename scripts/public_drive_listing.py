"""Read public benchmark folder metadata without executing page JavaScript."""
import argparse
import json
from pathlib import Path
import re
import urllib.request


def listing(folder_id):
    if not re.fullmatch(r'[A-Za-z0-9_-]+',folder_id):raise ValueError('Invalid public folder ID')
    url='https://drive.google.com/drive/folders/'+folder_id
    with urllib.request.urlopen(url,timeout=25) as response:
        data=response.read(2_000_001)
    if len(data)>2_000_000:raise ValueError('Unexpectedly large listing')
    page=data.decode('utf-8')
    match=re.search(r"window\['_DRIVE_ivd'\] = '((?:\\.|[^'])*)'",page)
    if not match:raise ValueError('No public folder listing; do not infer access')
    escaped=match.group(1).replace("\\'","'")
    escaped=re.sub(r'\\x([0-9a-fA-F]{2})',r'\\u00\1',escaped)
    entries=json.loads(json.loads('"'+escaped+'"'))[0]
    rows=[{'id':x[0],'name':x[2],'mime':x[3]} for x in entries]
    return {'folder_id':folder_id,'url':url,'entries':rows,
            'scope':'Public first-page metadata, not a guarantee of complete folder enumeration'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder_id');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=listing(a.folder_id)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
