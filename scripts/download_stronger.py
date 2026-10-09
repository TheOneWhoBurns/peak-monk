import json
from pathlib import Path
import subprocess
import sys
from huggingface_hub import snapshot_download

ROOT=Path(__file__).resolve().parents[1]
repo='mlx-community/Qwen3-4B-Instruct-2507-4bit'
revision='50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b'
sha='2a73c6c248601ab904e035548abd8e6abb65ea27dcb5f342fb0a8910eb44173f'
snapshot_download(repo,revision=revision,local_dir=ROOT/'models/stronger',
                  allow_patterns=['*.json','*.txt','*.jinja','README.md'],max_workers=2)
subprocess.run([sys.executable,str(ROOT/'scripts/range_download.py'),'--repo',repo,
                '--revision',revision,'--sha256',sha,'--size','2263022417',
                '--destination','models/stronger'],check=True)
(ROOT/'stronger_provenance.json').write_text(json.dumps({
    'repo':repo,'revision':revision,'sha256':sha,'upstream':'Qwen/Qwen3-4B-Instruct-2507',
    'purpose':'Stronger non-thinking character candidate and local synthetic-data teacher',
    'license':'Apache-2.0'},indent=2)+'\n')
