import json
from pathlib import Path
import subprocess
import sys
from huggingface_hub import snapshot_download

ROOT=Path(__file__).resolve().parents[1]
repo='mlx-community/Qwen3-1.7B-4bit'
revision='3b1b1768f8f8cf8351c712464f906e86c2b8269e'
sha='0e86d9677e519323849eac1bc272caae88567a481ff188c431f70be543d9995f'
snapshot_download(repo,revision=revision,local_dir=ROOT/'models/reference',
                  allow_patterns=['*.json','*.txt','*.jinja','README.md'],max_workers=2)
subprocess.run([sys.executable,str(ROOT/'scripts/range_download.py'),'--repo',repo,
                '--revision',revision,'--sha256',sha,'--destination','models/reference'],check=True)
(ROOT/'reference_provenance.json').write_text(json.dumps({
    'repo':repo,'revision':revision,'sha256':sha,'upstream':'Qwen/Qwen3-1.7B',
    'purpose':'Prompted instruction-tuned comparison; native template with enable_thinking=False',
    'license':'Apache-2.0'},indent=2)+'\n')
