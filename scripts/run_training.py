"""Run one fixed configuration and preserve timings, logs and checkpoint identity."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import resource
import subprocess
import sys
import time
import yaml

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/train.yaml')
    p.add_argument('--name',required=True)
    args=p.parse_args()
    config=Path(args.config)
    settings=yaml.safe_load(config.read_text())
    results=Path('results');results.mkdir(exist_ok=True)
    (results/f'{args.name}.yaml').write_text(config.read_text())
    start=time.monotonic()
    started=datetime.now(timezone.utc).isoformat()
    proc=subprocess.run([sys.executable,'-m','mlx_lm','lora','--config',str(config)],
                        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    elapsed=time.monotonic()-start
    clean=re.sub(r'\x1b\[[0-9;]*m','',proc.stdout)
    (results/f'{args.name}.txt').write_text(clean)
    print(clean,flush=True)
    adapter=Path(settings['adapter_path'])/'adapters.safetensors'
    meta={'started_at':started,'wall_seconds':elapsed,'exit_code':proc.returncode,
          'configuration_sha256':hashlib.sha256(config.read_bytes()).hexdigest(),
          'child_max_rss_bytes_macos':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
          'metric_note':'RSS is not total MLX Metal allocation; inference capture records MLX peak memory.'}
    if proc.returncode==0:
        meta.update(adapter_sha256=hashlib.sha256(adapter.read_bytes()).hexdigest(),
                    adapter_bytes=adapter.stat().st_size)
    (results/f'{args.name}-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    raise SystemExit(proc.returncode)

if __name__=='__main__':main()
