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
    objective=p.add_mutually_exclusive_group()
    objective.add_argument('--corrected-objective',action='store_true')
    objective.add_argument('--conversation-objective',action='store_true')
    args=p.parse_args()
    config=Path(args.config)
    settings=yaml.safe_load(config.read_text())
    results=Path('results');results.mkdir(exist_ok=True)
    (results/f'{args.name}.yaml').write_text(config.read_text())
    start=time.monotonic()
    started=datetime.now(timezone.utc).isoformat()
    source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    data_hashes={name:hashlib.sha256((Path(settings['data'])/name).read_bytes()).hexdigest()
                 for name in ['train.jsonl','valid.jsonl']}
    engine = [str(Path(__file__).with_name('train_conversations.py'))] if args.conversation_objective else [str(Path(__file__).with_name('train_local.py'))] if args.corrected_objective else ['-m','mlx_lm','lora']
    proc=subprocess.Popen([sys.executable,*engine,'--config',str(config),'--train'],
                          stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    output=[]
    for line in proc.stdout:
        clean_line=re.sub(r'\x1b\[[0-9;]*m','',line)
        output.append(clean_line)
        print(clean_line,end='',flush=True)
    proc.wait()
    elapsed=time.monotonic()-start
    clean=''.join(output)
    (results/f'{args.name}.txt').write_text(clean)
    adapter=Path(settings['adapter_path'])/'adapters.safetensors'
    metrics=adapter.parent/'training_metrics.jsonl'
    if metrics.exists():
        (results/f'{args.name}-metrics.jsonl').write_bytes(metrics.read_bytes())
    meta={'started_at':started,'wall_seconds':elapsed,'exit_code':proc.returncode,
          'source_commit':source_commit,'data_sha256':data_hashes,
          'loss_version':'all_assistant_spans_v1' if args.conversation_objective else 'assistant_eos_v2' if args.corrected_objective else 'legacy',
          'evaluation_loss_version':'assistant_eos_v2' if (args.conversation_objective or args.corrected_objective) else 'legacy',
          'configuration_sha256':hashlib.sha256(config.read_bytes()).hexdigest(),
          'child_max_rss_bytes_macos':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
          'metric_note':'RSS is not total MLX Metal allocation; inference capture records MLX peak memory.'}
    if proc.returncode==0:
        meta.update(adapter_sha256=hashlib.sha256(adapter.read_bytes()).hexdigest(),
                    adapter_bytes=adapter.stat().st_size)
    (results/f'{args.name}-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    raise SystemExit(proc.returncode)

if __name__=='__main__':main()
