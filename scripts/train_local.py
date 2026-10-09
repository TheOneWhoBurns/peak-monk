"""MLX LoRA driver with a corrected assistant-through-EOS objective.

Uses the upstream optimizer, checkpointing and adapter implementation. Explicit
runtime injection leaves the installed library untouched and is tested separately.
"""
import functools
import importlib
import json
from pathlib import Path
from assistant_objective import ThroughEOS, assistant_loss

lora = importlib.import_module('mlx_lm.lora')
original_load = lora.load_dataset


def load_dataset(args, tokenizer):
    datasets = original_load(args, tokenizer)
    return tuple(ThroughEOS(d, tokenizer.eos_token_id) if len(d) else d for d in datasets)


lora.load_dataset = load_dataset
lora.train = functools.partial(lora.train, loss=assistant_loss)


class LocalAudit:
    def __init__(self, path):
        self.path = Path(path)/'training_metrics.jsonl'
        self.path.parent.mkdir(parents=True,exist_ok=True)
        if self.path.exists():
            raise ValueError('Use a fresh adapter directory for each experiment')

    def write(self, kind, info):
        with self.path.open('a') as f:
            f.write(json.dumps(dict(kind=kind, **info))+'\n')

    def on_train_loss_report(self, train_info):
        self.write('train',train_info)

    def on_val_loss_report(self, val_info):
        self.write('validation',val_info)


def reporting(report_to, project_name=None, log_dir=None, config=None):
    if report_to:
        raise ValueError('This local-only driver records metrics to disk')
    return LocalAudit(log_dir)


lora.get_reporting_callbacks = reporting

if __name__ == '__main__':
    lora.main()
