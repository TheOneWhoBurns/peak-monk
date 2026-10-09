"""Local full-conversation SFT; frozen evaluation still uses last-reply loss.

Optional development capture reuses the resident model, avoiding a second GPU
copy on the 8 GB Mac. It saves exact adapter snapshots and never assigns ratings.
"""
import argparse
import hashlib
import json
from pathlib import Path
import mlx.core as mx
from mlx.utils import tree_flatten
import train_local
from conversation_objective import ConversationDataset, conversation_batches, conversation_loss
from mlx_lm.tuner.trainer import train, evaluate
from mlx_lm.tuner.datasets import CacheDataset, ChatDataset
from assistant_objective import ThroughEOS, assistant_loss
from evaluate import capture_conversations, ROOT

context = {}


def load_dataset(args, tokenizer):
    datasets = train_local.original_load(args, tokenizer)
    rows = [json.loads(line) for line in (Path(args.data) / 'train.jsonl').read_text().splitlines()]
    train_set = ConversationDataset(rows, tokenizer, args.max_seq_length)
    context.update(args=args, tokenizer=tokenizer)
    print(f'Full-conversation windows: {len(train_set)}; assistant targets: {sum(sum(mask) for _, mask in train_set.items)}', flush=True)
    return (train_set, *[ThroughEOS(d, tokenizer.eos_token_id) if len(d) else d for d in datasets[1:]])


class Audit(train_local.LocalAudit):
    def on_train_loss_report(self, info):
        super().on_train_loss_report(info)
        cfg = context['args']
        iteration = info['iteration']
        interval = getattr(cfg, 'audit_every', 0)
        if not interval or iteration % interval:
            return
        model, tok = context['model'], context['tokenizer']
        snapshot = Path(cfg.adapter_path + f'-step{iteration:04d}')
        snapshot.mkdir(exist_ok=False)
        mx.save_safetensors(str(snapshot/'adapters.safetensors'), dict(tree_flatten(model.trainable_parameters())))
        (snapshot/'adapter_config.json').write_bytes((Path(cfg.adapter_path)/'adapter_config.json').read_bytes())
        adapter_sha = hashlib.sha256((snapshot/'adapters.safetensors').read_bytes()).hexdigest()
        suite = getattr(cfg, 'audit_suite', 'evaluation/dev-roleplay-v2.json')
        args = argparse.Namespace(suite=suite, prompted=False, temperature=0., top_p=1., presence_penalty=0., max_tokens=160, seed=41)
        out = Path('results')/(Path(cfg.adapter_path).name + f'-step{iteration:04d}.json')
        fingerprint = hashlib.sha256()
        for name in ('config.json','tokenizer.json','tokenizer_config.json','chat_template.jinja'):
            path=Path(cfg.model)/name
            if path.exists(): fingerprint.update(name.encode()); fingerprint.update(path.read_bytes())
        validation = [json.loads(line) for line in (ROOT/'data/prepared/valid.jsonl').read_text().splitlines()]
        was_training = model.training
        model.eval()
        try:
            loss = evaluate(model, CacheDataset(ThroughEOS(ChatDataset(validation,tok,mask_prompt=True), tok.eos_token_id)),batch_size=1,num_batches=-1,max_seq_length=768,loss=assistant_loss)
            result = dict(model=cfg.model, adapter=str(snapshot), adapter_sha256=adapter_sha, prompted=False, serialization_sha256=fingerprint.hexdigest(), validation_loss=loss, loss_version='assistant_eos_v2', training_loss_version=getattr(cfg, 'loss_version', 'all_assistant_spans_v1'), suite=suite,suite_sha256=hashlib.sha256(Path(suite).read_bytes()).hexdigest(), temperature=0.,top_p=1.,presence_penalty=0.,max_tokens=160,seed=41,quality_scoring='Unscored development capture; explicit review required',conversations=[], capture_mode='Resident model at completed optimizer/microbatch state; exact adapter saved before capture')
            out.write_text(json.dumps(result,indent=2)+'\n')
            capture_conversations(model,tok,args,result,out)
        finally:
            model.train(was_training)
            mx.clear_cache()


def reporting(report_to, project_name=None, log_dir=None, config=None):
    if report_to: raise ValueError('Local files only')
    return Audit(log_dir)


def train_with_context(model, **kwargs):
    context['model'] = model
    cfg = context['args']
    if getattr(cfg, 'token_weighted_accumulation', False):
        from token_weighted_objective import token_weighted_loss, batch_iterator
        accumulation = cfg.grad_accumulation_steps
        if cfg.iters % accumulation or cfg.steps_per_report % accumulation:
            raise ValueError('Token-weighted iterations/reports must cover complete accumulation groups')
        return train(model=model, **kwargs, loss=token_weighted_loss,
                     iterate_batches=batch_iterator(accumulation))
    return train(model=model, **kwargs, loss=conversation_loss, iterate_batches=conversation_batches)


train_local.lora.load_dataset = load_dataset
train_local.lora.train = train_with_context
train_local.lora.get_reporting_callbacks = reporting

if __name__ == '__main__':
    train_local.lora.main()
