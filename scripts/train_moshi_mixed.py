"""Transactional text/audio adaptation with separate optimizer states.

All gradients update the same temporal/text-head adapters. Native audio replay
retains the unchanged depth objective. No text forcing occurs at inference.
"""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
from bounded_speech_runtime import BoundedSpeechRuntime
from bounded_text_runtime import atomic_json
from sample_dialogue_paths import sha
from train_moshi_disk import durable_tree
from moshi_modality_schedule import make_schedule


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--until-step', type=int)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    out = Path(cfg['output'])
    for path, expected in [(cfg['weights'], cfg['weights_sha256']),
                           (cfg['parent'], cfg['parent_sha256']),
                           (cfg['text_data'], cfg['text_data_sha256']),
                           (cfg['tokenizer'], cfg['tokenizer_sha256'])]:
        if sha(path) != expected:
            raise ValueError('Declared input changed: '+path)
    import sentencepiece as spm
    processor = spm.SentencePieceProcessor(model_file=cfg['tokenizer'])
    if processor.get_piece_size() != 32000:
        raise ValueError('Expected native Moshi text vocabulary')
    text_rows = json.loads(Path(cfg['text_data']).read_text())['rows']
    if len({r['family'] for r in text_rows}) != len(text_rows):
        raise ValueError('This small notebook experiment requires distinct episode families')
    for row in text_rows:
        row['tokens'] = [32000] + processor.encode(row['text'])
        if not 2 <= len(row['tokens']) <= cfg['max_text_tokens']:
            raise ValueError('Text episode exceeds declared complete-context limit')
        if row['split'] not in ('train', 'valid'):
            raise ValueError('Unknown text split')
    text_train = [r for r in text_rows if r['split']=='train']
    text_valid = [r for r in text_rows if r['split']=='valid']
    audio_rows = []
    for directory in cfg['audio_data']:
        for path in sorted(Path(directory).glob('*.npz')):
            row = json.loads(path.with_suffix('.json').read_text())
            if sha(path) != row['file_sha256']:
                raise ValueError('Encoded audio changed')
            audio_rows.append(dict(row, path=str(path)))
    if len({r['id'] for r in audio_rows}) != len(audio_rows):
        raise ValueError('Duplicate audio episode')
    audio_index = {r['id']:r for r in audio_rows if r['split']=='train'}
    audio_valid = [r for r in audio_rows if r['split']=='valid']
    if not text_train or not text_valid or not audio_valid:
        raise ValueError('Training and diagnostic splits required')
    if set(cfg['audio_replay_ids'])-set(audio_index):
        raise ValueError('Replay may use only existing training audio')
    if len(cfg['audio_replay_ids']) != len(set(cfg['audio_replay_ids'])):
        raise ValueError('Repeated replay selection')
    schedule = make_schedule([r['id'] for r in text_train], cfg['audio_replay_ids'],
                             cfg['seed'], audio_only=cfg.get('audio_only', False))
    if cfg.get('audio_only'):
        reference_path = Path(cfg['mixed_reference_metadata'])
        if sha(reference_path) != cfg['mixed_reference_metadata_sha256']:
            raise ValueError('Mixed reference changed')
        reference = json.loads(reference_path.read_text())
        if schedule != [item for item in reference['schedule'] if item['modality']=='audio']:
            raise ValueError('Audio control order differs from mixed experiment')
        varying = {'output','checkpoints','audio_only','mixed_reference_metadata',
                   'mixed_reference_metadata_sha256'}
        if {k:v for k,v in cfg.items() if k not in varying} != {k:v for k,v in reference['config'].items() if k not in varying}:
            raise ValueError('Audio control changed more than removal of text updates')
    text_index = {r['id']:r for r in text_rows}
    names = ['train_moshi_mixed.py','train_moshi_disk.py','moshi_text_adjoint.py',
             'moshi_disk_adjoint.py','moshi_weight_stream.py','moshi_layerwise.py',
             'moshi_staged.py','moshi_disk_metrics.py','bounded_speech_runtime.py',
             'bounded_text_runtime.py','moshi_training.py','moshi_split.py',
             'moshi_modality_schedule.py']
    metadata = dict(config=cfg, config_sha256=sha(args.config),
        source_sha256={n:sha(Path(__file__).with_name(n)) for n in names},
        data_sha256={r['id']:r['file_sha256'] for r in audio_rows},
        text_token_counts={r['id']:len(r['tokens']) for r in text_rows},
        schedule=schedule, audio_validation_ids=[r['id'] for r in audio_valid],
        text_validation_ids=[r['id'] for r in text_valid],
        serialization='Native input32000 + original prose; no appended EOS. Every prose token supervised.',
        optimizers='Separate AdamW moments and step counts for text and native audio; shared adapters.')
    start_step = 0
    counts = dict(text=0,audio=0)
    previous = None
    parent = Path(cfg['parent'])
    if args.resume:
        if json.loads((out/'metadata.json').read_text()) != metadata:
            raise ValueError('Resume requires identical sources, configuration, and data')
        latest = json.loads((out/'latest.json').read_text())
        start_step, counts = latest['step'], latest['optimizer_steps']
        previous = out/latest['directory']
        parent = previous/'candidate.safetensors'
        if sha(parent) != latest['adapter_sha256']:
            raise ValueError('Committed adapter changed')
    else:
        out.mkdir(parents=True,exist_ok=False)
        atomic_json(out/'metadata.json',metadata)
    if args.until_step is not None and args.until_step < 1:
        raise ValueError('Requested step must be positive')
    stop = min(len(schedule),args.until_step if args.until_step is not None else len(schedule))
    if not start_step <= stop <= len(schedule):
        raise ValueError('Invalid requested stopping step')
    with BoundedSpeechRuntime(out,memory_limit=cfg['memory_limit'],max_seconds=cfg['max_seconds']) as runtime:
        import numpy as np
        import mlx.core as mx
        from moshi_mlx.models import config_v0_1
        from moshi_weight_stream import MoshiParts
        from moshi_text_adjoint import TextMoshiParts,disk_text_grad,disk_text_score
        from moshi_split import delay_streams
        from moshi_disk_adjoint import disk_episode_grad,disk_adamw_update
        from moshi_disk_metrics import disk_metrics
        model_cfg = config_v0_1()

        def boundary(phase,**details):
            metrics = runtime.response_done()
            if metrics['mlx_peak_bytes'] > cfg['max_peak_bytes']:
                runtime.event('resource_guard_stop',reason='mixed training peak limit',**metrics)
                raise RuntimeError('Mixed native experiment exceeded peak plan')

        def text_batch(row):
            tokens = row['tokens']
            return mx.array([tokens]),mx.array([[0]+[1]*(len(tokens)-1)])

        def validate(step,adapter):
            if step:
                snapshot = out/f'checkpoint-{step:04d}.safetensors'
                if not snapshot.exists(): os.link(adapter,snapshot)
            report_path = out/f'validation-{step:04d}.json'
            if report_path.exists():
                if json.loads(report_path.read_text())['adapter_sha256'] != sha(adapter):
                    raise ValueError('Validation belongs to another adapter')
                return
            text_scores, audio_scores = [], []
            for row in text_valid:
                parts = TextMoshiParts(cfg['weights'],model_cfg,adapter,rank=cfg['rank'],scale=cfg['scale'])
                score = disk_text_score(parts,*text_batch(row),head_tokens=cfg['head_tokens'],boundary=boundary)
                text_scores.append(dict(id=row['id'],**score))
                del parts
                runtime.event('text_validation_episode',step=step,id=row['id'],**score,**runtime.response_done())
            for row in audio_valid:
                parts = MoshiParts(cfg['weights'],model_cfg,adapter,rank=cfg['rank'],scale=cfg['scale'])
                streams = delay_streams(mx.array(np.load(row['path'])['codes'][None]),model_cfg)
                score = disk_metrics(parts,streams,head_frames=cfg['head_frames'],boundary=boundary)
                audio_scores.append(dict(id=row['id'],**score))
                del parts,streams
                runtime.response_done()
            report = dict(step=step,adapter_sha256=sha(adapter),text_episodes=text_scores,audio_episodes=audio_scores,
                text_ce=sum(r['loss']*r['target_tokens'] for r in text_scores)/sum(r['target_tokens'] for r in text_scores),
                audio_joint_ce=sum(r['joint_ce_by_padding_weight']['0.5'] for r in audio_scores)/len(audio_scores),
                native_spoken_ce=sum(r['text_groups']['spoken']['ce_sum'] for r in audio_scores)/sum(r['text_groups']['spoken']['tokens'] for r in audio_scores),
                scope='Small fixed teacher-forced diagnostics. No native free-generation acceptance claim.')
            if step == 0 and abs(report['audio_joint_ce']-cfg['expected_parent_audio_joint_ce']) > 1e-6:
                raise ValueError('Parent native diagnostic no longer matches recorded reference')
            atomic_json(report_path,report)
            print(json.dumps({k:v for k,v in report.items() if k not in ('text_episodes','audio_episodes')}),flush=True)

        if not (out/'validation-0000.json').exists():
            validate(0,Path(cfg['parent']))
        if start_step in cfg['checkpoints']:
            validate(start_step,parent)
        for index in range(start_step,stop):
            step = index+1
            item = schedule[index]
            kind = item['modality']
            staging = out/f'.step-{step:04d}-pending'
            final = out/f'step-{step:04d}'
            if staging.exists(): shutil.rmtree(staging)
            if final.exists(): shutil.rmtree(final)  # Uncommitted generation only.
            staging.mkdir()
            for mode in ('text','audio'):
                state_dir = staging/('optimizer-'+mode)
                state_dir.mkdir()
                if previous:
                    for path in (previous/('optimizer-'+mode)).glob('*.safetensors'):
                        os.link(path,state_dir/path.name)
            tick = time.monotonic()
            runtime.event('mixed_update_begin',step=step,**item)
            with tempfile.TemporaryDirectory(prefix='jun-mixed-gradient-') as temporary:
                scratch = Path(temporary)/'gradient'
                if kind=='text':
                    parts = TextMoshiParts(cfg['weights'],model_cfg,parent,rank=cfg['rank'],scale=cfg['scale'])
                    batch,mask = text_batch(text_index[item['id']])
                    gradient = disk_text_grad(parts,batch,mask,scratch,head_tokens=cfg['head_tokens'],boundary=boundary)
                    del batch,mask
                else:
                    parts = MoshiParts(cfg['weights'],model_cfg,parent,rank=cfg['rank'],scale=cfg['scale'])
                    streams = delay_streams(mx.array(np.load(audio_index[item['id']]['path'])['codes'][None]),model_cfg)
                    gradient = disk_episode_grad(parts,streams,scratch,head_frames=cfg['head_frames'],padding_weight=cfg['text_padding_weight'],boundary=boundary)
                    del streams
                if not math.isfinite(gradient['loss']) or not math.isfinite(gradient['gradient_norm']) or gradient['gradient_norm']<=0:
                    raise ValueError('Invalid mixed gradient')
                next_counts = dict(counts)
                next_counts[kind] += 1
                update = disk_adamw_update(parts,gradient,staging/('optimizer-'+kind),staging/'candidate.safetensors',
                    lr=cfg[kind+'_learning_rate'],weight_decay=cfg['weight_decay'],clip_norm=cfg['clip_norm'],expected_step=next_counts[kind])
            record = dict(step=step,**item,loss=gradient['loss'],gradient_norm=gradient['gradient_norm'],
                optimizer_steps=next_counts,update=update,seconds=time.monotonic()-tick,metrics=runtime.response_done())
            atomic_json(staging/'update.json',record)
            durable_tree(staging)
            staging.rename(final)
            candidate = final/'candidate.safetensors'
            digest = sha(candidate)
            atomic_json(out/'latest.json',dict(step=step,directory=final.name,adapter_sha256=digest,optimizer_steps=next_counts))
            if step in cfg['checkpoints']:
                snapshot = out/f'checkpoint-{step:04d}.safetensors'
                if not snapshot.exists(): os.link(candidate,snapshot)
            old = previous
            previous,parent,counts = final,candidate,next_counts
            if old and old!=final: shutil.rmtree(old)
            runtime.event('mixed_update_committed',**record,adapter_sha256=digest)
            print(json.dumps(record),flush=True)
            del parts,gradient
            gc.collect()
            runtime.response_done()
            if step in cfg['checkpoints']: validate(step,parent)
        atomic_json(out/'segment-completion.json',dict(step=stop,planned_steps=len(schedule),optimizer_steps=counts,
            status='training complete' if stop==len(schedule) else 'segment complete',metrics=runtime.response_done()))


if __name__=='__main__': main()
