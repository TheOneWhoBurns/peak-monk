"""Transactional, resumable full-episode native LoRA using streamed frozen layers."""
import argparse,gc,json,math,os,random,shutil,tempfile,time
from pathlib import Path
from bounded_speech_runtime import BoundedSpeechRuntime
from bounded_text_runtime import atomic_json
from sample_dialogue_paths import sha


def durable_tree(directory):
    for path in directory.rglob('*'):
        if path.is_file():
            with path.open('rb') as f:os.fsync(f.fileno())
    for path in [p for p in directory.rglob('*') if p.is_dir()]+[directory]:
        fd=os.open(path,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--resume',action='store_true');p.add_argument('--until-step',type=int);a=p.parse_args()
    cfg=json.loads(a.config.read_text());out=Path(cfg['output']);config_sha=sha(a.config)
    names=['train_moshi_disk.py','moshi_disk_adjoint.py','moshi_weight_stream.py','moshi_layerwise.py','moshi_staged.py','moshi_disk_metrics.py','bounded_speech_runtime.py','bounded_text_runtime.py','moshi_training.py','moshi_split.py']
    source_hashes={n:sha(Path(__file__).with_name(n)) for n in names}
    if sha(cfg['weights'])!=cfg['weights_sha256'] or sha(cfg['parent'])!=cfg['parent_sha256']:raise ValueError('Base or parent changed')
    rows=[]
    for directory in cfg['data']:
        for path in sorted(Path(directory).glob('*.npz')):
            meta=json.loads(path.with_suffix('.json').read_text())
            if sha(path)!=meta['file_sha256']:raise ValueError('Data changed')
            if directory!=cfg['data'][0] and meta['split']!='train':raise ValueError('Extra data must be training-only')
            rows.append(dict(meta,path=str(path)))
    if len({r['id'] for r in rows})!=len(rows):raise ValueError('Duplicate episode IDs')
    train=[r for r in rows if r['split']=='train'];valid=[r for r in rows if r['split']=='valid']
    assert train and valid
    rng=random.Random(cfg['seed']);schedule=[]
    while len(schedule)<cfg['steps']:
        order=list(range(len(train)));rng.shuffle(order);schedule.extend(order)
    schedule=schedule[:cfg['steps']]
    metadata=dict(config=cfg,config_sha256=config_sha,source_sha256=source_hashes,data_sha256={r['id']:r['file_sha256'] for r in rows},schedule=[train[i]['id'] for i in schedule],validation_ids=[r['id'] for r in valid])
    start_step=0;parent=Path(cfg['parent']);previous=None
    if a.resume:
        saved=json.loads((out/'metadata.json').read_text())
        if saved!=metadata:raise ValueError('Resume requires identical source, data and configuration')
        latest=json.loads((out/'latest.json').read_text());start_step=latest['step'];previous=out/latest['directory'];parent=previous/'candidate.safetensors'
        if sha(parent)!=latest['adapter_sha256']:raise ValueError('Latest adapter changed')
    else:
        if out.exists():raise ValueError('Use a fresh experiment directory')
        out.mkdir(parents=True);atomic_json(out/'metadata.json',metadata)
    stop=min(cfg['steps'],a.until_step or cfg['steps'])
    if start_step>stop:raise ValueError('Requested step precedes committed progress')
    with BoundedSpeechRuntime(out,memory_limit=cfg['memory_limit'],max_seconds=cfg['max_seconds']) as runtime:
        import numpy as np
        import mlx.core as mx
        from moshi_mlx.models import config_v0_1
        from moshi_weight_stream import MoshiParts
        from moshi_split import delay_streams
        from moshi_disk_adjoint import disk_episode_grad,disk_adamw_update
        from moshi_disk_metrics import disk_metrics
        def validate_checkpoint(step, parent):
            candidate_sha=sha(parent)
            snapshot=out/f'checkpoint-{step:04d}.safetensors'
            if not snapshot.exists():os.link(parent,snapshot)
            report_path=out/f'validation-{step:04d}.json'
            if report_path.exists():return
            def validation_boundary(phase,**details):
                runtime.response_done()
            captures=[]
            for validation in valid:
                parts=MoshiParts(cfg['weights'],config_v0_1(),parent,rank=cfg['rank'],scale=cfg['scale'])
                streams=delay_streams(mx.array(np.load(validation['path'])['codes'][None]),parts.cfg)
                score=disk_metrics(parts,streams,head_frames=cfg['head_frames'],boundary=validation_boundary)
                captures.append(dict(episode=validation['id'],**score));del parts,streams
            report=dict(step=step,adapter_sha256=candidate_sha,episodes=captures,validation_joint_ce=sum(c['joint_ce_by_padding_weight']['0.5'] for c in captures)/len(captures),scope='Fixed teacher-forced validation; spoken CE and each audio codebook retained. Free-running review still required.')
            atomic_json(report_path,report);print(json.dumps(dict(step=step,validation_joint_ce=report['validation_joint_ce'])),flush=True)
        if start_step in cfg['checkpoints']:
            validate_checkpoint(start_step,parent)
        for index in range(start_step,stop):
            step=index+1;row=train[schedule[index]];tick=time.monotonic()
            staging=out/f'.step-{step:04d}-pending';final=out/f'step-{step:04d}'
            # Only uncommitted work from this exact experiment can be removed.
            if staging.exists():shutil.rmtree(staging)
            if final.exists():
                # A crash may occur after durable directory rename but before
                # pointer commit; discard that unreferenced generation and replay.
                shutil.rmtree(final)
            staging.mkdir();states=staging/'optimizer';states.mkdir()
            if previous:
                for path in (previous/'optimizer').glob('*.safetensors'):os.link(path,states/path.name)
            def boundary(phase,**details):
                metrics=runtime.response_done()
                if phase!='head_gradient_chunk':runtime.event(phase,step=step,**details,**metrics)
            parts=MoshiParts(cfg['weights'],config_v0_1(),parent,rank=cfg['rank'],scale=cfg['scale'])
            streams=delay_streams(mx.array(np.load(row['path'])['codes'][None]),parts.cfg)
            runtime.event('update_begin',step=step,episode=row['id'])
            with tempfile.TemporaryDirectory(prefix='jun-native-gradient-') as tmp:
                grad=disk_episode_grad(parts,streams,Path(tmp)/'gradient',head_frames=cfg['head_frames'],padding_weight=cfg['text_padding_weight'],boundary=boundary)
                if not math.isfinite(grad['loss']) or not math.isfinite(grad['gradient_norm']) or grad['gradient_norm']<=0:raise ValueError('Invalid gradient')
                update=disk_adamw_update(parts,grad,states,staging/'candidate.safetensors',lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'],clip_norm=cfg['clip_norm'],expected_step=step)
            record=dict(step=step,episode=row['id'],frames=row['frames'],loss=grad['loss'],gradient_norm=grad['gradient_norm'],update=update,seconds=time.monotonic()-tick,metrics=runtime.response_done())
            atomic_json(staging/'update.json',record);durable_tree(staging);staging.rename(final)
            candidate=final/'candidate.safetensors';candidate_sha=sha(candidate)
            atomic_json(out/'latest.json',dict(step=step,directory=final.name,adapter_sha256=candidate_sha))
            if step in cfg['checkpoints']:
                snapshot=out/f'checkpoint-{step:04d}.safetensors'
                if not snapshot.exists():os.link(candidate,snapshot)
            old=previous;previous=final;parent=candidate
            if old and old!=final:shutil.rmtree(old)
            runtime.event('update_committed',**record,adapter_sha256=candidate_sha);print(json.dumps(record),flush=True)
            del grad,streams,parts;gc.collect();runtime.response_done()
            if step in cfg['checkpoints']:
                validate_checkpoint(step,parent)
        atomic_json(out/'segment-completion.json',dict(step=stop,planned_steps=cfg['steps'],status='training complete' if stop==cfg['steps'] else 'checkpoint segment complete',metrics=runtime.response_done()))

if __name__=='__main__':main()
