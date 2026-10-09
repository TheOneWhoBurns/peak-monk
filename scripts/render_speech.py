"""Render original dialogues with stock open-weight Kokoro voices on CPU.

Output is synthetic training material, not model-generated Jun speech. Requires
subsequent ASR/timestamp quality checks before use in native speech training.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import onnxruntime as ort
import soundfile as sf
from kokoro_onnx import Kokoro


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--corpus', default='data/conversations.jsonl')
    p.add_argument('--limit', type=int, default=8)
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--split', choices=['train','validation'], default='train')
    args = p.parse_args()
    root = Path('models/kokoro-hf')
    assets = json.loads(Path('sources/kokoro-onnx-hf.json').read_text())
    voices = {}
    for name in ('bm_george', 'af_heart', 'am_michael'):
        data = root/'voices'/f'{name}.bin'
        expected = next(a['lfs']['sha256'] for a in assets['siblings'] if a['rfilename'] == f'voices/{name}.bin')
        assert hashlib.sha256(data.read_bytes()).hexdigest() == expected
        voices[name] = np.fromfile(data, dtype='<f4').reshape(-1, 1, 256)
    np.savez(root/'voices.npz', **voices)
    model_file = root/'onnx/model_quantized.onnx'
    assert hashlib.sha256(model_file.read_bytes()).hexdigest() == 'fbae9257e1e05ffc727e951ef9b9c98418e6d79f1c9b6b13bd59f5c9028a1478'
    options = ort.SessionOptions(); options.intra_op_num_threads=4; options.inter_op_num_threads=1
    session = ort.InferenceSession(str(model_file), sess_options=options, providers=['CPUExecutionProvider'])
    tts = Kokoro.from_session(session, str(root/'voices.npz'))
    rows = [json.loads(l) for l in Path(args.corpus).read_text().splitlines()]
    rows = [r for r in rows if r['split']==args.split and r['family'] not in ('everyday_french','language_flexibility')]
    out = Path('data/renders'); out.mkdir(parents=True,exist_ok=True)
    for ci, row in enumerate(rows[args.start:args.start+args.limit], start=args.start):
        ident = row['family']; dest=out/f'{ident}.json'
        if dest.exists(): continue
        record = {'id':ident, 'family':ident, 'sources':row['sources'],
                  'split':'valid' if args.split=='validation' else 'train', 'turns':[],
                  'tts_model':'onnx-community/Kokoro-82M-v1.0-ONNX', 'tts_revision':assets['sha'],
                  'quality_status':'pending ASR/timestamp review; not accepted training audio'}
        for j, message in enumerate(row['messages']):
            voice = 'bm_george' if message['role']=='assistant' else ('af_heart' if ci%2==0 else 'am_michael')
            start=time.monotonic()
            audio, rate=tts.create(message['content'],voice=voice,speed=.95 if voice=='bm_george' else 1.0,
                                  lang='en-gb' if voice=='bm_george' else 'en-us')
            assert rate==24000 and np.isfinite(audio).all() and len(audio)>rate*.1
            peak=float(np.max(np.abs(audio))); assert peak>1e-4
            if peak>.95: audio=audio*(.95/peak)
            wav=out/f'{ident}-{j:02d}.wav'; sf.write(wav,audio,rate,subtype='PCM_16')
            gap=400
            # Explicit interruption family: caller enters before Jun has stopped.
            if ident=='phone_interruption' and j>1 and message['role']=='user': gap=-240
            record['turns'].append(dict(message,wav=wav.name,voice=voice,gap_ms=gap,
                duration_seconds=len(audio)/rate,render_seconds=time.monotonic()-start,
                wav_sha256=hashlib.sha256(wav.read_bytes()).hexdigest()))
        dest.write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n')
        print(ident,len(record['turns']),round(sum(t['duration_seconds'] for t in record['turns']),1),flush=True)
    manifests=[]
    for path in sorted(out.glob('*.json')):
        item=json.loads(path.read_text())
        if 'turns' in item and 'tts_model' in item: manifests.append(item)
    (out/'manifest.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in manifests))


if __name__=='__main__': main()
