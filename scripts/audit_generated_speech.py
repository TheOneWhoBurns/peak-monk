"""Transcribe decoded native model audio without feeding its text channel to ASR.

Run after model inference exits, to keep memory bounded. ASR is an inspectable
intelligibility proxy, not a human naturalness rating or calibrated latency test.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import numpy as np
from faster_whisper import WhisperModel


def main():
    p=argparse.ArgumentParser()
    p.add_argument('root',help='Contains one directory per probe, each with agent.wav')
    p.add_argument('--model',default=str(Path.home()/'.cache/huggingface/hub/models--Systran--faster-whisper-large-v3/snapshots/edaa852ec7e145841d8ffdb056a99866b5f0a478'))
    args=p.parse_args()
    files=sorted(Path(args.root).glob('*/agent.wav'))
    if not files:raise ValueError('No decoded native speech found')
    model=WhisperModel(args.model,device='cpu',compute_type='int8',cpu_threads=4,local_files_only=True)
    for path in files:
        dest=path.parent/'audio-audit.json'
        sha=hashlib.sha256(path.read_bytes()).hexdigest()
        if dest.exists():
            prior=json.loads(dest.read_text())
            if prior['wav_sha256']!=sha:raise ValueError('Audio changed; preserve the previous audit and use a fresh output directory')
            continue
        raw=subprocess.check_output(['ffmpeg','-v','error','-i',str(path),'-f','f32le','-ac','1','-ar','16000','-'])
        samples=np.frombuffer(raw,dtype='<f4').copy()
        start=time.monotonic()
        segments,info=model.transcribe(samples,language='en',beam_size=1,word_timestamps=True,
            vad_filter=True,condition_on_previous_text=False)
        segments=list(segments)
        data=dict(wav=str(path),wav_sha256=sha,asr_model=args.model,
            asr_initial_prompt=None,condition_on_previous_text=False,vad_filter=True,
            seconds=time.monotonic()-start,duration_seconds=len(samples)/16000,
            transcript=' '.join(s.text.strip() for s in segments),
            segments=[dict(start=s.start,end=s.end,text=s.text,avg_logprob=s.avg_logprob,
                no_speech_prob=s.no_speech_prob,compression_ratio=s.compression_ratio,
                words=[dict(word=w.word,start=w.start,end=w.end,probability=w.probability) for w in (s.words or [])]) for s in segments],
            scope='Unprompted ASR of actual decoded native audio. Word timestamps are decoder estimates; no automatic quality pass or response-latency claim.',
            review='pending')
        # Read only after transcription; it cannot influence the ASR decoder.
        text_file=path.parent/'transcript.txt'
        data['model_text_channel']=text_file.read_text().strip() if text_file.exists() else None
        dest.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
        print(path.parent.name,data['transcript'],flush=True)


if __name__=='__main__':main()
