"""Offline activity/word evidence for fixed native probes; never a pass score."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import soundfile as sf


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('capture', type=Path)
    p.add_argument('--inputs', type=Path, default=Path('evaluation/audio-roleplay-v2'))
    a = p.parse_args()
    output = a.capture / 'overlap-evidence.json'
    if output.exists():
        raise ValueError('Preserve existing evidence; use a fresh capture')
    cases = []
    for ident in ['s04', 's05']:
        meta = json.loads((a.inputs / f'{ident}.json').read_text())
        event = meta['placements'][1]
        caller_path = a.inputs / f'{ident}.wav'
        agent_path = a.capture / ident / 'agent.wav'
        caller, rate = sf.read(caller_path, dtype='float32')
        agent, agent_rate = sf.read(agent_path, dtype='float32')
        if rate != 24000 or agent_rate != rate or caller.ndim != 1 or agent.ndim != 1:
            raise ValueError('Expected24kHzmono measured waveforms')
        if sha(caller_path) != meta['wav_sha256']:
            raise ValueError('Caller waveform differs from fixed probe')
        audit_path = a.capture / ident / 'audio-audit.json'
        audit = json.loads(audit_path.read_text())
        if sha(agent_path) != audit['wav_sha256'] or audit['asr_initial_prompt'] is not None:
            raise ValueError('Expected exact output with unprompted ASR')
        words = [w for s in audit['segments'] for w in s['words']]
        measurements = []
        for shift in [0., .08]:
            shifted = np.pad(agent, (round(shift * rate), 0))
            size = max(len(caller), len(shifted))
            pair = np.stack([np.pad(caller, (0, size-len(caller))),
                             np.pad(shifted, (0, size-len(shifted)))], axis=1)
            activity = np.sqrt(np.mean(pair[:size//480*480].reshape(-1,480,2)**2,axis=1)) > 10**(-45/20)
            times = np.arange(len(activity)) * .02
            during = (times >= event['start_seconds']) & (times+.02 <= event['end_seconds'])
            before = (times >= event['start_seconds']-1) & (times+.02 <= event['start_seconds'])
            measurements.append(dict(agent_wave_shift_seconds=shift,
                simultaneous_rms_active_seconds=float(np.all(activity[during],axis=1).sum()*.02),
                agent_active_in_preceding_second=float(activity[before,1].sum()*.02)))
        cases.append(dict(id=ident,caller_event=event,
            caller_wav_sha256=sha(caller_path),agent_wav_sha256=sha(agent_path),asr_sha256=sha(audit_path),
            nearby_words=[w for w in words if w['end'] >= event['start_seconds']-2 and w['start'] <= event['end_seconds']+3],
            later_words=[w for w in words if w['start'] > event['end_seconds']+3],
            alignment_sensitivity=measurements))
    report = dict(scope='Offline20msRMS at-45dBFS plus unprompted ASR estimated word times; activity is not VAD or intelligibility. Zero and80ms output shifts both reported, no live latency or automatic interruption pass.',cases=cases)
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({c['id']:c['alignment_sensitivity'] for c in cases},indent=2))


if __name__ == '__main__':
    main()
