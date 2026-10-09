"""Place a caller interruption inside the first qualifying control speech run.

No seed search or outcome-based timing selection. This prepares a new input;
it does not infer an interruption success or calibrated application latency.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf


def select_trigger(words, agent, rate, offset, earliest, latest):
    words = sorted(words, key=lambda w: w['start'])
    runs = []
    for word in words:
        start, end = float(word['start']), float(word['end'])
        if end <= start or not word.get('word', '').strip():
            continue
        if runs and start - runs[-1][1] <= .24:
            runs[-1][1] = max(runs[-1][1], end)
        else:
            runs.append([start, end])
    for start, end in runs:
        # Reserve a real spoken lead-in and a counterfactual continuation.
        trigger = math.ceil(max(start + offset + .6, earliest) / .08) * .08
        if trigger > latest or trigger + 1.4 > end + offset:
            continue
        a = round((trigger - offset - .6) * rate)
        b = round((trigger - offset + 1.4) * rate)
        if a < 0 or b > len(agent):
            continue
        window = agent[a:b]
        frame = rate // 50
        window = window[:len(window) // frame * frame].reshape(-1, frame)
        active = np.sqrt(np.mean(window ** 2, axis=1)) > 10 ** (-45 / 20)
        if float(active.mean()) >= .75:
            return trigger
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--control-input', required=True, type=Path)
    parser.add_argument('--control-output', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    meta = json.loads((args.control_input / 'control.json').read_text())
    metrics = json.loads((args.control_output / 'metrics.json').read_text())
    audit = json.loads((args.control_output / 'audio-audit.json').read_text())
    expected_codes = args.control_input / 'control.npz'
    if (Path(metrics['input']).resolve() != expected_codes.resolve() or
            metrics['input_sha256'] != hashlib.sha256(expected_codes.read_bytes()).hexdigest() or
            metrics['seed'] != 41):
        raise ValueError('Native control must use the recorded control codes and fixed seed')
    if audit['wav_sha256'] != hashlib.sha256((args.control_output / 'agent.wav').read_bytes()).hexdigest():
        raise ValueError('ASR audit no longer matches the actual decoded control audio')
    first = next(f for f in metrics['frames'] if f['agent_frame_available'])
    offset = (first['input_frame'] - first['agent_audio_frame_index']) * .08
    pcm, rate = sf.read(args.control_input / 'control.wav', dtype='int16')
    clip, clip_rate = sf.read(args.control_input / 'second-turn.wav', dtype='int16')
    agent, agent_rate = sf.read(args.control_output / 'agent.wav', dtype='float32')
    assert rate == clip_rate == agent_rate == 24000
    words = [w for segment in audit['segments'] for w in segment['words']]
    trigger = select_trigger(words, agent, rate, offset,
                             meta['first_turn']['end_seconds'] + .4,
                             (len(pcm) - len(clip)) / rate - 1.)
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        'control_input': str(args.control_input), 'control_output': str(args.control_output),
        'seed': 41, 'trigger_seconds': trigger, 'frame_availability_offset_seconds': offset,
        'timing_scope': 'Offline frame alignment, not calibrated live application latency',
        'selection': 'Earliest ASR word run, gaps <=240ms, with 0.6s preceding and 1.4s following speech; >=75% active 20ms RMS windows at -45dBFS',
        'status': 'inconclusive: no sustained control speech' if trigger is None else 'input prepared; actual trial and overlap review required',
    }
    if trigger is not None:
        sample = round(trigger * rate)
        assert sample % 1920 == 0
        changed = pcm.copy()
        assert not np.any(changed[sample:sample + len(clip)])
        changed[sample:sample + len(clip)] = clip
        assert np.array_equal(changed[:sample], pcm[:sample])
        path = args.output / 'user.wav'
        sf.write(path, changed, rate, subtype='PCM_16')
        report['input_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        report['next_check'] = 'Fresh-codec encode, then verify code prefix equals the control before native inference'
    (args.output / 'intervention.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
