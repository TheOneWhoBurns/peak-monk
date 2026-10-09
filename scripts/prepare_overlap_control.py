"""Remove the second caller turn to create an uninterrupted native control.

The existing frozen probe is unchanged. The control supports a separate,
predeclared adaptive-timing diagnostic; it is never training data.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('case', choices=['s04', 's05'])
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    source = Path('evaluation/audio-roleplay-v2') / args.case
    metadata = json.loads(source.with_suffix('.json').read_text())
    pcm, rate = sf.read(source.with_suffix('.wav'), dtype='int16')
    assert rate == 24000 and pcm.ndim == 1
    assert len(metadata['placements']) == 2
    first, interruption = metadata['placements']
    control = np.zeros_like(pcm)
    start, end = first['start_sample'], first['end_sample']
    control[start:end] = pcm[start:end]
    assert end < interruption['start_sample']
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output / 'control.wav'
    sf.write(path, control, rate, subtype='PCM_16')
    clip = pcm[interruption['start_sample']:interruption['end_sample']]
    sf.write(args.output / 'second-turn.wav', clip, rate, subtype='PCM_16')
    record = {
        'case': args.case, 'source': str(source.with_suffix('.wav')),
        'source_sha256': hashlib.sha256(source.with_suffix('.wav').read_bytes()).hexdigest(),
        'control_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'first_turn': first, 'second_turn_text': interruption['text'],
        'sample_rate': rate, 'samples': len(control),
        'status': 'Prepared control waveform; codec encoding and actual native trial pending',
        'scope': 'Additional development diagnostic; frozen probe preserved; excluded from training',
    }
    (args.output / 'control.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
