"""Average two LoRA weight updates exactly by concatenating their low-rank factors.

Uses CPU memory maps, one tensor at a time. This averages layer weight updates,
not model predictions. Rank doubles; quality must be evaluated separately.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess

import numpy as np


def header(path):
    with path.open('rb') as source:
        length = struct.unpack('<Q', source.read(8))[0]
        contents = json.loads(source.read(length))
    return length + 8, {k: v for k, v in contents.items() if k != '__metadata__'}


def combine(name, a, b):
    if name.endswith('.lora_a'):
        return np.concatenate([a, b], axis=1)
    if name.endswith('.lora_b'):
        return np.concatenate([a * .5, b * .5], axis=0)
    raise ValueError(f'Unsupported adapter tensor: {name}')


def digest(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('first', type=Path)
    parser.add_argument('second', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    parents = [args.first, args.second]
    configs = [json.loads((p / 'adapter_config.json').read_text()) for p in parents]
    keys = ['model', 'num_layers', 'fine_tune_type', 'lora_parameters']
    assert all(configs[0][k] == configs[1][k] for k in keys)
    assert configs[0]['fine_tune_type'] == 'lora'
    files = [p / 'adapters.safetensors' for p in parents]
    bases, headers = zip(*(header(p) for p in files))
    assert headers[0].keys() == headers[1].keys()
    out_header = {}
    position = 0
    for name in sorted(headers[0]):
        a, b = (h[name] for h in headers)
        assert a['shape'] == b['shape'] and a['dtype'] == b['dtype'] == 'F32'
        shape = list(a['shape'])
        assert len(shape) == 2
        axis = 1 if name.endswith('.lora_a') else 0
        assert name.endswith(('.lora_a', '.lora_b'))
        assert shape[axis] == configs[0]['lora_parameters']['rank']
        shape[axis] *= 2
        size = int(np.prod(shape)) * 4
        out_header[name] = {'dtype': 'F32', 'shape': shape,
                            'data_offsets': [position, position + size]}
        position += size
    encoded = json.dumps(out_header, separators=(',', ':')).encode()
    encoded += b' ' * ((-len(encoded)) % 8)
    args.output.mkdir(parents=True, exist_ok=False)
    destination = args.output / 'adapters.safetensors'
    with destination.open('wb') as target:
        target.write(struct.pack('<Q', len(encoded)))
        target.write(encoded)
        for name in sorted(headers[0]):
            arrays = [np.memmap(path, mode='r', dtype='<f4', shape=h[name]['shape'],
                                offset=base + h[name]['data_offsets'][0])
                      for path, base, h in zip(files, bases, headers)]
            assert all(np.isfinite(a).all() for a in arrays)
            result = combine(name, *arrays)
            target.write(result.astype('<f4', copy=False).tobytes())
            del arrays, result
    config = {k: configs[0][k] for k in keys}
    config['lora_parameters'] = dict(config['lora_parameters'])
    config['lora_parameters']['rank'] *= 2
    config['adapter_path'] = str(args.output)
    config['derivation'] = 'Equal average of two layer weight updates via factor concatenation'
    (args.output / 'adapter_config.json').write_text(json.dumps(config, indent=2) + '\n')
    report = {
        'status': 'Derived candidate only; no quality evaluation yet',
        'parents': [{'path': str(p), 'sha256': digest(f)} for p, f in zip(parents, files)],
        'weights': [.5, .5], 'rank': config['lora_parameters']['rank'],
        'scale': config['lora_parameters']['scale'],
        'adapter_sha256': digest(destination), 'adapter_bytes': destination.stat().st_size,
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'limitation': 'Averaging layer updates does not average nonlinear model outputs or guarantee preservation of either parent behavior.',
    }
    (args.output / 'derivation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
