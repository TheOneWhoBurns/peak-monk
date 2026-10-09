"""Serialize native inference, codec decoding, and ASR within the Mac budget.

This records evidence only. Character and overlap judgments remain explicit;
an ASR transcript or successful subprocess never constitutes an automatic pass.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def run(command, log):
    with log.open('w') as output:
        subprocess.run([sys.executable, *command], stdout=output,
                       stderr=subprocess.STDOUT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapter', required=True, type=Path)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if not args.adapter.is_file() or not args.config.is_file():
        raise ValueError('A saved candidate and its actual training config are required')
    args.output.mkdir(exist_ok=False)
    manifest = {
        'adapter': str(args.adapter),
        'adapter_sha256': hashlib.sha256(args.adapter.read_bytes()).hexdigest(),
        'config': str(args.config),
        'config_sha256': hashlib.sha256(args.config.read_bytes()).hexdigest(),
        'seed': 41,
        'status': 'running; unscored',
        'scope': 'Six fixed development probes, native speech generation. No training text forcing.',
        'cases': [],
    }
    output = args.output / 'audit-manifest.json'
    output.write_text(json.dumps(manifest, indent=2) + '\n')
    for index in range(1, 7):
        name = f's{index:02d}'
        inputs = Path('evaluation/audio-roleplay-v2') / (name + '.npz')
        target = args.output / name
        run(['scripts/infer_moshi_codes.py', str(inputs), '--adapter', str(args.adapter),
             '--adapter-config', str(args.config), '--seed', '41', '--output', str(target)],
            args.output / (name + '-inference.log'))
        run(['scripts/mimi_audio.py', 'decode', str(target / 'agent_codes.npz'),
             '--output', str(target / 'agent.wav')], args.output / (name + '-decode.log'))
        metrics = json.loads((target / 'metrics.json').read_text())
        manifest['cases'].append({
            'id': name, 'input_sha256': metrics['input_sha256'],
            'input_seconds': metrics['input_seconds'],
            'generation_seconds': metrics['generation_seconds'],
            'peak_model_memory_bytes': metrics['peak_model_memory_bytes'],
            'text': (target / 'transcript.txt').read_text().strip(),
        })
        output.write_text(json.dumps(manifest, indent=2) + '\n')
        print(name, manifest['cases'][-1]['text'], flush=True)
    run(['scripts/audit_generated_speech.py', str(args.output)], args.output / 'asr.log')
    for case in manifest['cases']:
        audit = json.loads((args.output / case['id'] / 'audio-audit.json').read_text())
        case['unprompted_asr'] = audit['transcript']
    manifest['status'] = 'evidence complete; explicit character and overlap review required'
    manifest['model_only_rtf'] = (sum(c['generation_seconds'] for c in manifest['cases']) /
                                  sum(c['input_seconds'] for c in manifest['cases']))
    output.write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
