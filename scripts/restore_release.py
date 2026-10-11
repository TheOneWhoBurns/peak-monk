"""Restore a published Jun adapter and its exact base serialization, without loading MLX."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', default='A-Martinez-2508/peak-monk-jun-run26')
    parser.add_argument('--revision', required=True, help='Published Hugging Face commit SHA')
    args = parser.parse_args()
    from huggingface_hub import snapshot_download, hf_hub_download
    with tempfile.TemporaryDirectory(prefix='jun-release-') as download_dir:
        snapshot = Path(snapshot_download(args.repo, revision=args.revision, local_dir=download_dir))
        manifest = json.loads((snapshot/'release-manifest.json').read_text())
        for name, expected in manifest['files'].items():
            if '..' in Path(name).parts or Path(name).is_absolute():
                raise ValueError('Invalid manifest path')
            path = snapshot/name
            if digest(path) != expected:
                raise ValueError(f'Checksum mismatch: {name}')
        with tempfile.TemporaryDirectory(prefix='jun-metadata-') as temporary:
            metadata = snapshot/'base-metadata'
            if (snapshot/'base-metadata.zip').exists():
                metadata = Path(temporary)
                with zipfile.ZipFile(snapshot/'base-metadata.zip') as archive:
                    for entry in archive.infolist():
                        if (entry.is_dir() or Path(entry.filename).name != entry.filename
                                or entry.filename in ('.', '..') or entry.file_size > 100_000_000):
                            raise ValueError('Invalid base metadata archive entry')
                    archive.extractall(metadata)
            provenance = json.loads((metadata/'base4b_provenance.json').read_text())
            base = ROOT/'models/base4b'
            adapter = ROOT/'adapters/run26-step0128'
            for path in (base, adapter):
                path.mkdir(parents=True, exist_ok=True)
            pairs = [(snapshot/name, adapter/name) for name in ('adapters.safetensors', 'adapter_config.json')]
            pairs += [(p, base/p.name) for p in metadata.iterdir() if p.is_file()]
            for source, destination in pairs:
                if destination.exists() and digest(destination) != digest(source):
                    raise ValueError(f'Refusing to overwrite different local file: {destination}')
            weight = Path(hf_hub_download(provenance['repo'], 'model.safetensors',
                revision=provenance['revision'], local_dir=str(base)))
            if digest(weight) != provenance['weight_sha256_verified']:
                raise ValueError('Base weight checksum mismatch')
            for source, destination in pairs:
                if not destination.exists():
                    shutil.copy2(source, destination)
    print('Restored and verified. Run: .venv/bin/python scripts/chat.py --adapter adapters/run26-step0128')


if __name__ == '__main__':
    main()
