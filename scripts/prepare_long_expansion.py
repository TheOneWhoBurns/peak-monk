"""Prepare an append-only long-dialogue expansion without modifying a running run."""
import argparse
import hashlib
import json
from pathlib import Path

from prepare import parse


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=Path('data/prepared-full-relations-v3'))
    parser.add_argument('--new-source', type=Path, action='append', required=True)
    parser.add_argument('--repeat', type=int, default=2)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.repeat < 1:
        raise ValueError('Repeat count must be positive')
    prior = [json.loads(line) for line in (args.source / 'train.jsonl').read_text().splitlines()]
    families = {r['family'] for r in prior}
    additions = []
    sources = []
    for source in args.new_source:
        manifest_path = source.with_name(source.stem + '-manifest.json')
        manifest = json.loads(manifest_path.read_text())
        if manifest['sha256'] != digest(source) or manifest.get('exact_heldout_prompt_overlap') is not False:
            raise ValueError(f'Missing matching source review: {source}')
        rows = parse(source)
        for row in rows:
            if row['family'] in families:
                raise ValueError(f'Duplicate family: {row["family"]}')
            families.add(row['family'])
            if [m['role'] for m in row['messages']] != ['user', 'assistant'] * 10:
                raise ValueError('Expected ten complete exchanges per new family')
            row.update(split='train', source_file=str(source), source_sha256=digest(source),
                       source_kind='Original authored synthetic fiction; broad influence labels only')
            additions.append(row)
        sources.append({'path': str(source), 'sha256': digest(source),
                        'manifest_sha256': digest(manifest_path)})
    weighted = [dict(row, curriculum_repetition=i) for row in additions for i in range(args.repeat)]
    args.output.mkdir(parents=True, exist_ok=False)
    original = (args.source / 'train.jsonl').read_bytes()
    if not original.endswith(b'\n'):
        raise ValueError('Original corpus must end with a newline')
    (args.output / 'train.jsonl').write_bytes(original + ''.join(
        json.dumps(r, ensure_ascii=False) + '\n' for r in weighted).encode())
    for name in ('valid.jsonl', 'test.jsonl'):
        (args.output / name).write_bytes((args.source / name).read_bytes())
    rows = prior + weighted
    distinct = {json.dumps(r['messages'], sort_keys=True): r for r in rows}
    manifest = {
        'status': 'Prepared for a future experiment; not used by run17',
        'original_source': str(args.source), 'original_train_sha256': digest(args.source / 'train.jsonl'),
        'new_sources': sources, 'new_family_count': len(additions), 'new_family_weight': args.repeat,
        'weighted_records': len(rows), 'distinct_texts': len(distinct),
        'weighted_character_turns': sum(sum(m['role'] == 'assistant' for m in r['messages']) for r in rows),
        'distinct_character_turns': sum(sum(m['role'] == 'assistant' for m in r['messages']) for r in distinct.values()),
        'split_policy': 'All new families train-only; original validation and test bytes unchanged',
        'limitations': 'Same-author synthetic data and review. Repetition does not add independent examples.',
        'output_sha256': {name: digest(args.output / name) for name in ('train.jsonl', 'valid.jsonl', 'test.jsonl')},
    }
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
