# Restoring the research archive

The research is paused. Run26 is an unevaluated text adapter, not an accepted full-duplex voice release. Local heavy artifacts and environments were removed after backup verification.

## Latest text checkpoint

On an Apple Silicon Mac, recreate the separate text environment using Python 3.14 and the recorded dependency lock:

```sh
python3.14 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python scripts/restore_release.py --revision 58f6402c376d9e5040d6b2439381cc8f76f053fd
.venv/bin/python scripts/chat.py --adapter adapters/run26-step0128
```

The restore script downloads the 142 MB adapter and exact metadata from Hugging Face, validates the upload manifest, restores the pinned 2.263 GB base weight from its original publisher, and installs the training tokenizer/serialization overrides. It refuses to overwrite different local metadata. It does not load a model until the chat command. The ZIP metadata restore and overwrite protection have a focused fixture test. A fresh full model restoration was not run, to avoid immediately recreating the files removed for space.

## Selected historical checkpoints and generated assets

Use the public [archive release](https://github.com/TheOneWhoBurns/peak-monk/releases/tag/archive-20261011). Download `retained-checkpoints-manifest.json`, its two `retained-checkpoints.tar.gz.part*` files, and optionally `project-assets-manifest.json` plus its `project-assets.tar.gz.part*` files. Each manifest lists part SHA256 values and exact file SHA256 values. Check the part hashes before extracting. Concatenate parts in lexical order and extract in a temporary directory before copying needed files into the repository:

```sh
cat retained-checkpoints.tar.gz.part* | tar -xz -C /path/to/empty-restore-directory
cat project-assets.tar.gz.part* | tar -xz -C /path/to/empty-restore-directory
```

The six retained snapshots are text run20/32, run22/140, run26/64, run26/128, and speech08/67 and speech09/48. Small adapter metadata from other runs is archival evidence; absence of that run's weights is intentional. There are no optimizer-state recovery guarantees. Exact paths and retention reasons are in `results/checkpoint-retention-20261011.json`.

Other public base/codec/render/ASR weights were not mirrored. Their pinned original URLs and verified hashes are recorded in the October 10/11 cleanup receipts, particularly `results/final-artifact-cleanup-20261011.json`. The project-assets archive preserves model metadata and authored audio/code artifacts. The speech environment can be recreated separately with Python 3.12 and `requirements-speech-dev.txt`.
