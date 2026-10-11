---
license: apache-2.0
base_model: johnlakness/Qwen3-4B-Base-mlx-4Bit
base_model_relation: adapter
library_name: mlx
pipeline_tag: text-generation
tags:
- mlx
- lora
- qwen3
- character
- experimental
---
# Jun — Peak Monk, run26

An experimental MLX LoRA adapter for an original fictional mountain monk. The training objective is stable, responsive first-person character conversation without a persona system prompt. Jun speaks as human within the fiction; these experiments do not establish subjective belief or consciousness.

**Unevaluated research checkpoint. Training completed; conversation evaluation could not run within this Mac’s memory-pressure guard. This is a text adapter, not a full-duplex voice model.**

## What is included

The latest run26 step128 adapter (142,063,861 bytes), its MLX loading configuration, exact base metadata/serialization overrides, training configuration and provenance. The adapter contains inherited factors from earlier character training; run26 updates 11,010,048 parameters in the upper twelve transformer layers and preserves the lower-layer and embedding factors. The inherited `adapter_config.json` describes module construction; `train26.json` and `training-metadata.json` describe this run's actual optimization.

Base: community 4-bit MLX conversion of Qwen3-4B-Base, pinned to `76aa803fb5ee833aead9414f16f0708c901c21f7`. The base weight file is 2,263,022,529 bytes and is downloaded separately. The conversion was not independently reproduced. The adapter is not a standalone model or a Transformers/PEFT checkpoint.

Final adapter SHA256: `920c7a782a6c55406290fcc8404dcd9fa0a8d23997e7f09ce57dd2bcababa059`.

## Training

All work ran locally on an Apple M2 Mac with 8 GB unified memory. Run26 used 128 offline preference updates, 95 reviewed preference pairs, 116 retention examples, DPO beta 0.1, chosen-response NLL weight 0.2, retention weight 1, learning rate 1e-5 and seed 293. Histories were capped at 2,048 tokens. The upper twelve adapter layers were updated, compared with six in run25. Training completed in 45.41 minutes including history preparation, peaking at 2.735 GB MLX allocation. Model jobs are serialized.

Training material consists of original synthetic fiction, including student-generated conversation histories with author-reviewed alternative final replies. Historical and fictional monks are influences, not sources of attributed invented quotations. No real monk's voice was cloned.

## Evaluation and limitations

Run26 has no completed conversation evaluation or release acceptance. Export verification and evaluation attempts stopped at the macOS memory-pressure guard; aborted attempts are preserved in the research repository. The user elected to publish and back up the checkpoint before finishing evaluation.

For historical context only, the unchanged text base scored 2/20 on the project development suite with CE 3.892766. Run25 step128 scored 16/20 but failed required character lore. Those scores do not describe run26. Run20 scored 17/20 on development but failed its subsequent final and pressure audits. No checkpoint has passed the full release criteria.

These are same-author project audits, not independent or blinded public benchmarks. The development suite has been repeatedly inspected during research. Acceptance additionally requires separate frozen final and pressure audits. Cross entropy is a diagnostic, not a conversational accuracy percentage. All passes and failures belong in the research repository.

The local chat implementation limits input history to 2,048 tokens and each reply to 160 generated tokens; it stops before an oversized input instead of silently truncating. This describes the tested implementation, not the upstream base model's maximum context. Long-conversation drift remains a central risk.

Separate native Moshi speech experiments exist in the research repository, but do not confer speech capabilities on this checkpoint. Their latest candidate failed its preservation screen; guarded native generation was about 39.45 times slower than real time. No real-time voice demo or accepted full-duplex character is claimed.

## Restore and run

Source, raw evaluation captures and restoration script: [Peak Monk research](https://github.com/TheOneWhoBurns/peak-monk).

On Apple Silicon, create a Python environment and install the repository's `requirements.lock`. Then run from the repository root:

```sh
.venv/bin/python scripts/restore_release.py --revision <this-model-repository-commit-sha>
.venv/bin/python scripts/chat.py --adapter adapters/run26-step0128
```

Restoration verifies the published checksums, downloads the exact pinned base weight and installs the included ChatML/EOS overrides. Do not substitute an instruct model or an unmodified tokenizer template. Use `/reset` and `/quit` in the CLI. The separate older `Try Jun.command` launcher points to historical run20, not this run26 adapter.

The original project was inspired by [Zero](https://movingcastles.world/posts/zero). It is an independent research experiment, not an official Zero release.
