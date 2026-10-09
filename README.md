# Peak Monk — Jun

An autoresearch laboratory for an original mountain monk character model. Work in progress: multiple text-adapter pilots have been trained and rejected; no accepted release checkpoint yet.

The project learns character behavior in weights, using original synthetic conversations with documented historical and fictional influences. All computation must fit the user's Apple M2 Mac with 8 GB memory. Text experiments use quantized Qwen models. Native full-duplex speech uses a custom Moshi MLX training path; it has passed small-model forward/gradient checks but has not yet produced a trained speech checkpoint.

The speech approach caches a frozen temporal prefix, unloads its weights, and trains only a suffix and text head with joint text/audio supervision. The split matches the full calculation's loss and gradients on the implementation test. Actual pretrained-weight memory and audio quality remain to be measured. Stock Kokoro voices provide synthetic training material; their output is never presented as the trained character's speech.

Historical text experiments use the original MLX loss. A discovered padding-boundary issue is corrected in `scripts/assistant_objective.py`; new runs use `assistant_eos_v2` and a fresh baseline. Loss numbers from these versions are not comparable. Raw generated conversations, not lower loss alone, determine whether a checkpoint passes.

See `program.md`, `character/core.md`, and `sources/manifest.json`. Model files, adapters and the Python environment are local ignored artifacts; research code, data and measured results are versioned. Never commit credentials.

Jun speaks as a human monk within the character. The human_character_v2 rubric replaces the earlier mistaken disclosure requirement; historical results retain their original rubric. New supervision removes model/fiction explanations from Jun's replies, and longer dialogue probes test persistence under pressure. The program documents provenance outside the character conversation.

The small authored corpus is a current quality limitation. Early loss improvements have not yielded an accepted character checkpoint. Rejected teacher simulations and interrupted/failed training runs are retained for inspection. Fifty speech conversations (48 train, 2 validation; 19.1 minutes of timeline) have fresh-instance Mimi codes after a reset-contamination bug was detected and corrected. These codes are prepared training inputs, not a trained voice.
