# Autoresearch: peak monk

Goal: a runnable, genuinely fine-tuned open-weight Jun character with coherent responsive conversation, without a character system prompt.
Scope: training configuration and train corpus only after the initial harness is committed.
Iterations: unlimited until acceptance is met; no artificial 25-iteration cap.
Primary selection metric: mean assistant-token validation cross entropy (lower is better), on frozen independent scenario families. This measures fit, not personality quality.
Verify: `.venv/bin/python scripts/evaluate.py --adapter adapters/run01 --output results/run01.json` (use the candidate adapter/output path for each run; measures frozen validation loss and records development conversations; untouched final audit lives separately). This command never starts training.
Guard: `.venv/bin/python -m pytest -q` plus generated development conversation audit for coherence, character consistency, relevance, honesty and non-repetition. Reject a loss improvement with a material conversational regression.

Baseline is the unfine-tuned base model with identical serialization. Also compare a character-prompted baseline. Never describe author-scored outputs as independent blind evaluation. Record training time, peak memory, tokens/sec, checkpoint and generation parameters. Freeze evaluation before experiments; hash all inputs. Keep best adapters separately from latest adapters. An experiment changes one training choice or one documented data intervention. Commit, train under a stated comparable budget, evaluate, keep/discard and log failures too.

Acceptance for the local character milestone: inference from saved base+adapter without persona system text; validation loss at least 15% below untrained baseline; at least 80% of frozen development cases rated acceptable under the rubric, no severe safety/honesty failure; then at least 80% on a separate final audit, with all generated transcripts published locally for review. These are modest project-specific gates, not a public benchmark claim. If final audit fails, do not tune on its individual answers; add a new independently specified audit for a subsequent release and disclose the failed one.

The intended speech milestone is native full duplex. Text adapters do not confer this capability. Document a concrete native speech fine-tuning path and measured hardware requirements. Do not claim completion of native speech training without an actual trained speech checkpoint and interruption/overlap evaluation.

Synthetic records retain author, source influence, conversation family and split. Split entire families before expanding turns. No system persona in training samples. No evaluation prompts, answers or paraphrases in training. No bulk modern copyrighted prose, attributed fabricated monk quotes, or celebrity voice cloning.
