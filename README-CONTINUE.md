# Jun: continuation handoff

Paused on October 11, 2026, at the user's request. Continue on October 12 when asked; no automatic job is scheduled.

## Published and preserved

- [Latest Run26 text adapter on Hugging Face](https://huggingface.co/A-Martinez-2508/peak-monk-jun-run26), pinned to `58f6402c376d9e5040d6b2439381cc8f76f053fd`: one adapter plus metadata, 146 MB.
- [Research source and results on GitHub](https://github.com/TheOneWhoBurns/peak-monk).
- [Verified compact archive](https://github.com/TheOneWhoBurns/peak-monk/releases/tag/archive-20261011): six selected checkpoints and generated audio/codes/model metadata, about 940 MB total. Part and file checksums are included. Redundant intermediate checkpoints and optimizer copies were deliberately discarded.

## Exact research status

Run26 completed all 128 training updates, but its saved-export numerical verification and full conversation evaluation remain incomplete. There is no accepted release. Earlier attempts stopped at the Mac's memory-pressure guard; they are not quality results. The latest published adapter is explicitly unevaluated. Native speech work is a separate track and has not passed acceptance.

After publication and cleanup, a final-evaluation restore was briefly started. The user then cancelled it in favor of this handoff and another cleanup. Both download processes and their child download were stopped; the restored Python environment, partial base download and temporary archive/model bundles were deleted. No model was loaded and no new training or conversation evaluation ran during that attempt. The lightweight restore fixture test passed (1 test).

## Next session

1. Check actual macOS memory pressure and active processes before restoring or loading anything. Pressure changed from warning to normal during the cancelled restore, so re-measure rather than assuming either state. Keep Firefox tabs and unrelated work intact.
2. Restore only the text environment, pinned Qwen3-4B-Base and Run26 step64/128 exports, using [restore instructions](docs/restore-publication.md). Step128 is on HF; step64 and the two training-side `step*-export-score.json` references are in the checkpoint archive. The restore script now uses a temporary HF download directory to avoid leaving another cached adapter copy.
3. Run `scripts/verify_repair_export.py` serially for both saved exports, retaining the existing memory/RSS/swap/time guards and full-backbone score checks. The conservative streamed configuration uses zero resident layers and a bounded pressure pause; inspect the previous runtime records before choosing arguments.
4. Run both complete, unchanged development captures using `scripts/evaluate_bounded.py`: `evaluation/dev-roleplay-v2.json`, `assistant_eos_v2` loss, 160 maximum new tokens, temperature 0, top-p 1, and no persona system prompt. Preserve failed attempts and write to new result paths.
5. Review every conversation against the registered rubric. Require at least 16/20, all essential identity/control/senses/lore cases, no severe failure, and at least 15% same-base validation-loss improvement. Select by acceptable count, then lower CE. Both checkpoints may fail.
6. Only a development-qualified candidate proceeds to frozen v3 final and pressure audits. Do not train on evaluation material. If neither qualifies, publish the failed research results honestly.
7. Update the existing public model card and GitHub results. Avoid duplicate model uploads. Verify remote artifacts, stop model processes, and remove newly restored weights, environments and temporary files again, as the user requested.

Research-plan details: `results/run26-plan.json`, `results/run26-training-summary.json`, `results/run26-wrapup-handoff-20261011.json`. The older wrap-up file predates publication; its `published: false` and staging instructions are historical and superseded by this README. Publication and deletion receipts are under `results/publication-archive/`, `results/huggingface-publication-20261011.json`, `results/final-artifact-cleanup-20261011.json`, and `results/final-wrapup-cancelled-20261011.json`.
