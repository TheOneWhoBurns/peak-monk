# Peak Monk — Jun

An autoresearch laboratory for an original human mountain-monk character learned in model weights. **No checkpoint has passed release acceptance.** Text and native speech adapters have actually been trained, but reliable character continuity remains unresolved.

Jun speaks as human within the fiction. Historical and fictional monks inform original synthetic material; these are not authentic transcripts, attributed quotations, or cloned voices. The character and source records are in `character/core.md` and `sources/manifest.json`. Behavioral consistency can be tested; subjective belief has not been established.

## Current experiments

| Track | Latest evidence | Decision |
| --- | --- | --- |
| Text run25 | Qwen3-4B-Base, 4-bit, no persona system message. 128 offline preference updates, 95 reviewed pairs, 116 retention examples. Step 64: 14/20, CE 2.57523. Step 128: 16/20, CE 2.60996. | Both rejected. Essential late-female-teacher lore still fails; step 64 also fails persona control. |
| Text run26 | Registered comparison changing trainable adapter depth from 6 to 12, retaining run25's parent, data, objective, seed and schedule. | Preflight passed: two finite updates, normal pressure, 2.74 GB peak, both full-backbone exports exactly reproduce target scores. Full 128-update run launching with deferred suffix loading; quality remains unmeasured. |
| Native speech08 | Exact full-episode training over 67 clips. Selected step 67 reduced fixed two-clip joint CE from 1.89361 to 1.84231. | Rejected: full unforced introduction was awkward and repetitive. |
| Native speech09 | Alternates 24 native-start prose updates and 24 native audio replay updates, with separate AdamW states on shared adapters. Tiny numerical proof and actual two-update preflight passed. | All 48 updates complete. Both checkpoints rejected by the preservation screen; no native generation escalation. |

The text base scored 2/20 with CE 3.892766 under the same serialization/objective. Recent trained candidates generally score 13–17/20. Run20 passed development but failed its first final and pressure audits. These are same-author project evaluations, not independently judged public benchmarks. Text and native audio losses are different objectives and cannot be compared as a common accuracy scale.

Run25's saved full-backbone exports reproduced every checked training-side target score exactly. Training took about 32.5 minutes including history preparation, with 2.64 GB peak MLX allocation and normal system pressure. This verifies the implementation and resource run; it does not override the rejected conversations. See `results/run25-training-summary.json`, both `results/run25-step*-review.json` files, and the text ledger in `autoresearch/loop-261009-base4b-human-v2/`.

## Local use and hardware limits

Everything runs on this Apple M2 Mac with 8 GB memory. Model jobs are serialized. The text environment is `.venv` (MLX 0.32.3, mlx-lm 0.32.0); speech uses `.venv-speech` (MLX 0.26.5, moshi-mlx 0.3.0, rustymimi 0.4.1). Keep them separate.

To inspect a rejected research checkpoint after the active model job finishes:

```sh
.venv/bin/python scripts/chat.py --adapter adapters/run25-step0128
```

The CLI verifies the adapter's training base, streams frozen weights, retains the whole conversation up to its declared 2,048-token input bound, and supports `/reset` and `/quit`. It adds no persona system text. It rejects an oversized history rather than silently truncating it. `results/chat-streamed-smoke/` records the earlier actual CLI check.

Native speech uses Moshi's temporal/depth transformers, both audio streams, and Mimi; it is not a speech-to-text/text/TTS cascade. Current guarded native generation is **offline**: the full 24-second speech08 introduction took about 39.45 seconds per second of input, excluding codec work. It used 1.11 GB peak MLX allocation with normal pressure. Earlier approximately 1× native timings used a different residency policy and do not describe the current runtime.

A GPU kernel panic occurred during an earlier full-resident experiment (`results/system-crash-20261009.json`). Current runtimes prohibit explicit wiring and stop on abnormal pressure, excessive process RSS, swap growth, or elapsed time. An MLX memory guideline is not a hard cap or protection against driver bugs. Full-resident attempts subsequently stopped at the pressure guard; the default audit path now streams temporal weights. Two proposed I/O/module-reuse optimizations preserved sampled codes but did not improve timing (`results/moshi-inference-optimization-comparison.json`).

## Training and data

The latest text curriculum uses real student-generated histories with explicitly reviewed alternative final replies. Erroneous earlier replies remain context and receive no positive target loss. The larger pilot produced 128 replies across four new conversation families, admitting 64 corrections and 23 coherent replies for retention. This is offline author preference training, not GRPO or independent reward judging.

Speech09 uses native initial text token 32000. A prior prose probe using SentencePiece BOS 1 produced misleadingly high loss and was discarded. Matched diagnostics identified the initial token, rather than padding probability, as the major cause (`results/moshi-text-initial-token-diagnostic.json`). Its 24 training and 6 validation prose episodes contain original Jun fiction and no artificial final EOS. The mixed-update preflight changed all 130 adapter tensors, saved separate 33-file optimizer states, and peaked at 559 MB. The complete run peaked at 563 MB with normal pressure. Final prose CE improved 7.7%, but native spoken-token CE worsened 5.9%, above the declared 2% tolerance; see `results/speech09-training-summary.json`. Numerical and state checks are in `results/moshi-mixed-optimizer-parity.json` and `results/speech09-two-update-preflight.json`.

The 67 native training clips total 25.14 minutes; the longest is 36.16 seconds. Six original eight-exchange conversations are drafted in `data/speech-long-dialogues-v1.jsonl` to add longer coherent context and late questions. Their text is reviewed, but rendering, unprompted ASR/timestamp review, measured duration, and Mimi encoding are pending. They are not part of speech09. Stock Kokoro voices provide synthetic source audio, not Jun model-generated output. The current source audit is `results/speech-current-source-audit.json`; old aggregate render manifests contain archival examples replaced in the active encoded corpus.

Speech10 is a registered audio-only control for speech09, removing prose updates while preserving all 24 audio episodes, their order, parent, objective and optimizer parameters. Its historical schedule proof and two focused tests passed; model execution remains queued behind text work (`results/speech10-plan.json`).

## Acceptance and fresh audits

`program.md` defines the unchanged gates: at least 80% acceptable development conversations, every essential identity/control/senses/lore case, no severe failure, and at least 15% matched validation-loss reduction. A selected candidate must then pass a separate final audit and at least 7/8 pressure conversations, with no more than one hard identity break. Native speech additionally needs actual checkpoint generation and interruption/overlap evaluation.

Fresh final and pressure suites are frozen in `evaluation/release-v3-frozen.json` before run26 training. They have never been run. A pre-use amendment clarified two ambiguous teacher cues; original drafts and the reason are preserved under `evaluation/archive/release-v3-draft01/`. The 95 current new caller prompts have no exact matches across 74 tracked conversation corpora (35,818 caller occurrences). They are same-author capability tests, not semantically independent benchmarks.

`scripts/audit_prompt_exclusion.py` reads the explicit evaluation `turns` schema, rejects empty audits, verifies frozen v3 hashes, and automatically includes registered release exclusions. The first ad hoc check of the long-speech draft used the wrong schema; its invalid evidence was superseded transparently. Exact exclusion still does not establish semantic independence. No final/pressure prompts, outputs, or deliberate paraphrases may become training material.

## Local storage

The October 10 cleanup reclaimed 13.79 GB of observed free disk space (5.89 → 19.69 GB). Unused downloaded PersonaPlex and Qwen3.5 weight files were removed; exact revisions, hashes and restore URLs remain in `results/disk-cleanup-20261010.json`. These historical experiments require restoring their weights before rerunning. Current Qwen3-4B-Base, Moshi, Mimi, speech render/ASR models and both environments remain installed.

Completed run23–25 and two old probe prefix tensors were removed; their manifests, training inputs, configurations, reference scores and results remain. Regenerate those caches for any future cache-dependent verification. All unique trained checkpoints remain at their existing paths. Byte-identical completed checkpoints now share hard links; treat these archival weights as immutable and write future experiments to fresh paths. The cleanup record distinguishes actual free-space gain from file-size estimates.

## Evidence and reproducibility

- `results/run25-plan.json`, `results/run26-plan.json`, `results/speech09-plan.json`: declared experiments and selection rules.
- `results/qwen-streamed-inference-parity.json`, `results/text-streamed-real-probe/parity.json`, `results/text-streamed-training-parity.json`: full-context text streaming checks, actual matching replies, byte-identical training adapters.
- `results/moshi-disk-adjoint-parity.json`, `results/moshi-disk-adjoint-real-probe/result.json`: exact full-episode gradient/update checks and actual longest-clip resource probe.
- `results/moshi-text-adjoint-native-start-parity.json`, `results/moshi-text-readonly-parity.json`: native text gradient and read-only scoring checks.
- `results/speech08-training-summary.json`, `results/speech08-native-introduction/review.json`: trained native checkpoint and failed generation evidence.
- `results/run25-explicit-prompt-exclusion.json`, `results/release-v3-prompt-exclusion.json`, `results/release-v3-historical-exclusion.json`: counted exclusion audits and their limits.

Latest full core test run: 25 passed, 1 skipped. After the frozen-audit registry change, the two focused exclusion tests and actual corpus audits also passed. Models, adapters, and environments are local ignored artifacts; source, authored data, configurations, raw text captures, and result reports are versioned. No accepted release or publication is implied.
