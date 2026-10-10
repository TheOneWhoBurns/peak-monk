# Peak Monk — Jun

An autoresearch laboratory for an original human mountain-monk character learned in model weights. **No checkpoint has passed release acceptance yet.** Actual text and native speech adapters have been trained; the remaining problem is reliable character behavior and continuity in free conversation.

Jun speaks as a human within the fiction. Historical and fictional monks inform original synthetic conversations; they are not presented as authentic transcripts or cloned voices. See `character/core.md`, `sources/manifest.json`, and `program.md` for the character, provenance, and evaluation rules.

## Current state

Model training and experimental inference run locally on an Apple M2 Mac with 8 GB memory. Heavy model jobs are serialized. There is no cloud GPU or required phone interface.

| Track | Evidence | Current limitation |
| --- | --- | --- |
| Text | Verified Qwen3-4B-Base, 4-bit, simple ChatML with no persona system prompt. The base scored 2/20 development cases; recent trained candidates score 13–17/20 under the same author rubric. | Essential continuity/control failures remain. Run20 passed development but failed its first final and pressure audits. |
| Native speech | Moshi's original temporal and depth transformers, both audio streams, Mimi codec, and learned adapters. Run07 passed two of six native development cases. Run08 lowered matched validation CE by2.71%. | Run07 identity failed; run08 failed natural introduction. Loss, token accuracy, and audio generation do not establish a character-quality pass. |
| Speech training memory | Exact full-episode gradients now stream one temporal layer at a time. The actual longest 452-frame episode plus an AdamW update passed at 564 MB peak MLX allocation. | This is a training-memory measurement, not an inference or real-time performance claim. |
| Speech inference memory | A complete24second native introduction ran with streamed temporal weights at1.11GBpeak MLX allocation and normal pressure. | It took39.45seconds per second of input audio, excluding codec work. This is offline generation, not real-time interaction. |

These are local development results, not independently judged public benchmark scores. Text and speech losses use different tokenizations and objectives and must not be compared as a common accuracy scale.

## Resource policy after the crash

A GPU kernel panic occurred during an earlier full-resident experiment. The saved diagnostic is `results/system-crash-20261009.json`. The current bounded runtimes prohibit explicit memory wiring, serialize jobs with a shared lock, and stop on abnormal system pressure, excessive process RSS, elapsed time, or swap growth. MLX's memory limit is an allocation guideline, not a guaranteed hard cap or protection against driver bugs.

Recent full-resident Moshi and text loading attempts stopped at the pressure guard. A bounded wait for OS reclamation at the load boundary did not resolve those attempts. Native speech now uses streamed inference by default in `scripts/audit_native_candidate.py`. Streamed Qwen inference passed tiny sampling/KV parity and an actual whole-history, full36-layer export check with zero target-score difference at1.50GBpeak. Six archived short/long-history replies also matched exactly at1.43GBpeak, and two streamed-prefix training updates reproduced byte-identical adapters at2.26GBpeak. See `results/text-streamed-real-probe/parity.json` and `results/text-streamed-training-parity.json`.

**Historical approximately 1× native-generation timings and 5.5 GB training peaks predate this policy.** They do not describe the current guarded runtime. The legacy resident inference scripts remain for historical reproducibility and are not the recommended execution path on this Mac.

## Active experiments

### Speech run08

`configs/speech08.json` continues from run07's best-loss adapter. It declares one shuffled pass over 67 reviewed training episodes, including ten additional overlap examples, with checkpoints at steps 16, 32 and 67. All 32 temporal adapters and the text head receive full-episode gradients. Each update durably saves both adapter and optimizer state; resume requires matching source, model, data, and configuration hashes.

All 67 updates and the three declared validations are complete. Fixed validation joint CE was 1.88053 at step16, 1.91555 at step32, and 1.84231 at step67, compared with 1.89361 for the parent under the same diagnostic implementation. The final checkpoint was selected for native review and rejected on the first essential case. Its full introduction was “Jun, there is my name. Jun, Jun.” in both the text channel and unprompted decoded-audio ASR. The name appeared, but awkward phrasing and later repetition failed naturalness. Under the predeclared screen, the remaining cases were not run. The validation set contains only two clips; its 2.71% relative loss reduction did not predict a character-quality pass. See `results/speech08-training-summary.json` and `results/speech08-native-introduction/review.json`.

The experiment changes several things together: full gradients, ten overlap examples, continuation initialization, and learning rate. It is not an isolated estimate of the effect of gradient truncation.

```sh
.venv-speech/bin/python scripts/train_moshi_disk.py --config configs/speech08.json --resume --until-step 67
```

Do not run another model process concurrently. Any resumed training requires the original hashed trainer dependencies. See `results/speech08-plan.json`, `results/speech08-step0016-training.json`, and `results/speech08-step0032-training.json`.

Two inference optimizations—reusing a temporal module and exposing a compact virtual tensor file—preserved exact emitted codes in the short actual-model probe but did not improve speed. They are optional experiments, not defaults. See `results/moshi-inference-optimization-comparison.json`.

### Text on-policy repairs

Run22's final checkpoint scored 16/20 and remains a research parent. Run23 trained corrected final replies against complete student-generated histories while caching a frozen lower backbone. Its step16 and step32 exports scored 13/20 and 16/20; both were rejected, including persistent teacher-lore failure. See `results/run23-training-summary.json` and the development ledger under `autoresearch/loop-261009-base4b-human-v2/`.

The first trajectory pilot contains 180 actual student replies across four seeded roots in two ordinary-conversation families. Late identity was often retained; incoherent continuations, caller-fact ownership, and invented history were more frequent failures. This does not establish that assistant instruction tuning causes the drift: the backbone is a base model, and matched depth controls are still limited.

Fifty-one reviewed positive student replies are included in run24 retention. Earlier erroneous replies remain context and receive no positive loss. Matched-depth mentor probes07and09 are complete and reviewed. Eight additional spines across four new families have now produced128 actual replies, reviewed for run25 in `results/dialogue-pilot-08/review-summary.json`. All these trajectories are training-only.

## Evidence and reproducibility

- `results/moshi-disk-adjoint-real-probe/result.json`: actual longest-episode gradient and optimizer memory probe.
- `results/moshi-disk-adjoint-parity.json`: tiny full-gradient, optimizer-state, and exported-parameter parity.
- `results/speech-run07-diagnostic-comparison.json`: spoken tokens, padding, EPAD and audio-codebook diagnostics versus base.
- `results/speech-run07-closed.json`: all rejected native run07 followups.
- `results/speech08-inference-plan.json`: current inference feasibility and failed full-residency attempts.
- `results/speech-current-source-audit.json`: current audio/code provenance, including corrected human-character renders. Old aggregate render manifests are historical, not authoritative for replaced examples.
- `autoresearch/`: experiment metrics and keep/discard records.

Moshi EPAD marks the boundary before the next word after padding; it is not a dialogue turn-end token. Diagnostics report it separately. See the primary-source notes in `results/native-followup-reading-20261009.json`.

Keep `.venv` for text (MLX 0.32.3 / mlx-lm 0.32.0) and `.venv-speech` for native speech (Python 3.12, MLX 0.26.5, moshi-mlx 0.3.0, rustymimi 0.4.1). They are deliberately separate. Base weights, adapters, and environments are local ignored artifacts; code, authored data, and measured result reports are versioned. Do not commit credentials.

The old final and pressure suites have been exposed by run20. Their prompts and failed answers are excluded from subsequent training. Any future release needs fresh held-out audits after development acceptance. No current result is a held-out release pass.

### Run24 and native text-transfer research

Run24 completed96 updates using32 reviewed repairs and93 retention examples with full sampled histories. Every retention example was covered and each repair appeared three times. Both exports,48 and96, independently reproduced their saved training-side token scores exactly after reload. Both scored16/20 on the development review and failed essential Mara consistency, so both were rejected. No fresh final or pressure audit was launched. The two teacher probes found strong wording dependence: asking who reliably elicited Mara, while indirect recollections often left the teacher unnamed and produced muddled continuity. Some indirect prompts permit a different teacher, so the review distinguishes definite contradictions from preferred canonical rewrites. These are training diagnostics across four roots/two families, not independent quality benchmarks. See `results/run24-plan.json` and `data/on-policy-repairs-v3/manifest.json`.

A separate text-only gradient path for Moshi passed tiny matched-batch numerical tests and guarded real-model updates. The first prose experiment used the wrong initial token and is discarded; the corrected diagnostic is described below. See `results/moshi-text-adjoint-native-start-parity.json` and `results/moshi-text-readonly-parity.json`.

Run25 is running as a distinct offline preference experiment. The expanded training-only sampling produced128 replies across four additional families;64 repairs and23 coherent continuations were explicitly selected. After adding earlier data and excluding a length-censored alternative, the curriculum has95 preference pairs and116 retention rows. The cached-prefix DPO calculation passed full-model numerical checks and two actual guarded updates; the full-backbone export check reproduced every selected token score exactly. The declared128-update run first caches complete histories and frozen reference scores, then trains the top six adapter layers. No development results are available yet. See `results/run25-plan.json`.

Native prose diagnostics found that Moshi requires its native initial token32000 for this path: base CE on one authored prose sample was3.34 with that token versus10.23 with SentencePiece BOS1. The trained speech08 adapter scored2.94; one corrected disposable text update scored2.84 on the same training sample. This is implementation evidence, not speech-quality improvement. The wrong-initial-token probe is discarded. A24-training/6-validation original fictional notebook corpus is prepared, with no mixed training run started. See `results/moshi-text-initial-token-diagnostic.json` and `data/native-notebook-v1.json`.

Speech09 is registered in `results/speech09-plan.json`: alternate24 native-start prose updates with24 native audio replay updates, using separate AdamW states on shared adapters. Numerical mixed-optimizer verification and the guarded two-update actual preflight are pending. It has not started and makes no transfer-quality claim.

The existing67 native training clips total25.14minutes, with a36.16second maximum. Six original eight-exchange conversations are drafted in `data/speech-long-dialogues-v1.jsonl`, including late identity/teacher questions and updated caller-fact recall. Their text passed same-author continuity and exact prompt-exclusion checks. Audio rendering, unprompted ASR/timestamp review, measured duration, and encoding remain pending; they are not part of speech09. See `results/speech08-duration-audit.json` and `results/speech-long-dialogues-v1-plan.json`.
