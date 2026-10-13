---
quick: 261013-i7u
status: complete
completed: 2026-10-13
implementation_commits: [d518971, 7378428, 687e660]
experimental_status: live_trial_pending
---

# Cached H1 review and isolated low-learning-rate handoff

Quick implementation is complete; **Phase 3 and the future GPU trial are not**.

## Delivered

- CPU cache auditor recomputes B0 lambda2/8/32 and H1 step500/1000 on the exact
  same frozen 30 validation IDs, from per-image quality and cached detections.
  Actual archive run_info and H1 evaluation contracts verify long-side512,
  DDIM50, seed231, guidance3 and original-size evaluation. No learned model is
  initialized. Mismatched IDs, dimensions, split, bytes/bitrate or protocol fail
  closed; missing protocol closes the trial gate.
- Tracked Vietnamese report and versioned JSON include source IDs, selected
  record/contract hashes, denominators, descriptive bitrate brackets and missing
  evidence. Ten-case inventory contains nine actual missed-frame IDs; no images
  were downloaded/generated and no visual diagnosis is claimed. Old scalar
  component/LR logs were unavailable in this bounded local review; mounted-Drive
  CSV/TensorBoard reading is supported without automatic TensorBoard installs.
- Separate low-LR config/helper, author lambda2 weights-only initialization and
  fresh optimizer, main LR1e-5 and unchanged entropy_aux LR1e-3. Maximum200 total
  optimizer updates; fresh bounded session Timer. Both training and evaluation
  require explicit enablement plus acknowledged SHA256 of a verified review.
  No extension/full-dev/test or foreign-resume CLI options exist.
- Own checkpoint identity, LR and frozen contract are checked before resume.
  Snapshots, archives, registry, metrics, logs and resource ledger have isolated
  names. Opt-in LossCSV records current logged component scalars/actual LRs and
  stamps checkpoints without another model pass; absent components stay blank.
  Completed decode is reused after scoring failures, preserving run metadata.
- CPU-first generated recovery notebook includes permanent safe main pull and
  streamed subprocess output. All heavy flags are false. CPU report needs no
  nvidia-smi, images/checkpoint staging or detector environment; those are gated.
  No RAM++ weight copy, B0 inference or Xie recreation is added. One canonical
  new notebook link replaces copy-paste fixes; later script fixes use cell2A.
- H2 is documentary only: same source/optimizer/LRs/data/crops/seed/steps and
  evaluation between ordinary and ROI-weighted branches, only ROI loss differs.
  No H2 model change, GPU job or roadmap/phase-completion mutation.

## Actual cached findings

| Point | Original-pixel bpp | Detector AP | Missed positive frames |
|---|---:|---:|---:|
| B0 lambda2 | 0.005990175 | 0.394392636 | 0.30 |
| B0 lambda8 | 0.002863426 | 0.271557684 | 0.50 |
| B0 lambda32 | 0.000560957 | 0.165924092 | 0.60 |
| H1 step500 | 0.004319599 | 0.360321571 | 0.35 |
| H1 step1000 | 0.004216718 | 0.326651790 | 0.45 |

H1 step1000 uses29.6% fewer bits than B0lambda2, but lower detector agreement.
It lies between B0lambda8 andlambda2; no measured B0 point dominates its AP at
no greater rate. That is NOT a measured matched-rate improvement or significance
claim. The500→1000 detector decline justifies a conditional short LR hypothesis
test, not a conclusion that fine-tuning globally failed or LR caused it. Pseudo
boxes are detector outputs, not human boxes; missed rate is frame-level. The
200-step probe also differs in training duration from old500/1000 checkpoints.

Input selected-evidence SHA256:
`6135e89c5085569d41e41701cb41ebea8131530032d4e8291a1918ed3fbf04d8`.

Root independently replayed the actual bounded bundle plus observed protocol
sidecar: protocol_verified=true, recommendation=conditional_low_lr_trial. Its
JSON is byte-identical to the tracked report (file SHA256
`6a375077d3ce6ffddf12ccaee4484095efaf556e99a5c998edf026ad4aea2096`).
Raw evidence remains outside git in the thread workspace; no multi-GB checkpoint
or image archive was downloaded. Drive profile was verified as the user's
Trần Đăng Dương account; there were no Drive writes or use of the Tuấn account.

## Verification

- Intentional consent RED test preceded helper implementation. Root added and
  reproduced a RED missing-review_source failure before fixing the integration.
- Independent read-only integration review found two P1s: missing trial archive
  link and timestamp-sensitive retry after scoring failure. Both fixed, with
  regression tests. Trial loss review now points to its own CSV, not old logs.
- Full final CPU suite: `h1-timer-cpu-venv/Scripts/python.exe -m pytest tests -q
  -p no:cacheprovider --basetemp <thread>/recovery-final-full-261013`:
  **253 passed, 2 skipped, 65 warnings**, exit0. Warnings include existing Pillow,
  Lightning small-loader/mid-epoch-resume and source escape warnings. Real tiny
  Lightning fit/save/resume tests verify own checkpoint identity, optimizer LRs
  and append CSV global steps1,2,3, not just AST/static mocks.
- Notebook builder equality, code-cell compilation, CPU route, off flags,
  source-schema display, safe pull and real stdout/stderr exit propagation pass.
- `git diff --check` reviewed; old config chain, manifest, train.py, old helper,
  old pilot notebook and model/callbacks remain unchanged. New config/callback
  are separate. All commits explicitly scoped, no reset/force push.

No local test certifies a full Diff-ICMH GPU fit or scientific improvement.

## Handoff

Open `Wild_Diff_ICMH_H1_Recovery.ipynb` from main. CPU1→2→3→4 reads actual cache.
After reading/approval choose L4, rerun1–4, set RUN_TRIAL=True and
ACKNOWLEDGE_REVIEW=REVIEW_SHA256, then5→6→7. At200 set RUN_TRIAL=False and
RUN_EVALUATION=True, rerun4→5→6→8→9. Send comparison.json, throughput.json and
loss_components.csv; do not autoextend or launch H2. Approximate training-only
18.8min/0.48CU excludes setup/load/checkpoint/evaluation and is not a bill.

## Execution notes

Typed planner and executor dispatched with inherited model; executor used
disjoint audit/notebook workers. Executor/notebook worker hit model capacity
before final integration; root preserved their work and completed missing fixes,
real replay, tests and commits. No reinitialization or lost user work. Runtime
isolation was explicitly downgraded to sequential main under the approved main/
pull workflow; temporary generated isolation sentinel was removed after work.
LossCSV moved to new utils/training_diagnostics.py instead of modifying shared
model/callbacks.py, to protect the old frozen pilot. GSD summary/state record this
implementation without marking experimental Phase 3 complete.
