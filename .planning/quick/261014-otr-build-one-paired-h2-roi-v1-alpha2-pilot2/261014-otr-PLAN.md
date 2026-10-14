---
phase: quick
plan: 261014-otr
type: execute
wave: 1
depends_on: []
autonomous: true
allow_default_branch_commits: true
requirements: [H2-01, H2-03, H2-04, H2-06, INFRA-02]
files_modified:
  - train.py
  - configs/train_kgalagadi_h2_roi_pilot.yaml
  - tools/phase3_h2_roi.py
  - utils/h2_training_diagnostics.py
  - tools/h2_roi_preflight.py
  - tools/colab_h2_continuation.py
  - tools/build_colab_h1_recovery_notebook.py
  - Wild_Diff_ICMH_H1_Recovery.ipynb
  - tests/test_h2_roi_loss.py
  - tests/test_phase3_h2_roi.py
  - tests/test_h2_training_diagnostics.py
  - tests/test_h2_asset_scope.py
  - tests/test_h1_recovery_notebook.py
  - docs/H2_ROI_PILOT.md
  - COLAB_TRAINING.md
estimate:
  tokens: 42000
  raw_tokens: 42000
  tasks: 3
  confidence: low
must_haves:
  truths:
    - One explicitly approved ROI-V1 alpha2/background1 run is capped at 200 optimizer steps and compared with the cached completed H1 recovery200 uniform control.
    - The user continues inside the current Recovery notebook and L4 runtime without losing or recopying valid local images, checkpoints or environments.
    - H2 and H1 share author lambda2 initialization, fresh optimizer, pooled data, crop/transforms, seed, learning rates and 200 steps; only guide-loss ROI weighting differs.
    - Real transformed mask overlays and a source-bound acknowledgement precede training; malformed provenance, missing masks or foreign checkpoints cannot launch a heavy job.
    - Only H2 is decoded/evaluated on frozen dev30; H1/B0/Xie inference, sweeps, full-dev, test and extensions are absent from the runnable pilot surface.
    - ROI/background loss, mask fraction, rate components and learning rates are recorded in a separate H2 CSV/checkpoint identity without changing H1 artifacts.
  artifacts:
    - path: tools/phase3_h2_roi.py
      provides: Source-bound preflight, bounded training/resume, cached paired evaluation and status CLI
    - path: configs/train_kgalagadi_h2_roi_pilot.yaml
      provides: Paired low-LR pooled512 ROI-V1 alpha2 configuration
    - path: utils/h2_training_diagnostics.py
      provides: H2-only component CSV and checkpoint identity callback
    - path: tools/h2_roi_preflight.py
      provides: Deterministic real training-transform overlays and acknowledgement evidence
    - path: tools/colab_h2_continuation.py
      provides: Same-kernel explicit-action continuation interface
    - path: Wild_Diff_ICMH_H1_Recovery.ipynb
      provides: Existing canonical notebook updated in place with linear H2 continuation
  key_links:
    - from: tools/phase3_h2_roi.py
      to: configs/train_kgalagadi_h2_roi_pilot.yaml
      via: Validated paired contract before train.py invocation
    - from: configs/train_kgalagadi_h2_roi_pilot.yaml
      to: model/diffeic.py
      via: Existing roi_loss_enabled/roi_weight/roi_background_weight guide-loss consumer
    - from: tools/h2_roi_preflight.py
      to: dataset/camera_trap_dataset.py
      via: Actual processing resize and joint crop/flip path rather than an independently redrawn transform
    - from: tools/colab_h2_continuation.py
      to: tools/phase3_h2_roi.py
      via: Explicit user-selected actions, streamed subprocess logs and immutable acknowledgement hash
    - from: tools/build_colab_h1_recovery_notebook.py
      to: Wild_Diff_ICMH_H1_Recovery.ipynb
      via: Deterministic rebuild preserving old H1 section and adding the same continuation surface
---

<objective>
Implement the approved single paired H2 ROI-V1 pilot200, retaining the completed H1 recovery200 as a cached control and the user's currently active notebook/runtime/assets.

Purpose: Advance the animal-prioritization mechanism in one implementation batch followed by one concentrated CPU validation batch. The 30-image pilot selects a direction; it cannot complete Phase 3/4 or prove a paper-level win.
Output: Separate H2 config/run/contracts/logging/evaluation, real overlay preflight, an in-place Recovery notebook update, and a once-only continuation cell for the notebook already open.
</objective>

<execution_context>
@C:/Users/ADMIN/.codex/gsd-core/workflows/execute-plan.md
@C:/Users/ADMIN/.codex/gsd-core/templates/summary.md
</execution_context>

<context>
@AGENTS.md
@.planning/PROJECT.md
@.planning/STATE.md
@.planning/ROADMAP.md
@.planning/REQUIREMENTS.md
@tools/phase3_h1_recovery.py
@tools/phase3_h1.py
@utils/training_diagnostics.py
@utils/checkpoint_contract.py
@model/diffeic.py
@dataset/camera_trap_dataset.py
@tools/build_colab_h1_recovery_notebook.py
@C:/Users/ADMIN/Documents/Codex/2026-10-07/b-n-c-d-n-n/h1-recovery-completed-261014.json
</context>

<decisions>
- D-01: Exactly one V1 trial: foreground alpha2, background1, semantic ROI off and semantic coefficient0; no alpha sweep or other mechanisms.
- D-02: Match completed H1_low_lr_200: same author lambda2 weights and fresh optimizer, seed20260916, pooled20 sites, processing long-side512, uniform crop256/bbox probability0, identical flips, main LR1e-5/aux LR1e-3, accumulation8, 200 total optimizer steps. Do not initialize H2 from the trained H1 checkpoint, which would give only H2 another 200 updates.
- D-03: Modify Wild_Diff_ICMH_H1_Recovery.ipynb and its builder in place. Continue in the current L4 runtime; preserve /content. A pull changes repository files, not the cells of an already-open Colab notebook. Because that browser tab is inaccessible to the agent, ship one once-only continuation cell appended in that same notebook/kernel. Subsequent fixes use its existing safe pull and rerun the continuation, with no notebook/runtime replacement.
- D-04: Reuse H1/B0 results and local assets/environments. Evaluation stages only selected30 originals. No H1/B0/Xie inference, H3, full-dev, held-out test or automatic extension.
- D-05: Before GPU, verify paired source/data/config, generate approximately10 real transformed overlays and require their explicit source-bound acknowledgement. After implementation, run one concentrated CPU validation pack, not repeated trial runs.
- D-06: Preserve existing H1 configs, headers, checkpoint keys, runs and reports. H2 has its own run/artifact namespace, diagnostics and identity. Claims remain qualified by dev30, different achieved bitrates and MegaDetector pseudo-reference.
</decisions>

<implementation_notes>
Discovery is Level0: existing ROI loss, joint transforms, training, snapshot, evaluator and notebook patterns are already present; no new package, external service integration or architecture choice is needed. Keep the installed Colab core packages and ready detector environment. No package-manager install task is introduced.

H1 control on Drive is runs/h1_low_lr_recovery_ls512/lambda_2 with experiment_contract.json, checkpoints/last.ckpt and loss_components.csv; cached evaluation is phase3/h1_low_lr_recovery/eval/step_000200_quick with H1.jsonl, H1.detections.jsonl, comparison.json and done.json. H1's stale conditional_low_lr_trial recommendation is historical audit output, not permission to train H1 again.

Use H2 run runs/h2_roi_v1_alpha2_ls512/lambda_2 and artifact root phase3/h2_roi_v1_alpha2. H2 method ID is H2_roi_v1_alpha2_200. Author checkpoint relative path remains difficmh_models/CNscale1.0_1_1_2_2_WTagGCM_bs16x1_lr0.00005_cfg7.0/model.ckpt. Frozen dev30 SHA256 is f1f175c1df398f624ec101565fdd12a382cd618d0b4f22ad7fff411fbe9609e8; actual completed control contract SHA256 is 73061ee145b3f2010c692daa43587ccf281edb5a9fedcb1fa8b51a3cdaa1dbe0. Bind actual files and metadata rather than writing evidence from intended flags. User-owned trusted checkpoints may be loaded only from validated run/author paths, never arbitrary report-supplied paths.

The existing normalized guide weighting is in DiffEIC._weighted_spatial_mean, ROI/background means in _region_spatial_mean, and p_losses already adds weighted l_guide to the optimization loss. Preserve the existing uniform path. Test the production functions and branch, not a copied formula. CameraTrapDataset.__getitem__ applies processing resize and its _joint_transform jointly transforms image/mask. ROI-known and coverage must be verified for selected train/validation records; an absent detection record is not the same thing as a valid empty-frame record.

Sequential quick dispatch, isolation none; task-scoped main commits are authorized. Root owns final STATE/SUMMARY updates and push. Implement tasks together, then run their automated checks in the final batch. Do not force per-task RED/GREEN loops. Local validation uses the already-existing h1-timer-cpu-venv CPU interpreter; actual Colab GPU training/evaluation and visual acknowledgement remain user actions.
</implementation_notes>

<tasks>

<task type="tracer">
  <name>1. Wire the paired ROI pilot from validated control through bounded training and separate diagnostics</name>
  <files>configs/train_kgalagadi_h2_roi_pilot.yaml, tools/phase3_h2_roi.py, utils/h2_training_diagnostics.py, tests/test_h2_roi_loss.py, tests/test_phase3_h2_roi.py</files>
  <action>Per D-01/D-02/D-06, create a new config inheriting the preserved low-LR H1 chain and overriding only ROI guide flags/weights, H2 root and its separate diagnostics callback. Do not use the legacy H2 config, which changes site/LR/crop/duration. Implement CLI actions preflight, train, evaluate and status with no arbitrary alpha, target extension or test scope. Preflight validates the actual completed H1 contract, 200-step completion/evaluation markers, config-chain hashes, exact ordered dev30, source author digest, tags/detections/manifest/illumination and actual processing/decode metadata. Freeze a versioned H2 contract and acknowledged preflight before launch; reject orphan nonempty runs, changed contracts, missing records and foreign resumes. New runs warm-start the same author weights without optimizer restoration; subsequent runs resume only H2 last.ckpt with its own identity key wild_diff_h2_roi_contract_sha256, exact two optimizer groups/rates and global step at most200. Preserve session-time-limit behavior, max200 no-op, atomic checkpoint/snapshot and streamed logs. Implement an H2-only callback recording existing rate/loss/LR columns plus l_guide_roi, l_guide_bg and roi_fraction with no extra forward pass, and stamp/validate the H2 contract identity. Add production tensor/gradient tests: alpha1 equals uniform, alpha2 changes foreground/background relative gradients and reaches the total optimization objective, empty/full/tiny/downsample masks stay finite, and missing/malformed ROI is caught before GPU. Add a tiny real CPU Lightning fit/resume test for identity, optimizer states and bounded global step without loading Diff-ICMH weights.</action>
  <verify><automated>Run in the final batch: &amp; 'C:/Users/ADMIN/Documents/Codex/2026-10-07/b-n-c-d-n-n/h1-timer-cpu-venv/Scripts/python.exe' -m pytest tests/test_h2_roi_loss.py tests/test_phase3_h2_roi.py -q -p no:cacheprovider</automated></verify>
  <done>A real CPU traced fit/resume exercises the new config/identity/logging contract, ROI gradients are proven, unsafe provenance/resumes fail closed and no code changes existing H1 artifacts or launches a GPU job by default.</done>
</task>

<task type="auto">
  <name>2. Add source-bound real overlays and same-notebook continuation without recopying valid local assets</name>
  <files>tools/h2_roi_preflight.py, tools/colab_h2_continuation.py, tools/build_colab_h1_recovery_notebook.py, Wild_Diff_ICMH_H1_Recovery.ipynb, tests/test_h1_recovery_notebook.py</files>
  <action>Per D-03/D-04/D-05, create a CPU overlay path using the actual dataset processing/joint-transform code and deterministic local RNG state that is restored afterward. Select approximately10 reproducible train cases spanning daylight/night, small/ordinary animals and empty frames; record image IDs, paths, source/detection hashes, transform settings, seed and coverage. Do not modify the training sampler to manufacture animal-containing crops. Show original processed image, transformed crop/mask and latent-scale mask with truthful unavailable-case reporting and fail closed when source evidence cannot be checked. Freeze overlay report/contact sheet and require explicit acknowledgement of its hash plus verified paired preflight before training. Add a same-kernel continuation interface with explicit preflight/acknowledge/train/evaluate/status actions; displaying it never installs, copies full corpus or launches training/evaluation. Each action shows its scope, budget and progress, reports errors with the full saved log path and enforces the CLI gate. Reuse REPO/DRIVE_ROOT and existing safe_pull; the once-only cell runs this trusted local repository entrypoint after safe pull, so future code fixes require no repeated copy. Update the existing notebook builder and canonical notebook in place to expose the same linear H2 path after the preserved H1 section, with H1 heavy flags off. Stage overlays' originals only for preflight, train/validation union only for training, and the frozen30 only for evaluation; valid existing size-checked files are skipped, partial copies are atomic, path traversal is rejected. Reuse the existing ready /content/envs/h1-recovery-detect and installed core packages; if a required environment is not usable, report the explicit repair action rather than silently recreating it or resetting the runtime. Never call the old all-corpus prepare routine for quick evaluation.</action>
  <verify><automated>Run in the final batch: &amp; 'C:/Users/ADMIN/Documents/Codex/2026-10-07/b-n-c-d-n-n/h1-timer-cpu-venv/Scripts/python.exe' -m pytest tests/test_h1_recovery_notebook.py tests/data/test_camera_trap_dataset.py -q -p no:cacheprovider</automated><human-check>In the existing Colab runtime, append the single continuation cell, view real overlays and explicitly acknowledge them before any training action. Confirm it reports reuse of existing local assets and environment.</human-check></verify>
  <done>The generated notebook remains the same filename with valid Python cells and existing H1 content preserved. The once-only appended cell serves the new flow in the currently active kernel, has no default heavy side effect and enforces overlay acknowledgement. Tests prove cache reuse, scoped staging and source/image-mask transform alignment.</done>
</task>

<task type="auto">
  <name>3. Finish cached paired evaluation/reporting and run one concentrated validation pack</name>
  <files>tools/phase3_h2_roi.py, tests/test_phase3_h2_roi.py, tests/test_h2_training_diagnostics.py, docs/H2_ROI_PILOT.md, COLAB_TRAINING.md</files>
  <action>Per D-01/D-04/D-05/D-06, complete H2-only snapshot/decode50/quality/LPIPS/MegaDetector/machine evaluation for the exact frozen30 at seed231, cfg3, long-side512 and original evaluation sizes. Read H1 cached per-image quality/predictions plus its source-bound done/contract and actual decode metadata; validate identical ordered IDs and protocol before calculating a paired report. Freeze H2 evaluation identity and actual run_info, skip complete verified archives/quality/prediction stages, and reject stale or contradictory markers; no command may invoke H1/B0/Xie inference. Report both achieved bitrates and deltas in quality, pseudo-reference detection AP, missed-frame rate, empty-frame false positives and candidate hallucinations with their proper denominators, ROI/background logging and measured/estimated resource costs. Missing logged components are unavailable, not zero. Replace stale H1-trial guidance in the new H2 report with explicit pilot-only next-step status; never automatically extend or mark a whole phase complete. Write concise operational instructions emphasizing one cell in the existing notebook/runtime, safe pull for repairs, manual GPU acknowledgement and reuse of completed H1. Finish all changes first; then run one focused pack plus the full CPU suite once, preserving the old config-chain bytes and generated-notebook reproducibility. If defects are found, fix them and rerun only affected tests; record GPU/visual checks as pending rather than claiming them passed.</action>
  <verify><automated>Final concentrated batch from D:/PTHTTM/Wild-Diff-ICMH: &amp; 'C:/Users/ADMIN/Documents/Codex/2026-10-07/b-n-c-d-n-n/h1-timer-cpu-venv/Scripts/python.exe' -m pytest tests/test_h2_roi_loss.py tests/test_phase3_h2_roi.py tests/test_h2_training_diagnostics.py tests/test_h1_recovery_notebook.py tests/test_phase3_h1_recovery.py tests/test_review_h1_cached.py tests/data/test_camera_trap_dataset.py -q -p no:cacheprovider; then the same interpreter with -m pytest tests -q -p no:cacheprovider. Set PYTHONUTF8=1 and TEMP/TMP plus unique basetemp outside the repository as in prior successful CPU runs. Run git diff --check and rebuild/compile notebook cells; compare protected H1 config-chain hashes to the pre-edit baseline.</automated></verify>
  <done>The same-source, same-duration H2-versus-cached-H1 pilot is fully wired and CPU validated, original H1 tests/configs remain intact, the user receives one same-runtime continuation procedure, and no GPU run/full-phase completion or paper-level superiority is claimed.</done>
</task>

</tasks>

<threat_model>
ASVS level1; high severity findings block readiness.

| Trust boundary | Protection |
|---|---|
| Drive/JSON/manifest inputs into local staging and experiment selection | Validate record schema, hashes, exact IDs and resolved contained paths; strings are data, not commands |
| Notebook controls into expensive GPU actions | Explicit action and immutable report/overlay acknowledgement; fixed200 limit and finite bounded session hours |
| Checkpoint into torch deserialization/resume | Only user-trusted author or contained H2 run paths; own contract identity, optimizer rates/state and step bounds |
| Cached results into scientific conclusions | Actual decoder metadata and source digests; paired protocol gates and pilot/pseudo-reference caveats |

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|---|---|---|---|---|---|
| T-261014-01 | Tampering | H2/control contracts and results | high | mitigate | Freeze and verify digests, exact ordered selections, source author and configuration; reject contradictions before jobs |
| T-261014-02 | Elevation of privilege | Checkpoint loading and copy paths | high | mitigate | Only trusted contained checkpoint paths, traversal rejection and no arbitrary supplied executable paths |
| T-261014-03 | Denial of service | GPU/copy/evaluation dispatch | high | mitigate | Default-off heavy jobs, explicit acknowledgement, fixed200 cap, scoped/cache-aware staging and no expanded scope |
| T-261014-04 | Repudiation | Experiment provenance and claims | medium | mitigate | Separate immutable H2 identity, actual logs/decode metadata and caveated paired report |
| T-261014-SC | Tampering | Package installation | low | accept | No new package-manager install task; reuse already-validated installed environments and report missing prerequisites |
</threat_model>

<source_audit>
This is the explicitly approved bounded quick pilot, not a plan claiming to implement all of Phase3/4.

| SOURCE | ID | Item | Task | Status |
|---|---|---|---|---|
| GOAL | Quick261014-otr | One paired ROI-V1 pilot and same-runtime notebook continuation | 1/2/3 | COVERED |
| REQ | H2-01 | Latent guide-loss ROI weights and real gradient effect | 1 | COVERED |
| REQ | H2-03 | Separate ROI/background component logging | 1/3 | COVERED |
| REQ | H2-04 | Real transformed overlay evidence before training | 2 | COVERED |
| REQ | H2-06 | Empty-frame FP measured for the one approved alpha2 point | 3 | COVERED within quick scope |
| REQ | INFRA-02 | Existing uniform crop statistics and real mask evidence | 1/2/3 | COVERED within quick scope |
| RESEARCH | Existing-code evidence | Existing ROI consumer, joint transforms, provenance/resume/evaluation paths | 1/2/3 | COVERED |
| CONTEXT | D-01 | Single alpha2 V1, semantic disabled | 1/3 | COVERED |
| CONTEXT | D-02 | Exact paired source/duration/data/optimizer control | 1/3 | COVERED |
| CONTEXT | D-03 | Same notebook/runtime with once-only appended dispatcher | 2/3 | COVERED |
| CONTEXT | D-04 | Cached control/assets, dev30 only | 2/3 | COVERED |
| CONTEXT | D-05 | Pre-GPU overlays/acknowledgement and batched final CPU checks | 1/2/3 | COVERED |
| CONTEXT | D-06 | Preserve H1 and qualify pilot claims | 1/2/3 | COVERED |

Explicitly outside this user-approved quick scope: full Phase3 completion, H2 semantic Encoder9/V2 and alpha sweeps (H2-02/H2-05), H3 and full-dev/test/paper analysis. These remain in the canonical roadmap; do not silently remove them or mark them complete. No external API integration or database schema changes are introduced. A separate H2 experiment identity is added alongside preserved H1 because existing paid-for artifacts cannot be migrated; accepted coexistence is resolved by explicit method/run identity rather than step-number guessing.
</source_audit>

<verification>
One final concentrated CPU validation batch after all implementation. No repeated exploratory GPU tests, remote mutations, new Colab VM or corpus copy. Root records actual tests and protected-config equality, then makes scoped commits/push and updates STATE/SUMMARY without changing formal roadmap completion.
</verification>

<success_criteria>
- Same Recovery notebook and current runtime can load the continuation after safe pull.
- Exactly one paired alpha2 guide-only pilot is runnable with truthful source-bound ROI overlays, explicit consent and safe200-step resume.
- H2 evaluation reuses actual H1 control results and only stages/evaluates selected30 originals.
- CPU tensor/gradient, tiny real Lightning checkpoint, provenance, staging and notebook tests pass; old H1 contracts/configs remain unchanged.
- Operational handoff explains that GPU/visual validation is pending and dev30 cannot establish a publication-level win.
</success_criteria>

<output>
Root creates .planning/quick/261014-otr-build-one-paired-h2-roi-v1-alpha2-pilot2/261014-otr-SUMMARY.md and updates STATE after implementation; executor returns changed files and final validation evidence. Planner does not commit or modify code/state/roadmap.
</output>
