---
phase: quick
plan: 261014-twb
type: execute
wave: 1
depends_on: []
autonomous: true
allow_default_branch_commits: true
requirements: [INFRA-02, H2-03, H2-04, H2-06]
files_modified:
  - tools/build_colab_h1_recovery_notebook.py
  - tools/phase3_h2_roi.py
  - Wild_Diff_ICMH_H1_Recovery.ipynb
  - tests/test_h1_recovery_notebook.py
  - tests/test_colab_dependency_contract.py
  - docs/H2_ROI_PILOT.md
estimate:
  tokens: 18000
  raw_tokens: 18000
  tasks: 2
  confidence: low
must_haves:
  truths:
    - A new Colab L4 runtime can mount Drive, clone or safely pull main, explicitly set up H2 dependencies and its isolated detector, then display H2 controls in the existing Recovery notebook.
    - H2 setup preserves installed NumPy, SciPy, Torch and Torchvision versions and verifies training imports; detector setup creates or verifies /content/envs/h1-recovery-detect without H1 evaluation.
    - No default cell or H2 control display launches training, decode or detector inference. Preflight, visual acknowledgement, train200, eval30 and status remain explicit user actions.
    - H2 preflight stages at most ten originals, training stages only train/val originals plus author lambda2 and SD2.1 checkpoints, evaluation stages frozen dev30; completed H1 is read only from Drive.
    - CPU tests pass; live overlay approval, GPU train and GPU evaluation remain pending.
  artifacts:
    - path: tools/build_colab_h1_recovery_notebook.py
      provides: H2-only dependency and detector setup cells
    - path: Wild_Diff_ICMH_H1_Recovery.ipynb
      provides: Rebuilt canonical fresh-runtime notebook
    - path: docs/H2_ROI_PILOT.md
      provides: Linear fresh-session guide and scientific/cost guardrails
    - path: tests/test_h1_recovery_notebook.py
      provides: Cell order, safe defaults and no-H1-job contract tests
    - path: tests/test_colab_dependency_contract.py
      provides: Core-package and detector bootstrap contract tests
  key_links:
    - from: tools/build_colab_h1_recovery_notebook.py
      to: Wild_Diff_ICMH_H1_Recovery.ipynb
      via: Deterministic builder equality check
    - from: Wild_Diff_ICMH_H1_Recovery.ipynb
      to: tools/colab_h2_continuation.py
      via: Step 1/2 REPO and DRIVE_ROOT passed to explicit-action controls
    - from: Wild_Diff_ICMH_H1_Recovery.ipynb
      to: tools/phase3_h2_roi.py
      via: Required training imports and ready detector interpreter
---

<objective>
Make the existing Wild_Diff_ICMH_H1_Recovery.ipynb usable for the paired H2 pilot from a NEW Colab L4 runtime after the old runtime was deleted.

Purpose: Restore H2-only environment bootstraps without invoking historical H1 gates or spending GPU compute automatically.
Output: Same canonical notebook and builder, CPU contract tests, and fresh-session H2 instructions. No live H2 run.
</objective>

<execution_context>
@C:/Users/ADMIN/.codex/gsd-core/workflows/execute-plan.md
@C:/Users/ADMIN/.codex/gsd-core/templates/summary.md
</execution_context>

<context>
@AGENTS.md
@.planning/STATE.md
@.planning/quick/261014-otr-build-one-paired-h2-roi-v1-alpha2-pilot2/261014-otr-SUMMARY.md
@tools/build_colab_h1_recovery_notebook.py
@tools/build_colab_training_notebook.py
@tools/build_colab_phase2_notebook.py
@tools/colab_h2_continuation.py
@tools/phase3_h2_roi.py
@docs/H2_ROI_PILOT.md
</context>

<decisions>
- D-01: One canonical Recovery notebook in a NEW L4 runtime. Start with Step 1 Drive mount and Step 2 clone/safe-pull main; run H2 setup before the existing preflight/acknowledge/train200/eval30/status controls. No second notebook.
- D-02: H2 training setup follows existing H1 Step 5: filter requirements-colab NumPy/SciPy lines; constrain installed NumPy/SciPy/Torch/Torchvision; install CompressAI 1.2.8 with no dependencies/build isolation; install vendored RAM; refresh import caches; verify imports and unchanged core versions. No H1 review acknowledgement or H1 training.
- D-03: H2 detector setup follows H1 Step 8 prefix at /content/envs/h1-recovery-detect: virtualenv with Colab system sites, current huggingface-hub compatibility, PytorchWildlife import check and .ready marker. It must be self-contained and never evaluate H1.
- D-04: Retain existing staging: at most ten preflight originals; train/val originals plus author lambda2 and SD2.1 checkpoints for training; frozen dev30 originals for H2 evaluation. Cached H1 control on Drive is read only.
- D-05: No Run All, implicit GPU job, H1 train/eval, alpha sweep, full-dev/test, live Colab run or phase-complete claim. User explicitly approves real overlays before training; implementation verification is CPU only.
- D-06: Preserve H1 code/controls, run directories and scientific pairing. Dev30 and MegaDetector pseudo-reference limit conclusions; setup/evaluation costs are additional to the approximately 19-minute training estimate.
</decisions>

<tasks>

<task type="tracer" tdd="true">
  <name>1. Wire the fresh-runtime H2 setup through the canonical notebook</name>
  <files>tools/build_colab_h1_recovery_notebook.py, Wild_Diff_ICMH_H1_Recovery.ipynb, tests/test_h1_recovery_notebook.py</files>
  <behavior>
    - Step 1 and Step 2 provide DRIVE_ROOT/REPO; explicit H2 setup runs before the H2 controls and checks an L4 GPU.
    - H2 training setup can run without require_review_acknowledgement or H1 train/evaluate, and its default state installs nothing. It also prepares the pre-existing ./checkpoints model-config link and pycocotools needed when Step 3 is skipped.
    - H2 detector setup can run without H1 Step 8, verifies the expected interpreter and .ready marker, and dispatches no H1 evaluation.
    - Existing H1 heavy cells stay gated; displaying H2 buttons starts no install, copy, train, decode or inference.
  </behavior>
  <action>Per D-01/D-02/D-03/D-05, add explicit H2-only dependency and isolated detector setup cells after the retained H1 section and before its H2 continuation. Reuse the established H1 Step 3/5 dependency setup and Step 8 detector prefix, including pycocotools==2.0.11 --no-deps for new runtimes that skip H1 Step 3. Prepare the checked ./checkpoints symlink to /content/data/wild_diff_icmh/checkpoints for configs/model/diffeic.yaml sync_path; the H2 runner will stage the two approved checkpoint files later. Give the H2 detector cell an independent H2-namespaced logging helper; it cannot rely on executing H1 Step 8. Keep setup off by default or behind explicit controls, check Step1/2 and the L4 before installation, and fail closed on errors. Retain existing H1 Step5/Step8 code and flags. Rebuild Wild_Diff_ICMH_H1_Recovery.ipynb from the builder; adjust top/H2 guidance so it does not tell a new session to preserve deleted /content. Extend CPU notebook tests for cell order, standalone H2 setup, default no-job behavior, H1 call isolation, builder equality and compilation; update the existing AST heavy-cell gate test to recognize explicit H2 setup flags while still rejecting unguarded setup. Do not change tools/phase3_h2_roi.py staging or its cached H1 contract per D-04.</action>
  <verify><automated>&amp; 'C:/Users/ADMIN/Documents/Codex/2026-10-07/b-n-c-d-n-n/h1-timer-cpu-venv/Scripts/python.exe' -m pytest tests/test_h1_recovery_notebook.py -q -p no:cacheprovider</automated></verify>
  <done>The one generated Recovery notebook offers Step1 → Step2 → explicit H2 setup → H2 controls on a fresh L4; no H1 train/eval call or default GPU job occurs. The existing H2 bounded staging and cached H1 control remain intact per D-04.</done>
</task>

<task type="auto" tdd="true">
  <name>2. Prove setup safety and document the new-session sequence</name>
  <files>tests/test_colab_dependency_contract.py, docs/H2_ROI_PILOT.md</files>
  <behavior>
    - Generated H2 setup preserves NumPy/SciPy/Torch/Torchvision before/after versions and verifies the CompressAI/RAM/training import closure; it includes pycocotools and the guarded checkpoint link without staging images.
    - H2 detector setup owns its log and readiness check and never dispatches H1 evaluation.
    - Guide sequence is L4 → Step1 Drive → Step2 main → explicit H2 dependencies and detector setup → H2 controls preflight/inspect/ack/train200/eval30/status.
  </behavior>
  <action>Per D-01 through D-06, add CPU-only source/AST tests for the H2 dependency filter, core constraints/version assertion, CompressAI no-deps, RAM import visibility, virtualenv system-site reuse, huggingface-hub compatibility, PytorchWildlife import and detector ready marker. Update docs/H2_ROI_PILOT.md for the deleted runtime: /content assets are gone but the Drive H1 control remains; name exactly which new H2 setup cells to execute. Keep the prior same-session route only as a separate historical note. State the D-04 asset scopes, hash-bound visual approval, H2-only resume, cached H1 comparison, setup/train/eval CU caveats and dev30 inference limits. Explicitly prohibit Run All and historical H1 4–8. Run the focused CPU pack and git diff --check; do not run Colab or claim GPU/visual verification. Hand the scoped implementation commit to the orchestrator for the requested push to main.</action>
  <verify><automated>&amp; 'C:/Users/ADMIN/Documents/Codex/2026-10-07/b-n-c-d-n-n/h1-timer-cpu-venv/Scripts/python.exe' -m pytest tests/test_colab_dependency_contract.py tests/test_h1_recovery_notebook.py tests/test_phase3_h2_roi.py -q -p no:cacheprovider</automated></verify>
  <done>The fresh-run guide matches the notebook, CPU tests prove setup isolation and core-package invariants, and only the six listed files have implementation changes. Live overlays, H2 train and H2 eval remain pending user actions.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| GitHub main → Colab kernel | Notebook and script execution must follow the safely cloned/pulled main revision. |
| Package sources → Colab kernel | Installs could mutate the preinstalled CUDA/core package set. |
| Drive artifacts → local runtime | H1 evidence and H2 assets must stay source-bound and split-scoped. |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-quick-01 | Tampering | H2 pip setup | high | mitigate | Reuse existing dependency filter, core constraints, explicit CompressAI no-deps and vendored RAM source; assert versions before/after per D-02. No new package selected. |
| T-quick-02 | Denial of service | Colab CU budget | medium | mitigate | Explicit setup and separately gated H2 train/eval; document setup and evaluation cost outside the training estimate per D-05/D-06. |
| T-quick-03 | Tampering | Drive H1/H2 pairing | high | mitigate | Preserve the runner's provenance/acknowledgement and D-04 staging gates; no H1 evaluation from detector setup. |
</threat_model>

<source_audit>
| Source | Item | Coverage |
|--------|------|----------|
| GOAL | Fresh L4 runtime runs the approved H2 path in the same Recovery notebook | Tasks 1–2 |
| REQ | INFRA-02, H2-03, H2-04, H2-06 remain reachable through H2 preflight/train/eval | Tasks 1–2; existing runner retained |
| RESEARCH | No quick-task RESEARCH.md; reuse established H1 dependency and detector protocols | Task 1 |
| CONTEXT | D-01 through D-06 above; no new CONTEXT.md or deferred ideas | Tasks 1–2 |
</source_audit>

<verification>
Use the previously proven local CPU interpreter. Rebuild the notebook, run focused tests and git diff --check. Confirm implementation changes are scoped to the six files. No live Colab run or scientific approval is implied.
</verification>

<success_criteria>
The user has a linear new-L4-session route in Wild_Diff_ICMH_H1_Recovery.ipynb without H1 cells, H2 actions remain explicit/bounded, the setup protects Colab core packages, and CPU tests pass. Pushing the implementation to main is the executor/orchestrator handoff, not a planner action.
</success_criteria>

<output>
Create .planning/quick/261014-twb-fix-existing-recovery-notebook-for-a-new/261014-twb-SUMMARY.md after implementation and CPU verification.
</output>
