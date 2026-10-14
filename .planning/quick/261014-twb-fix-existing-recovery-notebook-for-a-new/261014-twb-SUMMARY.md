---
phase: quick
plan: 261014-twb
quick: 261014-twb
status: complete
completed: 2026-10-14
subsystem: colab-h2-pilot
tags: [colab, h2, recovery-notebook, dependency-isolation]
requires:
  - 261014-otr paired H2 ROI pilot and completed H1 recovery200 Drive evidence
provides:
  - Fresh-L4 H2 setup in the one canonical Recovery notebook
  - CPU contract tests and new-session guide
affects: [INFRA-02, H2-03, H2-04, H2-06]
requirements-completed: []
requirements-touched: [INFRA-02, H2-03, H2-04, H2-06]
implementation_commits: [70a2956]
plan_head_before: 523571a919607811e44e2bb359ef0d1fd669c8e0
commits: 1
actuals:
  tokens: 10449
  tasks: 2
  commits: 1
key-files:
  created: []
  modified:
    - tools/build_colab_h1_recovery_notebook.py
    - tools/phase3_h2_roi.py
    - Wild_Diff_ICMH_H1_Recovery.ipynb
    - tests/test_h1_recovery_notebook.py
    - tests/test_colab_dependency_contract.py
    - docs/H2_ROI_PILOT.md
experimental_status: h2_live_overlay_gpu_train_and_eval_pending
---

# Quick 261014-twb: Fresh-L4 H2 Recovery notebook setup

The existing canonical Recovery notebook now supports Step 1 Drive → Step 2 main → explicit H2 dependency and detector setup → H2 action buttons in a new Colab L4 runtime, without rerunning H1 or launching a default GPU job.

## Accomplishments

- Added opt-in H2-B training dependency setup based on the established H1 protocol: filter NumPy/SciPy requirements, constrain installed NumPy/SciPy/Torch/Torchvision, install CompressAI 1.2.8 with `--no-deps --no-build-isolation`, install vendored RAM, verify imports and core versions, add pycocotools 2.0.11, and prepare a guarded local `./checkpoints` symlink. No image or checkpoint staging occurs in the setup cell.
- Added opt-in H2-C isolated detector setup at `/content/envs/h1-recovery-detect` with H2-owned logging, Colab system packages, pinned current huggingface-hub compatibility, PytorchWildlife import verification and `.ready` validation. It makes no H1 evaluation call.
- Rebuilt `Wild_Diff_ICMH_H1_Recovery.ipynb` from its builder. H1 cells and controls remain; H2-D only displays the existing explicit preflight, acknowledgement, train200, eval30 and status controls.
- Updated the new-session guide: old `/content` is gone, while completed H1 evidence remains read-only on Drive. It states bounded preflight (≤10 originals), train/val plus two approved checkpoints, and frozen-dev30 evaluation staging, as well as setup/evaluation CU cost outside the approximately 19-minute training estimate.

## Verification

- Executor focused CPU pack: `tests/test_colab_dependency_contract.py tests/test_h1_recovery_notebook.py tests/test_phase3_h2_roi.py` — **65 passed** before the final wording fixes. After those fixes, the two directly changed notebook/dependency suites passed **41 tests**. A local full-pack rerun passed 64 but lacked `pycocotools` in that interpreter for one metric test.
- Generated notebook equals the builder and every code cell compiles; default H2 setup gates execute without assets or installs.
- `git diff --check` passed. Implementation diff is confined to the six plan-listed files; Git reported only Windows LF→CRLF advisory warnings.
- No live Colab runtime, Drive copy, GPU training, decode, detector inference, or visual overlay approval was performed.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Removed implicit pull from H2-D.** Step 2 already safely clones/pulls main before setup. A second pull when merely displaying controls could fail on local setup artifacts and would add an unnecessary network action. H2-D now verifies Step 1/2 paths and only loads the control UI. Verified by the notebook CPU tests.

## TDD Gate Compliance

The new notebook setup tests failed as intended before the builder change; the fresh-session guide test failed as intended before documentation was updated. Both were green in the executor's 65-test pack. The executor did not commit on protected `main`; the orchestrator committed the scoped implementation as `70a2956` under the user's explicit request to publish the notebook.

## Issues Encountered

The sandbox's default Windows Temp path was inaccessible to pytest fixtures. The complete focused pack passed using a task-local temporary directory, which was then removed. No test was skipped for this reason.

## Scientific and handoff status

This completes implementation and CPU verification only. The user must inspect real H2 masks and acknowledge their current hash before one bounded H2 200-step training run, then explicitly evaluate H2 on frozen dev30. Cached H1 is the read-only comparison. Dev30 and MegaDetector pseudo-reference limit conclusions; neither Phase 3 nor Phase 4 is scientifically complete. Commit `70a2956` must be pushed to `main` before the new Colab runtime loads the corrected notebook.

## Self-Check: PASSED

All six listed implementation files and this summary exist. The builder-generated notebook matches source, 41 directly affected tests passed after the final wording fixes, and implementation commit `70a2956` exists.
