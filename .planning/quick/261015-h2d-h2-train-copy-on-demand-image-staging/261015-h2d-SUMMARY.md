---
phase: quick
plan: 261015-h2d
subsystem: h2-training-data-staging
tags: [h2, copy-on-demand, dataloader, drive, colab]
requires: [H2-01, INFRA-01]
provides: [WILD_DATA_SOURCE_ROOT opt-in copy-on-demand for H2 train]
affects: [dataset/camera_trap_dataset.py, tools/phase3_h1.py, tools/phase3_h2_roi.py, train.py, docs/H2_ROI_PILOT.md]
key-files:
  modified: [dataset/camera_trap_dataset.py, tools/phase3_h1.py, tools/phase3_h2_roi.py, train.py, tests/test_phase3_h2_roi.py, tests/test_h2_asset_scope.py, docs/H2_ROI_PILOT.md]
  created: [tests/conftest.py, tests/test_dataset_copy_on_demand.py]
decisions:
  - Source root travels only via env var WILD_DATA_SOURCE_ROOT (configs are hashed into the frozen contract).
  - command_env always strips the variable and re-adds it only when p carries source_images (H2 train only).
status: complete
commits: 3
plan_head_before: ae74829bb880477a31eabfc7ae2a6fbe3f880d94
completed: 2026-10-15
actuals:
  tasks: 3
  commits: 3
---

# Quick 261015-h2d: H2 train copy-on-demand image staging Summary

H2 train now stages only the two fixed checkpoints and the dataloader copies each train/val original from `DRIVE_ROOT/images` to `/content/data/wild_diff_icmh/images` on first read, instead of pre-copying 8,690 images.

## What changed
- `dataset/camera_trap_dataset.py`: opt-in `WILD_DATA_SOURCE_ROOT`; `_contained_path` + `_copy_on_demand` (reuse local without touching source, unique `.partial-<pid>-` temp, size check, atomic `os.replace`, 3 attempts with the `_open_image` backoff, no global RNG). Unset env leaves the code path unchanged.
- `tools/phase3_h1.py::command_env`: pops inherited `WILD_DATA_SOURCE_ROOT`, sets it only when `p["source_images"]` is present.
- `tools/phase3_h2_roi.py`: `_stage(p, "train")` is checkpoint-only; `train()` prints the copy-on-demand line and launches with `dict(p, source_images=p["root"]/"images")`. Contract, paths, evaluate and configs untouched.
- `train.py::_run_asset_preflight`: verifies every selected train/val image under the source root when the env var is set (same error format); unchanged otherwise.
- `docs/H2_ROI_PILOT.md`: new staging description and the `BRANCH = 'phase3+4'` cell-2 instruction (user must set it by hand; notebook untouched).

## Tests
Command (baseline list; new run adds tests/test_dataset_copy_on_demand.py):
`python -m pytest -q -p no:cacheprovider tests/test_phase3_h2_roi.py tests/test_h2_asset_scope.py [tests/test_dataset_copy_on_demand.py] tests/test_h2_roi_loss.py tests/test_h2_training_diagnostics.py tests/test_phase3_h1.py tests/test_phase3_h1_recovery.py tests/test_h1_recovery_notebook.py tests/test_session_timer.py tests/data/test_camera_trap_dataset.py tests/test_colab_dependency_contract.py tests/test_checkpoint_contract.py tests/test_sources_compile.py tests/test_phase3_notebook_stream.py tests/test_review_h1_cached.py`

- Before: 147 passed, 6 skipped.
- After: 166 passed, 6 skipped (same environment skips; +19 new tests).
- `tests/test_h2_roi_preflight.py` still errors at collection (module-level `import torch`, absent on this Windows machine): pre-existing, recorded, not fixed.

## Gates
- `git diff --exit-code ae74829 -- configs/` clean; `*.ipynb` and `tools/build_colab_h1_recovery_notebook.py` clean.
- `_experiment_contract` key set/values asserted unchanged in the updated H2 train test.

## Deviations from Plan
None. Plan executed as written (the tracer test lives in tests/test_phase3_h2_roi.py as specified).

## Known Stubs
None.

## Push
`git push -u origin phase3+4` succeeded (new branch). Local HEAD == origin/phase3+4 == ba3ecc69630975ae37260aa6fdcf88b0f9ff9524. main untouched.

## Self-Check: PASSED
Created files exist (tests/conftest.py, tests/test_dataset_copy_on_demand.py); commits 1e6bdad, 92004ce, ba3ecc6 present on phase3+4.
