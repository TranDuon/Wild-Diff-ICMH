---
status: resolved
trigger: "Step 9 decode succeeds but direct execution of tools/evaluate_kgalagadi.py cannot import the top-level dataset package."
created: 2026-09-27
updated: 2026-09-27
---

## Symptoms

- Expected: after decoding two images, Step 9 evaluates them and writes the result registry.
- Actual: both images decode successfully, then the evaluator exits before parsing arguments.
- Error: `ModuleNotFoundError: No module named 'dataset'` at the first project import.

## Current Focus

hypothesis: Direct script execution puts `<repo>/tools` on `sys.path`, not the repository root that contains `dataset` and `utils`.
test: Assert the evaluator bootstraps the repository root before any project-package import.
expecting: `python tools/evaluate_kgalagadi.py ...` resolves `dataset.camera_trap_dataset` regardless of the caller's current import path.
next_action: Pull the fix in Colab and rerun Step 9; decode outputs may be overwritten safely.

## Evidence

- timestamp: 2026-09-27T16:20:00+07:00
  finding: Decode completed both 256×256 images, including 5/5 DDIM steps and output files.
  implication: Checkpoint loading, crop handling, compression and reconstruction now work.
- timestamp: 2026-09-27T16:21:00+07:00
  finding: The evaluator fails at line 15 before executing `main()`.
  implication: This is an entrypoint import-path error, not an evaluation metric or data error.
- timestamp: 2026-09-27T16:22:00+07:00
  finding: `tools/precompute_ram_tags.py`, another notebook-launched direct script, already bootstraps `Path(__file__).resolve().parents[1]`.
  implication: The evaluator needs the same established repository pattern.

## Eliminated

- hypothesis: The OOM fix did not work.
  evidence: The log shows both images cropped to 256×256, sampled 5/5, saved, and averaged successfully.
- hypothesis: The `dataset` package is absent from the pulled repository.
  evidence: Training and RAM tagging already import the same repository package; only direct execution changes `sys.path[0]`.

## Resolution

root_cause: `tools/evaluate_kgalagadi.py` was invoked as a direct script without adding the repository root to `sys.path` before top-level project imports.
fix: Bootstrap the resolved repository root before importing `dataset` and `utils`, matching the working RAM tagger entrypoint.
verification: Focused evaluator/crop/notebook tests pass 9/9; `python -m pytest -q` passes with 90 passed and 2 skipped; evaluator compiles successfully.
files_changed:
  - tools/evaluate_kgalagadi.py
  - tests/test_colab_dependency_contract.py

