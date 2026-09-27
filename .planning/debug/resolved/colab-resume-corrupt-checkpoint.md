---
status: resolved
trigger: "Colab Step 8 previously succeeded, but rerunning it now fails while restoring last.ckpt with PytorchStreamReader invalid header / archive is corrupted."
created: 2026-09-27
updated: 2026-09-27
---

## Symptoms

- expected: Step 8 restores the full-state project checkpoint and advances one optimizer step.
- actual: preflight reports a valid project checkpoint, then Lightning fails while reopening it.
- error: `PytorchStreamReader failed reading zip archive: invalid header or archive is corrupted`.
- timeline: Step 8 previously advanced the checkpoint from step 20 to 21; the next run targets step 22.
- reproduction: rerun Step 8 against `runs/h1_v2/A01/checkpoints/last.ckpt` stored on Google Drive.

## Current Focus

hypothesis: Directly overwriting a multi-GB `last.ckpt` on Google Drive and reopening it from Drive leaves no atomicity or stable-read guarantee; an interrupted/unstable write can destroy the only resume file.
test: Add atomic `.part` publication, stage Drive checkpoints to local disk, validate the staged copy, and regression-test corrupt-latest fallback.
expecting: An existing `last.ckpt` is never replaced until the new file is complete and valid; auto-resume skips an unreadable latest file and uses an older valid checkpoint or starts cleanly.
next_action: Pull the fix in Colab, rerun Step 7 once to replace the already-corrupt checkpoint, then run Step 8.

## Evidence

- timestamp: 2026-09-27
  finding: The supplied log selects `last.ckpt`, validates it once, then a second open by Lightning fails in `torch.load` with an invalid ZIP header.
  implication: The failure is checkpoint storage/read integrity, not model architecture, optimizer contract, or Cell 8 target-step logic.
- timestamp: 2026-09-27
  finding: `_save_final_checkpoint` writes directly to the final Drive path and `_resolve_resume` selects only the newest file without an integrity-aware fallback.
  implication: A failed write can corrupt the sole `last.ckpt`, and subsequent runs repeatedly select it.

## Eliminated

- hypothesis: Cell 8 is using the author checkpoint as a Lightning resume checkpoint.
  evidence: The log explicitly selects the project `last.ckpt` and reports the project contract, optimizer state, and global step before Lightning restore.
- hypothesis: The current failure is a model shape or dependency mismatch.
  evidence: The exception occurs inside PyTorch ZIP archive opening, before state restoration or tensor shape checks.

## Resolution

root_cause: The final full-state checkpoint was written directly over `last.ckpt` on Google Drive, and resume reopened that large remote file multiple times. A partial/unstable Drive write therefore destroyed the only resume target; the supplied file is already corrupt and cannot be repaired.
fix: Save to `last.ckpt.part`, validate the complete checkpoint, and atomically replace `last.ckpt`; stage resume candidates on local Colab disk before validation and Lightning restore; skip corrupt auto-resume candidates; make Step 8 explicitly resume `PROJECT_CKPT` so it cannot silently fall back to a new run.
verification: 15 focused checkpoint/notebook tests passed, all generated notebook cells compile, and the full suite passes with 93 passed and 2 skipped.
files_changed: [`train.py`, `tools/build_colab_training_notebook.py`, `Wild_Diff_ICMH_Kgalagadi_Train.ipynb`, `tests/test_checkpoint_contract.py`, `tests/test_phase1_completion_contract.py`]
