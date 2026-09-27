---
status: resolved
trigger: "Colab Step 7 reaches max_steps=20, then the notebook cannot find runs/h1_v2/A01/checkpoints/*.ckpt."
created: 2026-09-27
updated: 2026-09-27
---

## Symptoms

- Expected: the 20-step smoke run leaves a full-state `last.ckpt` for validation and resume testing.
- Actual: Lightning stops normally at `max_steps=20`, but the notebook raises `FileNotFoundError` because the checkpoint directory does not exist.
- Reproduction: run Step 7 with rolling checkpoint cadence `every_n_train_steps: 50` and `max_steps=20`.

## Current Focus

hypothesis: The rolling callback never fires in a run shorter than its 50-step cadence.
test: Save an explicit final full-state checkpoint after every normal `trainer.fit` return and regression-test the helper.
expecting: A 20-step run creates `checkpoints/last.ckpt` containing model, optimizer state and global step.
next_action: Pull the fixed entrypoint in Colab and rerun Step 7.

## Evidence

- timestamp: 2026-09-27T00:00:00+07:00
  finding: The log reaches `Trainer.fit stopped: max_steps=20 reached` before the notebook's checkpoint lookup fails.
  implication: Training itself succeeded; the failure is checkpoint scheduling, not the model or entropy migration.
- timestamp: 2026-09-27T00:01:00+07:00
  finding: The only rolling ModelCheckpoint callback is configured for every 50 optimizer steps.
  implication: It cannot save during a 20-step smoke run, and `save_last` does not independently force an end-of-fit save.

## Eliminated

- hypothesis: The entropy migration failed.
  evidence: The log reports all 14 legacy keys migrated and reaches 20 optimizer steps.

## Resolution

root_cause: The smoke run ends at step 20 while the first automatic checkpoint is scheduled at step 50.
fix: Save `checkpoints/last.ckpt` explicitly after a normal `trainer.fit` return, without changing the 50-step crash-recovery cadence used by long runs.
verification: Regression test proves a full-state save is requested and the file is present; full suite passes.
files_changed:
  - train.py
  - tests/test_checkpoint_contract.py
