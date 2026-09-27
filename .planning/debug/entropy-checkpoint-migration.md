---
status: resolved
trigger: "Colab smoke training succeeds, but the author checkpoint reports missing preprocess_model.entropy_bottleneck.matrices/biases/factors keys and unexpected legacy _matrix/_bias/_factor keys under CompressAI 1.2.8; aux_loss is about 8240."
created: 2026-09-27
updated: 2026-09-27
---

## Symptoms

- Expected: The published author checkpoint restores every learned entropy-bottleneck parameter before baseline evaluation or fine-tuning.
- Actual: `load_state_dict(..., strict=False)` leaves the new CompressAI ParameterList keys missing and treats the checkpoint's legacy numbered keys as unexpected.
- Error messages: No crash; `_IncompatibleKeys` reports `matrices.0..4`, `biases.0..4`, `factors.0..3` missing and `_matrix0..4`, `_bias0..4`, `_factor0..3` unexpected. Training then logs `aux_loss` around 8240.
- Timeline: First complete Colab 20-step smoke run on 2026-09-27 after the earlier runtime blockers were fixed.
- Reproduction: Run notebook Step 7 from the author checkpoint with CompressAI 1.2.8 and inspect the warm-start message.

## Current Focus

hypothesis: The author checkpoint uses CompressAI's legacy entropy-bottleneck key layout and must be migrated deterministically before shape validation and loading.
test: Add a focused regression covering every legacy matrix/bias/factor key and rejecting collisions or incomplete migration.
expecting: The migrated state matches the current model keys exactly for the entropy bottleneck, while unrelated checkpoint keys remain unchanged.
next_action: Pull the fix in Colab and rerun the 20-step smoke test; confirm the log reports 14 migrated entropy-bottleneck keys and no entropy-key incompatibilities.

## Evidence

- timestamp: 2026-09-27T00:00:00+07:00
  finding: The author checkpoint keys follow the legacy `_matrixN`, `_biasN`, and `_factorN` layout, while the active model exposes `matrices.N`, `biases.N`, and `factors.N` ParameterList keys.
  implication: `strict=False` cannot translate names; it silently leaves all learned entropy transforms at their newly initialized values.
- timestamp: 2026-09-27T00:01:00+07:00
  finding: The warm-start path previously called shape validation and `load_state_dict` directly without any key migration or completeness check.
  implication: Migration must be inserted before both operations, and ambiguity must fail closed.
- timestamp: 2026-09-27T00:02:00+07:00
  finding: Focused tests migrate all 14 parameter-list tensors, preserve unrelated state/metadata, reject dual-name collisions, reject incomplete lists, and assert load ordering.
  implication: The compatibility behavior is deterministic and regression protected.
- timestamp: 2026-09-27T00:03:00+07:00
  finding: Full local suite passes with 81 passed and 2 skipped after extending the same contract to inference, project resume, and the end-to-end Colab path.
  implication: The migration introduces no observed regression in the existing data, notebook, model-contract, and workflow tests.
- timestamp: 2026-09-27T00:04:00+07:00
  finding: Existing `runs/h1` checkpoints can contain current-name entropy tensors that were randomly initialized before this fix, so key inspection alone cannot repair or certify them.
  implication: New checkpoints carry contract version 2, resume rejects unversioned checkpoints, and the notebook uses a fresh `runs/h1_v2` root.

## Eliminated

## Resolution

root_cause: The published Diff-ICMH checkpoint was created with an older CompressAI key layout, and `strict=False` treated the renamed learned entropy parameters as unrelated missing/unexpected keys instead of restoring them.
fix: Added a non-mutating legacy-to-ParameterList key migration, collision and completeness validation, and invoked both before shape validation and warm-start/inference loading. New project checkpoints carry a contract version; resume rejects unsafe pre-fix checkpoints. The Colab pilot now writes to `h1_v2`.
verification: Focused migration and resume-contract tests pass; complete project suite reports 81 passed and 2 skipped; compileall and diff checks pass.
files_changed:
  - utils/checkpoint_contract.py
  - train.py
  - inference_partition.py
  - model/diffeic.py
  - tests/test_checkpoint_contract.py
