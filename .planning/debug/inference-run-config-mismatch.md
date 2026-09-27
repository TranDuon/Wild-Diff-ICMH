---
status: resolved
trigger: "Step 9 cannot load the H1 project checkpoint because 319 control-model tensors have checkpoint width 320 but inference model width 64."
created: 2026-09-27
updated: 2026-09-27
---

## Symptoms

- Expected: decode two test images from `runs/h1_v2/A01/checkpoints/last.ckpt`.
- Actual: inference exits before decoding with 319 tensor shape mismatches.
- Error: checkpoint control tensors start at width 320, while the inference model starts at width 64.

## Current Focus

hypothesis: Step 9 builds from the repository base config (`control_model_ratio=0.2`) instead of the resolved training config (`control_model_ratio=1.0`) saved in the run directory.
test: Resolve `config_model.yaml` adjacent to a project checkpoint before model construction and assert precedence in regression tests.
expecting: The inference model uses the exact training architecture and the checkpoint has no shared-key shape mismatch.
next_action: Pull the fix in Colab and rerun Step 9 only.

## Evidence

- timestamp: 2026-09-27T00:00:00+07:00
  finding: The failing merged inference config prints `control_model_ratio: 0.2`.
  implication: It instantiates 64-channel control layers.
- timestamp: 2026-09-27T00:01:00+07:00
  finding: The H1 training config and author checkpoint contract use `control_model_ratio: 1.0`.
  implication: The checkpoint correctly contains 320-channel control layers; the inference architecture is wrong.

## Eliminated

- hypothesis: The saved checkpoint is corrupted.
  evidence: Step 8 restored all model/optimizer states and advanced global step 20 to 21.

## Resolution

root_cause: Step 9 passed the generic base model config rather than the run-resolved model config saved with the checkpoint.
fix: Inference automatically prefers `<run>/config_model.yaml`; the notebook also passes that path explicitly and streams decode/evaluation logs to Drive.
verification: Config-resolution regression tests and complete test suite pass.
files_changed:
  - utils/checkpoint_contract.py
  - inference_partition.py
  - tools/build_colab_training_notebook.py
  - Wild_Diff_ICMH_Kgalagadi_Train.ipynb
  - COLAB_TRAINING.md
  - tests/test_checkpoint_contract.py
