---
status: awaiting_human_verification
trigger: "Colab notebook Step 7 smoke-test command invokes train.py and immediately exits status 1; the notebook shows only the parent CalledProcessError."
created: 2026-09-25
updated: 2026-09-25T19:30:00+07:00
---

## Symptoms

- expected: Step 7 loads the author checkpoint and completes a 20-optimizer-step Lightning smoke test, with visible progress and checkpoints/logs on Drive.
- actual: The training subprocess exits almost immediately and the cell raises `CalledProcessError`.
- error: `train.py ... returned non-zero exit status 1`; no useful child traceback is visible in the notebook output.
- timeline: First Step 7 run after Steps 1–6 completed successfully, including 971/971 RAM++ tags.
- reproduction: Run notebook Step 7 with the generated KGA:A01 tags, local dataset/checkpoints, L4 GPU, and dotted Lightning CLI overrides for 20 steps.

## Current Focus

hypothesis: "Confirmed: the selected CNscale1.0 author checkpoint has a full-width control module, while the Kgalagadi training config inherited control_model_ratio=0.2 from configs/model/diffeic.yaml."
test: "Merge the Kgalagadi override into the model config, require ratio 1.0, and regression-test both the checkpoint-name contract and shared-tensor shape validation."
expecting: "Step 7 reports 'Author checkpoint contract passed: control_model_ratio=1', loads the author checkpoint without zero-convolution size mismatches, then reaches Lightning trainer initialization."
next_action: "Commit/push the fix, pull it in Colab via Step 2, then rerun only Step 7 and confirm that checkpoint loading passes."

## Evidence

- timestamp: 2026-09-25
  checked: User's Step 7 screenshot.
  found: Command construction succeeds and points to existing author checkpoint/config, but the child exits before any Lightning progress or model-loading output is visible.
  implication: Failure is inside early `train.py` startup; the parent wrapper currently obscures the actionable error.

- timestamp: 2026-09-25
  checked: `train.py`, the Colab config, Step 7 generator, and Lightning/OmegaConf override construction.
  found: The dotted overrides and current Trainer keys are structurally valid; Step 7 used bare `subprocess.run(..., check=True)` and had no durable log or startup markers. Missing author/SD checkpoints were also discovered only after expensive setup.
  implication: No specific inner exception can be proven from the parent-only screenshot. The observability defect is deterministic and must be fixed before attributing a package/model failure.

- timestamp: 2026-09-25
  checked: Focused regression suite after instrumenting Step 7 and training startup.
  found: 26 tests pass; generated notebook streams merged stdout/stderr, writes a Drive log, enables unbuffered output, and `train.py` reports six startup phases plus early checkpoint-path validation.
  implication: A rerun will either proceed into training or provide sufficient evidence for a targeted second-cycle fix without guessing.

- timestamp: 2026-09-25
  checked: New streamed traceback, README Quick Start, checkpoint directory name, `configs/model/diffeic.yaml`, and the Colab training override.
  found: The checkpoint is `CNscale1.0_...` and contains control/zero-convolution tensors with 320/640/1280 channels, but the training model inherited `control_model_ratio: 0.2`, producing 64/128/256-channel control tensors.
  implication: The warm start is architecturally incompatible before training begins. The correct fix is to instantiate the published checkpoint's full-width ratio 1.0 architecture, not discard mismatched control weights.

- timestamp: 2026-09-25
  checked: Kgalagadi ratio override, checkpoint-name contract, tensor-shape guard, compile checks, and focused tests.
  found: The Kgalagadi config now overrides `control_model_ratio: 1.0`; `train.py` rejects a checkpoint/config ratio mismatch before model construction and reports any residual shared tensor mismatches before `load_state_dict`; 28 focused tests pass.
  implication: The known control-module mismatch is fixed and future architecture drift fails early with an actionable message.

## Eliminated

- Missing RAM++ tags: Step 6 completed 971/971 and wrote `KGA_A01.jsonl` on Drive.
- Missing GPU: the runtime shows an L4 with 22.5 GiB.

## Resolution

- root_cause: Step 7 first hid its child traceback; once exposed, it showed that the CNscale1.0 author checkpoint was being loaded into a control module built with the base config's 0.2 width ratio, so all control/zero-convolution channel shapes differed.
- fix: Preserve the streamed training diagnostics, override Kgalagadi training to `control_model_ratio: 1.0`, validate the CNscale value encoded in the checkpoint path before construction, and reject any residual shared tensor shape conflicts with a concise diagnostic rather than filtering weights.
- verification: `python -m pytest tests/test_colab_dependency_contract.py tests/data/test_split_check.py -q` -> 28 passed; `python -m compileall -q train.py utils/checkpoint_contract.py utils/common.py` -> passed; `git diff --check` -> passed.
- files_changed: [configs/train_kgalagadi_colab.yaml, train.py, utils/checkpoint_contract.py, tests/test_colab_dependency_contract.py]
