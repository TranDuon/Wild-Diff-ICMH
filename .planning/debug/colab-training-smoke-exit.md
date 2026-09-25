---
status: awaiting_human_verification
trigger: "Colab notebook Step 7 smoke-test command invokes train.py and immediately exits status 1; the notebook shows only the parent CalledProcessError."
created: 2026-09-25
updated: 2026-09-25T22:55:00+07:00
---

## Symptoms

- expected: Step 7 loads the author checkpoint and completes a 20-optimizer-step Lightning smoke test, with visible progress and checkpoints/logs on Drive.
- actual: The training subprocess exits almost immediately and the cell raises `CalledProcessError`.
- error: `train.py ... returned non-zero exit status 1`; no useful child traceback is visible in the notebook output.
- timeline: First Step 7 run after Steps 1–6 completed successfully, including 971/971 RAM++ tags.
- reproduction: Run notebook Step 7 with the generated KGA:A01 tags, local dataset/checkpoints, L4 GPU, and dotted Lightning CLI overrides for 20 steps.

## Current Focus

bug_class: bohrbug
reasoning_checkpoint:
  hypothesis: "FrozenOpenCLIPEmbedder unconditionally converts NLD token embeddings to legacy LND layout, while the OpenCLIP version selected by the supported >=2.22,<4 range declares transformer.batch_first=True; a batch of one is therefore interpreted as sequence length 1 and cannot use the canonical 77x77 causal mask."
  confirming_evidence:
    - "The Colab traceback reports attn_mask (77,77) but expected (1,1), exactly matching a batch-first attention block receiving LND (77,1,D) for batch size 1."
    - "The focused regression reproduces the defect: the current method sends (77,1,4) to a transformer declaring batch_first=True and fails, while the input embedding is (1,77,4)."
    - "The dataset and conditioning chain preserve text as list[str] and tokenize to N x 77, ruling out a one-token RAM tag payload."
  falsification_test: "If a batch-first transformer stub receives NLD and a legacy transformer stub receives LND after the fix, both must return the same public NLD shape; failure of either case would falsify the compatibility fix."
  fix_rationale: "Conditionally permuting only for transformers that do not declare batch_first preserves the layout contract of both OpenCLIP API generations and leaves the standard 77x77 causal mask untouched."
  blind_spots: "The exact Colab OpenCLIP wheel is unavailable locally, so verification uses its public transformer.batch_first contract and a faithful layout stub; final GPU execution remains a human verification step."
  candidate_causes:
    - "code: vendored Stable-Diffusion-era encoder hard-codes LND around manually invoked OpenCLIP blocks."
    - "environment: the broad open_clip_torch>=2.22,<4 constraint resolves to a newer batch-first transformer implementation on Colab."
    - "data: malformed txt batching was considered, but the dataset returns strings and default collation produces list[str], which tokenizes to N x 77."
  and_gate: "yes — the failure requires both the hard-coded legacy permutation and a batch-first OpenCLIP implementation; either the old sequence-first dependency or a layout-aware encoder would not fail."
next_action: "Commit/push the OpenCLIP layout compatibility fix, pull it in Colab via Step 2, then rerun only Step 7 and confirm the smoke test advances beyond the first text-conditioning pass."

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

- timestamp: 2026-09-25
  checked: New Colab traceback after the architecture fix and every LightningModule/DataModule/Callback hook in the repository against Lightning 2.6 hook contracts.
  found: Model construction, author-checkpoint loading, and Trainer initialization now pass. Training reaches Epoch 0 batch 0, where `LatentDiffusion.on_train_batch_start(self, batch, batch_idx, dataloader_idx)` raises because Lightning 2.6 correctly supplies only `(batch, batch_idx)`. DataModule and Callback signatures on the active path already match Lightning 2; two train-batch-end hooks used catch-all arguments instead of the explicit current contract.
  implication: The remaining observed failure is an API migration defect, not a dataset, GPU, checkpoint, or model-shape problem.

- timestamp: 2026-09-25
  checked: Lightning hook compatibility patch, AST signature/behavior regressions, focused dependency/data tests, compileall, and whitespace validation.
  found: `on_train_batch_start` now accepts the Lightning 2 call; train-batch-end hooks explicitly accept `(outputs, batch, batch_idx)`; 30 focused tests pass; compileall and `git diff --check` pass.
  implication: The confirmed hook mismatch is fixed without weakening argument validation or changing model behavior.

- timestamp: 2026-09-25
  checked: New Step 7 traceback, CameraTrapDataset text output, DiffEIC external-text path, LatentDiffusion conditioning path, and FrozenOpenCLIPEmbedder tensor transforms.
  found: The dataset returns a string and the DataLoader/conditioning path preserves a list of strings; tokenization produces N x 77 tokens. FrozenOpenCLIPEmbedder then unconditionally permutes NLD embeddings to LND before manually calling OpenCLIP residual blocks. The failing OpenCLIP block uses batch-first MultiheadAttention, so a batch of one is interpreted as target length 1 and rejects the 77x77 causal mask with exactly the observed `(77,77) but should be (1,1)` error.
  implication: This is an OpenCLIP tensor-layout API compatibility defect in the vendored encoder, not malformed RAM tags or a mask-generation defect.

- timestamp: 2026-09-25
  checked: Agent-authored focused tensor-layout regression before changing production code.
  found: The batch-first case deterministically fails because the current encoder sends `(77, 1, 4)` where the block contract requires `(1, 77, 4)`; the failure is at the unconditional NLD-to-LND permutation.
  implication: The hypothesis is reproduced independently of the large checkpoint/GPU path, and the fix site is localized to `FrozenOpenCLIPEmbedder.encode_with_transformer`.

- timestamp: 2026-09-25
  checked: Layout-aware encoder patch, focused regression, adjacent Colab/data tests, compileall, diff validation, and revert-and-reconfirm.
  found: The encoder now preserves NLD for batch-first OpenCLIP and only converts to LND for legacy transformers; the public output stays NLD and the 77x77 mask is unchanged. The focused test passes, 31 adjacent tests pass, compileall and `git diff --check` pass. Temporarily reverting only the production hunk makes the focused test fail with `(77,1,4)` versus `(1,77,4)`; reapplying makes it pass.
  implication: The minimal layout branch fixes the reproduced cause and retains compatibility with both supported OpenCLIP generations.

## Eliminated

- Missing RAM++ tags: Step 6 completed 971/971 and wrote `KGA_A01.jsonl` on Drive.
- Missing GPU: the runtime shows an L4 with 22.5 GiB.

## Resolution

- root_cause: The latest first-batch failure required two conditions: the vendored FrozenOpenCLIPEmbedder hard-coded the legacy LND residual-block layout, while Colab resolved the supported OpenCLIP range to a batch-first transformer. With batch size one, attention interpreted the tensor as sequence length one and rejected the unchanged 77x77 causal mask.
- fix: Preserve the streamed training diagnostics and prior checkpoint/Lightning migrations, then make FrozenOpenCLIPEmbedder inspect the transformer's batch-first contract and permute only for legacy sequence-first implementations. Keep the standard 77x77 attention mask intact.
- verification:
    target_test: {result: pass, command: "python -m pytest tests/test_colab_dependency_contract.py::ColabDependencyContractTests::test_openclip_text_encoder_honors_transformer_tensor_layout -q"}
    mutation_check: {result: skipped, reason_if_skipped: "Stryker is not applicable to this Python repository; revert-and-reconfirm exercises the exact layout branch instead."}
    no_op_deletion: {result: pass, deletion_justified_by_rca: false}
    adjacent_tests: {result: pass, suites_run: ["tests/test_colab_dependency_contract.py", "tests/data/test_split_check.py"], outcome: "31 passed"}
    revert_and_reconfirm: {result: pass, bug_returned_on_revert: true, fixed_on_reapply: true}
    compile: {result: pass, command: "python -m compileall -q ldm/modules/encoders/modules.py tests/test_colab_dependency_contract.py"}
    whitespace: {result: pass, command: "git diff --check"}
    guardrail_verdict: accepted
- files_changed: [configs/train_kgalagadi_colab.yaml, train.py, utils/checkpoint_contract.py, ldm/models/diffusion/ddpm.py, ldm/models/autoencoder.py, ldm/modules/encoders/modules.py, tests/test_colab_dependency_contract.py]
