---
status: awaiting_human_verification
trigger: "Colab notebook Step 7 smoke-test command invokes train.py and immediately exits status 1; the notebook shows only the parent CalledProcessError."
created: 2026-09-25
updated: 2026-09-25T23:30:00+07:00
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
  hypothesis: "The custom gradient-checkpoint wrapper passes every module parameter to CheckpointFunction, including the explicitly frozen SD backbone parameters. PyTorch 2.11 rejects requires_grad=False tensors in torch.autograd.grad even with allow_unused=True."
  confirming_evidence:
    - "The new Colab run completes forward and fails specifically at CheckpointFunction.backward line 149 inside torch.autograd.grad with `One of the differentiated Tensors does not require grad`."
    - "Every checkpoint input tensor is detached and re-enabled for grad at line 141, so the only non-grad differentiated inputs can be ctx.input_params."
    - "DiffEIC intentionally sets the frozen SD backbone parameters to requires_grad=False, while ResBlock/AttentionBlock/BasicTransformerBlock pass self.parameters() unfiltered into the wrapper."
  falsification_test: "A focused wrapper test must prove that a frozen parameter is absent from CheckpointFunction.apply while a trainable parameter and the checkpoint input remain present."
  fix_rationale: "Filter only explicit checkpoint parameter inputs by requires_grad; frozen weights remain visible to run_function through its module closure, while checkpoint inputs remain differentiable so gradient flow through the frozen backbone is preserved."
  blind_spots: "PyTorch is not installed in the local Windows test environment, so the focused regression exercises the wrapper contract without CUDA; the final full backward pass requires Colab verification."
  candidate_causes:
    - "code: the vendored Stable-Diffusion checkpoint helper assumes every module parameter requires gradients."
    - "architecture: DiffEIC now correctly freezes the SD backbone rather than merely excluding it from the optimizer."
    - "environment: PyTorch 2.11 validates differentiated inputs and rejects requires_grad=False tensors even with allow_unused=True."
  and_gate: "yes — the failure requires both explicit backbone freezing and the legacy unfiltered custom checkpoint parameter list; removing either condition avoids this exact exception."
next_action: "Review and commit/push the frozen-parameter checkpoint fix, pull it in Colab via Step 2, then rerun only Step 7 and confirm backward completes and the 20-step smoke test advances."

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

- timestamp: 2026-09-25
  checked: New Colab traceback after the OpenCLIP layout fix and the custom CheckpointFunction contract in `ldm/modules/diffusionmodules/util.py`.
  found: The first forward pass now succeeds and backward reaches `CheckpointFunction.backward`, where `torch.autograd.grad` receives all module parameters. DiffEIC explicitly freezes the SD backbone with `requires_grad=False`, and PyTorch 2.11 raises `One of the differentiated Tensors does not require grad` even when `allow_unused=True`.
  implication: Frozen parameters must remain captured by the module closure but must not be explicit differentiated inputs to the custom autograd function.

- timestamp: 2026-09-25
  checked: Focused checkpoint contract regression before and after filtering parameters, adjacent Colab/data tests, compileall, and whitespace validation.
  found: The regression failed before the production change because the frozen fake parameter reached `CheckpointFunction.apply`; after filtering on `requires_grad`, the focused test passes, 32 adjacent tests pass, compileall succeeds, and `git diff --check` reports no errors.
  implication: The fix preserves gradients through checkpoint input tensors and trainable parameters while excluding only invalid frozen autograd inputs.

## Eliminated

- Missing RAM++ tags: Step 6 completed 971/971 and wrote `KGA_A01.jsonl` on Drive.
- Missing GPU: the runtime shows an L4 with 22.5 GiB.

## Resolution

- root_cause: After the prior startup and OpenCLIP fixes allowed the first forward pass to complete, backward failed because the legacy custom checkpoint wrapper registered frozen SD parameters as differentiated inputs; PyTorch 2.11 rejects those requires_grad=False tensors regardless of allow_unused=True.
- fix: Keep frozen parameters available through each module's run-function closure, but pass only requires_grad=True parameters into CheckpointFunction and torch.autograd.grad; continue differentiating checkpoint input tensors so gradients flow into trainable upstream/control modules.
- verification:
    target_test: {result: pass, command: "python -m pytest tests/test_colab_dependency_contract.py::ColabDependencyContractTests::test_openclip_text_encoder_honors_transformer_tensor_layout -q"}
    mutation_check: {result: skipped, reason_if_skipped: "Stryker is not applicable to this Python repository; revert-and-reconfirm exercises the exact layout branch instead."}
    no_op_deletion: {result: pass, deletion_justified_by_rca: false}
    adjacent_tests: {result: pass, suites_run: ["tests/test_colab_dependency_contract.py", "tests/data/test_split_check.py"], outcome: "32 passed"}
    revert_and_reconfirm: {result: pass, bug_returned_on_revert: true, fixed_on_reapply: true}
    compile: {result: pass, command: "python -m compileall -q ldm/modules/encoders/modules.py tests/test_colab_dependency_contract.py"}
    whitespace: {result: pass, command: "git diff --check"}
    guardrail_verdict: accepted
- files_changed: [configs/train_kgalagadi_colab.yaml, train.py, utils/checkpoint_contract.py, ldm/models/diffusion/ddpm.py, ldm/models/autoencoder.py, ldm/modules/encoders/modules.py, tests/test_colab_dependency_contract.py]
