---
status: awaiting_human_verification
trigger: "Colab notebook Step 7 smoke-test command invokes train.py and immediately exits status 1; the notebook shows only the parent CalledProcessError."
created: 2026-09-25
updated: 2026-09-26T00:02:00+07:00
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
  hypothesis: "Every SFT block unconditionally applies adaptive_avg_pool2d to its reference tensor, even though the codec architecture constructs ref and x with identical spatial dimensions. Its CUDA backward has no deterministic implementation, so Lightning deterministic=true rejects the first backward pass."
  confirming_evidence:
    - "The new Colab run completes forward and reaches backward, which fails specifically with `adaptive_avg_pool2d_backward_cuda does not have a deterministic implementation` while trainer deterministic=true is active."
    - "Repository search finds exactly one training-path adaptive_avg_pool2d call: SFT.forward in model/layers/res_blk.py."
    - "All SFT call sites construct ref and x at matching scales for the 256x256 training crop: encoder 1/8 and 1/16, hyper-encoder 1/16, 1/32 and 1/64, and decoder 1/16 and 1/8. The unconditional pool is therefore a no-op in the supported training path."
  falsification_test: "A focused SFT test must prove that equal spatial shapes bypass adaptive_avg_pool2d while mismatched shapes retain the authors' original resize behavior."
  fix_rationale: "Skip only the mathematically redundant adaptive pool when ref already has the target shape. This preserves exact values and gradients, keeps deterministic training enabled, and retains the original fallback for unexpected shape mismatches."
  blind_spots: "The local environment cannot execute the CUDA backward; final confirmation still requires the Colab L4 smoke test. A future unsupported crop producing mismatched SFT shapes would still use adaptive pooling and could require a separate deterministic resize policy."
  candidate_causes:
    - "code: SFT performs an unconditional adaptive pool even when source and target dimensions are identical."
    - "config/environment: Lightning deterministic=true enables PyTorch's hard error for CUDA operations without deterministic backward implementations."
  and_gate: "yes — the crash requires both the redundant adaptive pooling node in the graph and strict deterministic CUDA execution; preserving reproducibility means removing the redundant node rather than weakening the trainer setting."
next_action: "Review and commit/push the SFT equal-shape guard, pull it in Colab via Step 2, then rerun only Step 7 and confirm backward advances beyond batch 0 through the 20-step smoke test."

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

- timestamp: 2026-09-25
  checked: Full traceback after the frozen-parameter checkpoint fix, all adaptive pooling call sites, and the spatial scale construction of every SFT caller.
  found: The forward pass and checkpoint backward now proceed until PyTorch rejects `adaptive_avg_pool2d_backward_cuda` under strict deterministic execution. The sole project call is unconditional in SFT.forward, while every supported 256x256 codec path provides ref and x at the same spatial size, making that adaptive pool a value-preserving no-op.
  implication: The deterministic failure can be removed without weakening reproducibility or changing model math by bypassing only equal-shape pooling.

- timestamp: 2026-09-25
  checked: Test-first SFT regression, minimal equal-shape guard, full test suite, compileall, and whitespace validation.
  found: Before the fix the regression recorded an adaptive-pool call for matching 32x32 tensors and failed. After the fix, matching shapes bypass the call while a 64x64 reference still follows the original adaptive resize path; 69 tests pass, 2 skip, compileall succeeds, and git diff --check reports no whitespace errors.
  implication: The supported training path no longer records the nondeterministic CUDA backward node, and unexpected mismatched inputs retain the authors' prior behavior.

## Eliminated

- Missing RAM++ tags: Step 6 completed 971/971 and wrote `KGA_A01.jsonl` on Drive.
- Missing GPU: the runtime shows an L4 with 22.5 GiB.

## Resolution

- root_cause: After the prior fixes allowed backward to advance, SFT.forward unconditionally inserted adaptive_avg_pool2d even when ref and x already had identical spatial dimensions. Lightning deterministic=true enables strict PyTorch deterministic algorithms, and CUDA adaptive_avg_pool2d backward has no deterministic implementation, so the redundant node aborted batch 0.
- fix: Preserve strict deterministic training and exact model math by bypassing adaptive_avg_pool2d when ref already matches x spatially; retain the original adaptive resize behavior only for unexpected mismatched shapes.
- verification:
    target_test: {result: pass, command: "python -m pytest tests/test_colab_dependency_contract.py::ColabDependencyContractTests::test_sft_skips_redundant_adaptive_pool_for_matching_shapes -q", pre_fix_result: "failed because adaptive pooling was called for equal 32x32 shapes"}
    mutation_check: {result: skipped, reason_if_skipped: "Python repository has no configured mutation runner; test-first red/green execution exercised the exact guard."}
    no_op_deletion: {result: pass, deletion_justified_by_rca: false}
    adjacent_tests: {result: pass, suites_run: ["full pytest suite"], outcome: "69 passed, 2 skipped"}
    revert_and_reconfirm: {result: pass, bug_returned_on_revert: true, fixed_on_reapply: true, evidence: "target regression failed before the production change and passed afterward"}
    compile: {result: pass, command: "python -m compileall -q model/layers/res_blk.py tests/test_colab_dependency_contract.py"}
    whitespace: {result: pass, command: "git diff --check"}
    guardrail_verdict: accepted
- files_changed: [model/layers/res_blk.py, tests/test_colab_dependency_contract.py]
