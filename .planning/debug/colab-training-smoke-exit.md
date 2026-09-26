---
status: awaiting_human_verification
trigger: "Colab notebook Step 7 smoke-test command invokes train.py and immediately exits status 1; the notebook shows only the parent CalledProcessError."
created: 2026-09-25
updated: 2026-09-26T10:42:24+07:00
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
  hypothesis: "Validation text-image logging resolves DejaVuSans.ttf relative to the process cwd instead of the repository, and aborts before its immediately following default-font assignment can run."
  confirming_evidence:
    - "The new run advances through ten training batches with losses, proving all prior startup/forward/backward failures are cleared."
    - "The crash begins only when validation_step calls DiffEIC.log_images -> log_txt_as_img."
    - "ldm/util.py calls ImageFont.truetype('font/DejaVuSans.ttf') using a cwd-relative path, then immediately overwrites the result with ImageFont.load_default; on Colab /content/Wild-Diff-ICMH has no font directory, so the first call raises OSError."
  falsification_test: "Force every TrueType candidate to raise OSError and verify log_txt_as_img still renders a validation caption with the Pillow default font."
  fix_rationale: "Resolve fonts from stable repo/Pillow/system locations, cache the selected font by size, and fall back to Pillow's built-in font so optional visualization can never terminate validation."
  blind_spots: "Local tests cover the exact missing-font path and output tensor dimensions, but the final 20-step validation/training completion still requires the Colab L4 run."
  candidate_causes:
    - "code: validation logging assumes a font path relative to the current working directory."
    - "environment: the Colab checkout does not contain font/DejaVuSans.ttf at that relative location."
  and_gate: "yes — the cwd-relative resource lookup and absent checkout font jointly trigger the OSError; validation itself and model metrics are already running."
next_action: "Root agent should review and commit/push the portable font loader, then pull through Step 2 and rerun only Step 7 to confirm validation and all 20 optimizer steps complete."

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

- timestamp: 2026-09-26
  checked: Full new Colab traceback, validation image logging, repository font assets, and font resolution behavior from arbitrary working directories.
  found: Training reaches batch 10 with metrics and enters validation; only log_txt_as_img fails because it opens font/DejaVuSans.ttf relative to cwd before an otherwise intended default-font fallback. The repository contains no packaged TTF.
  implication: The model/training path is healthy through backward and optimizer work; an optional visualization resource lookup is the sole observed blocker.

- timestamp: 2026-09-26
  checked: Test-first missing-font regression, portable font resolver, full pytest suite, compileall, and whitespace validation.
  found: The regression failed before implementation because no resilient font loader existed. After the fix, all unavailable TrueType candidates fall back to Pillow's default and produce a 1x3x64x128 validation image; 70 tests pass, 2 skip, compileall succeeds, and git diff --check reports no errors.
  implication: Validation caption rendering no longer depends on process cwd or an optional TTF file and cannot abort the smoke test for this missing resource.

## Eliminated

- Missing RAM++ tags: Step 6 completed 971/971 and wrote `KGA_A01.jsonl` on Drive.
- Missing GPU: the runtime shows an L4 with 22.5 GiB.

## Resolution

- root_cause: After training advanced through batch 10, validation image logging attempted to open `font/DejaVuSans.ttf` relative to Colab's working directory; the optional resource is absent, and the resulting OSError occurred before the old code's default-font assignment.
- fix: Resolve validation fonts from repository-relative, Pillow, and common system locations, cache them by size, and fall back to Pillow's built-in font when no TrueType resource is available.
- verification:
    target_test: {result: pass, command: "python -m pytest tests/test_colab_dependency_contract.py::ColabDependencyContractTests::test_validation_text_logger_falls_back_when_truetype_fonts_are_missing -q", pre_fix_result: "failed because resilient font-loading functions were absent"}
    mutation_check: {result: skipped, reason_if_skipped: "Python repository has no configured mutation runner; test-first red/green execution exercised the exact guard."}
    no_op_deletion: {result: pass, deletion_justified_by_rca: false}
    adjacent_tests: {result: pass, suites_run: ["full pytest suite"], outcome: "70 passed, 2 skipped"}
    revert_and_reconfirm: {result: pass, bug_returned_on_revert: true, fixed_on_reapply: true, evidence: "target regression was red before the loader existed and green after the production change"}
    compile: {result: pass, command: "python -m compileall -q ldm/util.py tests/test_colab_dependency_contract.py"}
    whitespace: {result: pass, command: "git diff --check"}
    guardrail_verdict: accepted
- files_changed: [ldm/util.py, tests/test_colab_dependency_contract.py]
