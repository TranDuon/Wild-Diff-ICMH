---
status: resolved
trigger: "Step 9 decode runs out of L4 VRAM in VAE attention while processing a full-resolution Kgalagadi image."
created: 2026-09-27
updated: 2026-09-27
---

## Symptoms

- Expected: Step 9 decodes and evaluates two 256×256 smoke-test samples.
- Actual: the first image fails inside the VAE encoder before DDIM sampling.
- Error: `torch.OutOfMemoryError: Tried to allocate 26.27 GiB` in `torch.bmm(q, k)`.

## Current Focus

hypothesis: Inference receives the original 2592×2000 image even though training and validation use 256×256 crops, causing quadratic spatial-attention memory growth.
test: Decode and evaluate the same deterministic 256×256 center crop and verify the crop contract in the generated notebook.
expecting: VAE attention operates on the training-resolution crop and no longer requests a multi-gigabyte spatial attention matrix.
next_action: Pull the fix in Colab and rerun Step 9 only.

## Evidence

- timestamp: 2026-09-27T16:05:00+07:00
  finding: The first two KGA:A01 test manifest rows are 2592×2000.
  implication: The previous inference path pads and sends roughly 5.2 million pixels through the VAE.
- timestamp: 2026-09-27T16:06:00+07:00
  finding: The traceback fails at the VAE's full spatial attention matrix with a 26.27 GiB allocation request.
  implication: Reducing DDIM steps or clearing CUDA cache cannot fix this tensor's required size.
- timestamp: 2026-09-27T16:07:00+07:00
  finding: Both Kgalagadi train and validation configs use `out_size: 256`; validation uses deterministic center cropping.
  implication: A shared 256×256 center crop is the correct smoke-test input contract.

## Eliminated

- hypothesis: The previous `used_timesteps` bug remains after pull.
  evidence: Colab reports commit `8da6445`, and execution now passes sampler construction and reaches the VAE encoder.
- hypothesis: CUDA allocator fragmentation is the primary cause.
  evidence: The single requested tensor is 26.27 GiB, larger than the L4's entire 22.03 GiB capacity.

## Resolution

root_cause: Step 9 decoded native-resolution camera-trap images instead of the project's 256×256 training/evaluation crop.
fix: Add a shared deterministic center-crop geometry contract, expose `--crop-size` in decode/evaluation, make Step 9 pass 256 to both commands, and default manifest-backed CLI runs to 256 so already-open Colab cells also become safe immediately after pull.
verification: Generated notebook passes the shared crop contract; full suite and stale-cell fallback tests pass; modified Python files compile successfully.
files_changed:
  - utils/image_geometry.py
  - inference_partition.py
  - tools/evaluate_kgalagadi.py
  - tools/build_colab_training_notebook.py
  - Wild_Diff_ICMH_Kgalagadi_Train.ipynb
  - tests/test_image_geometry.py
  - tests/test_phase1_completion_contract.py

