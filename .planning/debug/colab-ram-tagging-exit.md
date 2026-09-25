---
status: awaiting_human_verification
trigger: "Colab notebook Step 6 starts RAM++ tag generation for KGA:A01, reports 0/971 existing rows, then tools/precompute_ram_tags.py exits with status 1 and the notebook shows only CalledProcessError."
created: 2026-09-25
updated: 2026-09-25T18:02:00+07:00
---

## Symptoms

- expected: Step 6 loads the RAM++ checkpoint, processes 971 KGA:A01 images with visible progress, and writes `KGA_A01.jsonl` to Drive.
- actual: The subprocess exits immediately before processing any image; the wrapper raises `CalledProcessError`.
- error: `tools/precompute_ram_tags.py ... returned non-zero exit status 1`; the underlying child traceback is not preserved prominently by the notebook cell.
- timeline: First Step 6 run after Steps 1–5 and the dependency/RAM import fixes succeeded.
- reproduction: Run notebook Step 6 with local images at `/content/data/wild_diff_icmh/images`, RAM++ checkpoint at `/content/data/wild_diff_icmh/checkpoints/ram/ram_plus_swin_large_14m.pth`, site `KGA:A01`, batch size 8, and 4 workers.

## Current Focus

hypothesis: "Step 6 is the first code path that constructs RAM++; the vendored inference constructor unnecessarily downloads bert-base-uncased, and the notebook also hard-codes an L4-oriented batch of 8 on the user's 14.6 GiB T4. Either failure occurs at 0/971 while the old wrapper discards the useful child context."
test: "Remove tokenizer initialization from RAM++ inference, select batch size from VRAM, add preflight/stage diagnostics, stream combined child output to the notebook and a persistent Drive log, and run focused regressions."
expecting: "On the user's T4, Step 6 reports batch=1, advances through four named startup stages, then starts the progress bar; any remaining failure includes a full traceback and persistent log path."
next_action: "User pulls the fix and reruns Step 3 (to reinstall patched RAM) and Step 6 on Colab; inspect the persistent log only if it does not advance."

## Evidence

- timestamp: 2026-09-25
  checked: User's Colab Step 6 screenshot.
  found: Site row counting succeeds (`0/971`), then the exact Python command exits status 1 before any progress bar or processed-row output appears.
  implication: Notebook variables, site selection, and command construction work; failure is inside early startup of `precompute_ram_tags.py`.

- timestamp: 2026-09-25
  checked: `RAM_plus.__init__` versus Step 3 and Step 6 behavior.
  found: Step 3 only imports `TagGCM`; Step 6 constructs it. The inference constructor unconditionally called `BertTokenizer.from_pretrained('bert-base-uncased')`, although `generate_index` never uses the tokenizer.
  implication: Step 6 had an unnecessary network/cache dependency that Step 3 could not detect and that can fail before checkpoint loading or the first batch.

- timestamp: 2026-09-25
  checked: Notebook Step 6 resource settings against the screenshot GPU.
  found: The cell forced batch 8 and 4 workers, while the user's runtime is a Tesla T4 with 14.6 GiB rather than the earlier L4 with 22 GiB.
  implication: The first inference batch could fail at progress 0 with CUDA OOM even after model startup; T4 needs a conservative batch.

- timestamp: 2026-09-25
  checked: Installed RAM package resource contents.
  found: A clean non-editable install includes the Swin JSON configs and RAM tag-list text files from `MANIFEST.in`.
  implication: Missing package data is eliminated as the startup cause.

- timestamp: 2026-09-25
  checked: Focused regression and syntax suite.
  found: `23 passed`; modified Python files compile, every generated notebook code cell parses, and `git diff --check` reports no whitespace errors.
  implication: The fix is internally consistent and ready for the Colab acceptance run.

## Eliminated

- Notebook Step 3 environment initialization: it completed with `THÀNH CÔNG: môi trường và TagGCM đã sẵn sàng.`
- Empty site selection: Step 6 found 971 rows for `KGA:A01`.
- Missing RAM package config/tag resources: the regular wheel install contains both directories.
- Manifest schema mismatch: selected rows contain the required `image_id`, `site_id`, and `relative_path` keys.

## Resolution

- root_cause: Step 6 enters RAM++ inference initialization that Step 3 did not exercise; that path unnecessarily fetched a BERT tokenizer, while the notebook simultaneously used an unsafe fixed batch of 8 on a 14.6 GiB T4 and exposed only a generic parent `CalledProcessError`.
- fix: Skip tokenizer creation for RAM++ inference, choose batch 1 on sub-20-GiB GPUs (2 otherwise), validate checkpoint/image inputs before allocating the model, print four startup phases, and stream merged child output into both Colab and a persistent Drive log.
- verification: Automated regression suite passes (23 tests); Python and generated notebook syntax pass. Hardware acceptance remains: Step 6 must advance beyond 0/971 on the user's T4.
- files_changed:
  - src/recognize-anything/ram/models/ram_plus.py
  - tools/precompute_ram_tags.py
  - tools/build_colab_training_notebook.py
  - Wild_Diff_ICMH_Kgalagadi_Train.ipynb
  - tests/test_colab_dependency_contract.py
