---
status: awaiting_human_verification
trigger: "Colab notebook Step 6 starts RAM++ tag generation for KGA:A01, reports 0/971 existing rows, then tools/precompute_ram_tags.py exits with status 1 and the notebook shows only CalledProcessError."
created: 2026-09-25
updated: 2026-09-25T18:18:00+07:00
---

## Symptoms

- expected: Step 6 loads the RAM++ checkpoint, processes 971 KGA:A01 images with visible progress, and writes `KGA_A01.jsonl` to Drive.
- actual: The subprocess exits immediately before processing any image; the wrapper raises `CalledProcessError`.
- error: `tools/precompute_ram_tags.py ... returned non-zero exit status 1`; the underlying child traceback is not preserved prominently by the notebook cell.
- timeline: First Step 6 run after Steps 1–5 and the dependency/RAM import fixes succeeded.
- reproduction: Run notebook Step 6 with local images at `/content/data/wild_diff_icmh/images`, RAM++ checkpoint at `/content/data/wild_diff_icmh/checkpoints/ram/ram_plus_swin_large_14m.pth`, site `KGA:A01`, batch size 8, and 4 workers.

## Current Focus

hypothesis: "The streamed traceback proves the current failure is import-path bootstrap: direct execution of tools/precompute_ram_tags.py makes tools/ sys.path[0], so the repository-root model package is not discoverable even with cwd=REPO."
test: "Insert the repository root derived from __file__ before any project import, defer TagGCM until after parsing and preflight, and launch the script with --help from an unrelated working directory in a real subprocess."
expecting: "The subprocess bootstrap test exits 0 outside the repository; on Colab, Step 6 reaches the named preflight/checkpoint/model stages instead of failing with No module named 'model'."
next_action: "Root agent commits and pushes the two-file fix; user pulls it and reruns Step 6. Inspect the streamed traceback/log only if a new failure appears."

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

- timestamp: 2026-09-25
  checked: User's second Step 6 acceptance run with streamed child output.
  found: `tools/precompute_ram_tags.py` exits at line 18 on `from model.lfgcm import TagGCM` with `ModuleNotFoundError: No module named 'model'` before preflight or model allocation.
  implication: The persistent logging worked and isolated a deterministic Python script bootstrap issue, not CUDA memory, checkpoint content, or image data.

- timestamp: 2026-09-25
  checked: Direct-script import semantics and regression launched from a temporary directory.
  found: The script now derives `REPO_ROOT` from `__file__`, prepends it to `sys.path`, defers the heavyweight project import, and `python <absolute-script-path> --help` exits 0 from outside the repository. The focused suite reports `24 passed`; Python compilation and `git diff --check` pass.
  implication: The exact invocation mode used by Colab is covered and the `model` package no longer depends on the caller's working-directory import behavior.

## Eliminated

- Notebook Step 3 environment initialization: it completed with `THÀNH CÔNG: môi trường và TagGCM đã sẵn sàng.`
- Empty site selection: Step 6 found 971 rows for `KGA:A01`.
- Missing RAM package config/tag resources: the regular wheel install contains both directories.
- Manifest schema mismatch: selected rows contain the required `image_id`, `site_id`, and `relative_path` keys.

## Resolution

- root_cause: After the earlier diagnostic hardening exposed the child traceback, direct execution of `tools/precompute_ram_tags.py` was shown to put `tools/` rather than the repository root on `sys.path`, making the top-level `model` package undiscoverable before RAM++ initialization.
- fix: Bootstrap the repository root from the script's own `__file__`, prepend it to `sys.path`, and defer `TagGCM` import until after argparse and input preflight; retain the earlier tokenizer, VRAM-safe batch, progress, and persistent-log protections.
- verification: A real subprocess launched from an unrelated temporary directory reaches `--help` successfully; the focused suite passes (24 tests), modified Python files compile, and `git diff --check` passes. Hardware acceptance remains: Step 6 must reach preflight/model loading and advance beyond 0/971 on Colab.
- files_changed:
  - src/recognize-anything/ram/models/ram_plus.py
  - tools/precompute_ram_tags.py
  - tools/build_colab_training_notebook.py
  - Wild_Diff_ICMH_Kgalagadi_Train.ipynb
  - tests/test_colab_dependency_contract.py
