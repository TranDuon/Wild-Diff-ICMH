---
phase: quick-260916-e5o
plan: 01
subsystem: inference
tags: [kaggle, pytorch, diff-icmh, pyiqa, huggingface, xformers]
requires: []
provides:
  - Pinned one-image Kaggle GPU inference workflow from runtime preflight through ZIP export
  - Notebook-local Lightning 2.x and pyiqa compatibility seams for the legacy repository imports
  - Structural validation evidence for checkpoint, config, CLI, metric, and output wiring
affects: [phase-1, inference-smoke-test, evaluation]
actuals:
  tokens: 5551
  tasks: 1
  commits: 0
plan_head_before: aebaff6f61d0253e09e3f482d5887aa50e0a539a
tech-stack:
  added: [lightning-2.x-runtime, compressai-1.2.8-runtime, pyiqa-0.1.15.post2-runtime]
  patterns: [pinned detached checkout, fail-fast Kaggle preflight, notebook-local compatibility modules, argument-list subprocess]
key-files:
  created: []
  modified: [Diff_ICMH_Kaggle_Quick_Run.ipynb]
key-decisions:
  - "Preserve Kaggle's preinstalled Torch/Torchvision/CUDA stack and select xFormers from its detected CUDA wheel index."
  - "Keep all legacy Lightning and LPIPS compatibility code outside the checked-out repository under /kaggle/working/difficmh_compat."
  - "Treat local JSON/AST/static verification as structural evidence only; live Kaggle Run All remains required."
patterns-established:
  - "Expensive model downloads follow GPU, disk, HTTPS, dependency, and repository-import gates."
  - "External commands and inference use argument lists with check=True, never interpolated shell strings."
requirements-completed: [PRE-01, EVAL-08, REPT-04]
coverage:
  - id: D1
    description: "Runnable notebook wiring from preflight through selective checkpoint download, one-image inference, metrics, and ZIP export"
    verification:
      - kind: integration
        ref: "plan JSON/AST/static-wiring command executed with WSL Python 3.12.3"
        status: pass
      - kind: other
        ref: "focused ordering, TLS, Torch-preservation, compatibility, subprocess, and clean-output checks"
        status: pass
    human_judgment: false
  - id: D2
    description: "Live Kaggle GPU execution reaches reconstruction, bitstreams, metrics, and ZIP"
    verification:
      - kind: manual_procedural
        ref: "Kaggle Run All with GPU and Internet enabled"
        status: unknown
    human_judgment: true
    rationale: "No Kaggle runtime is available in the local execution environment."
duration: 12min
completed: 2026-09-16
status: complete
---

# Quick Plan 260916-e5o: Kaggle Diff-ICMH Inference Notebook Summary

**Pinned one-image Diff-ICMH Kaggle workflow with fail-fast runtime gates, selective checkpoint acquisition, compatibility shims, real CLI wiring, pyiqa metrics, and ZIP export**

## Performance

- **Duration:** 12 min
- **Started:** 2026-09-16T03:25:17Z
- **Completed:** 2026-09-16T03:36:26Z
- **Tasks:** 1
- **Files modified:** 1 implementation artifact plus this execution summary

## Accomplishments

- Reordered the notebook into a true top-to-bottom Kaggle path: GPU/disk/HTTPS preflight, pinned repository checkout, runtime-safe dependency setup, import smoke test, checkpoint acquisition, explicit one-image staging, inference, validation, visualization, metric recomputation, and export.
- Preserved Kaggle's preinstalled Torch stack while installing unified Lightning, CompressAI, pyiqa, vendored RAM++, and an xFormers wheel selected from the detected CUDA ABI.
- Added notebook-local compatibility modules for legacy `pytorch_lightning` type/rank-zero imports and the `lpips.LPIPS` call shape without rewriting repository sources.
- Wired the default smoke test to `BPP_WEIGHT = 2`, `kodim01.png`, 10 steps, seed 231, CUDA, and the three README OmegaConf overrides; documented 50 steps as the README-quality mode.

## Task Commit

Commit intentionally omitted. The checkout is on protected branch `main`, and `git status --short` reports the repository contents—including `Diff_ICMH_Kaggle_Quick_Run.ipynb`—as untracked. Staging the notebook in that state would violate protected-branch safety and create a misleading partial first snapshot. No unrelated files were staged or modified.

## Files Created/Modified

- `Diff_ICMH_Kaggle_Quick_Run.ipynb` - Nine clean Python cells and ordered markdown instructions for the Kaggle smoke workflow.
- `.planning/quick/260916-e5o-create-a-runnable-kaggle-inference-noteb/260916-e5o-SUMMARY.md` - Execution evidence and live-runtime caveat; not committed because the orchestrator owns GSD artifacts.

## Validation Evidence

- Plan verifier: **PASS** — valid notebook JSON, all 9 Python cells parsed by Python 3.12 AST, all 31 required wiring tokens present, and every code cell has `execution_count: null` with empty outputs.
- Focused static checks: **PASS** — preflight precedes clone/download; Torch is not pinned or replaced; no TLS bypass; no shell subprocess; compatibility setup precedes checkpoint downloads; LPIPS shim is an `nn.Module`; inference passes `cwd`, `env`, and `check=True`; saved execution state is empty.
- Stub scan: **PASS** — no TODO/FIXME/placeholder or hardcoded empty rendering stub patterns found in the notebook.
- Live Kaggle execution: **NOT RUN** — a Kaggle GPU runtime is not available here. End-to-end status remains unverified until Run All reaches the ZIP cell with a reconstruction, non-empty bitstreams, and printed bpp/PSNR/SSIM/LPIPS values.

## Decisions Made

- Pinned repository checkout to `aebaff6f61d0253e09e3f482d5887aa50e0a539a` so the notebook cannot silently drift from the inspected interfaces.
- Used `hf_hub_download` for each exact model file over normal TLS and asserted meaningful minimum sizes before inference.
- Cleared only notebook-dedicated input/output directories, and rooted the downloadable archive at `OUTPUT_DIR` to exclude unrelated Kaggle files.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Made the pyiqa LPIPS adapter compatible with repository freezing logic**
- **Found during:** Task 1 focused interface verification
- **Issue:** `utils.metrics.frozen_module()` calls `.eval()` and `.parameters()` on `lpips.LPIPS`; a plain callable adapter would fail during inference initialization.
- **Fix:** Implemented the compatibility `LPIPS` class as `torch.nn.Module` with `forward()`, preserving `.to()`, `.eval()`, `.parameters()`, and legacy input-range behavior.
- **Files modified:** `Diff_ICMH_Kaggle_Quick_Run.ipynb`
- **Verification:** Python AST verifier and focused `lpips_is_module` check passed.
- **Committed in:** Not committed due to protected/all-untracked repository state.

**2. [Rule 3 - Blocking] Used WSL Python for the required AST verifier**
- **Found during:** Task 1 automated verification
- **Issue:** The plan's `python -c` command could not start because Windows had no installed Python on `PATH`; `py -3` was only an empty launcher stub.
- **Fix:** Executed the same verifier with `/usr/bin/python3` 3.12.3 through WSL.
- **Files modified:** None
- **Verification:** The verifier printed `validated notebook JSON, 9 Python cells, and 31 wiring tokens`.
- **Committed in:** Not applicable.

---

**Total deviations:** 2 auto-fixed (1 bug, 1 blocking host-tool substitution)
**Impact on plan:** Both fixes were necessary for correctness and verification; scope remained limited to the notebook and summary.

## Issues Encountered

- The repository is on `main` and presents an unusual all-untracked worktree, so no safe atomic implementation commit could be created.
- Context7 was unavailable locally; implementation used the plan's researched version constraints and direct inspection of the pinned repository imports.

## User Setup Required

On Kaggle, enable GPU and Internet, upload/open the notebook, and choose Run All. Success requires the final cell to expose a non-empty ZIP after producing one reconstruction, bitstreams, and bpp/PSNR/SSIM/LPIPS values.

## Next Phase Readiness

- The notebook is structurally ready for the live one-image Kaggle smoke run.
- Do not spend broader evaluation or training compute until that live smoke run completes and its runtime/CU observations are recorded.

## Self-Check

**PASSED** — the notebook and summary exist; the summary declares `status: complete` and preserves the live-Kaggle caveat; all 9 code cells pass Python 3.12 AST parsing and retain null execution counts with empty outputs.

---
*Phase: quick-260916-e5o*
*Completed: 2026-09-16*
