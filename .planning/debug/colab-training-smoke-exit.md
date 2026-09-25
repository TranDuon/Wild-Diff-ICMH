---
status: investigating
trigger: "Colab notebook Step 7 smoke-test command invokes train.py and immediately exits status 1; the notebook shows only the parent CalledProcessError."
created: 2026-09-25
updated: 2026-09-25T19:05:00+07:00
---

## Symptoms

- expected: Step 7 loads the author checkpoint and completes a 20-optimizer-step Lightning smoke test, with visible progress and checkpoints/logs on Drive.
- actual: The training subprocess exits almost immediately and the cell raises `CalledProcessError`.
- error: `train.py ... returned non-zero exit status 1`; no useful child traceback is visible in the notebook output.
- timeline: First Step 7 run after Steps 1–6 completed successfully, including 971/971 RAM++ tags.
- reproduction: Run notebook Step 7 with the generated KGA:A01 tags, local dataset/checkpoints, L4 GPU, and dotted Lightning CLI overrides for 20 steps.

## Current Focus

hypothesis: "The specific child exception remains unknowable from the supplied screenshot because Step 7 discarded the only actionable subprocess context; the failure occurs somewhere between process import and trainer startup."
test: "Rerun the instrumented Step 7 after pulling the fix; inspect the first missing [Train N/6] marker and the persisted merged log."
expecting: "The notebook streams the complete child traceback, persists it on Drive, and identifies the exact startup phase instead of surfacing only CalledProcessError."
next_action: "Pull the instrumented notebook, rerun Step 7, and use /content/drive/MyDrive/wild_diff_icmh/logs/train_smoke_KGA_A01.log as evidence if a deeper failure remains."

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

## Eliminated

- Missing RAM++ tags: Step 6 completed 971/971 and wrote `KGA_A01.jsonl` on Drive.
- Missing GPU: the runtime shows an L4 with 22.5 GiB.

## Resolution

- root_cause: The immediate actionable root cause is an observability defect: Step 7 exposed only the parent's `CalledProcessError`, so the child exception needed to diagnose the smoke-test exit was not preserved in the supplied evidence.
- fix: Stream merged child stdout/stderr live, persist it to a Drive log, force unbuffered child output, add six startup phase markers, and validate both checkpoints before model construction.
- verification: `python -m pytest tests/test_colab_dependency_contract.py tests/data/test_split_check.py -q` -> 26 passed.
- files_changed: [train.py, tools/build_colab_training_notebook.py, Wild_Diff_ICMH_Kgalagadi_Train.ipynb, tests/test_colab_dependency_contract.py]
