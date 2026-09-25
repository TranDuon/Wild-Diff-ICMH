---
status: resolved
trigger: "Fresh Colab Step 3 now passes CompressAI smoke imports but `from model.lfgcm import TagGCM` fails with `ModuleNotFoundError: No module named 'ram'`."
created: 2026-09-25
updated: 2026-09-25T00:39:00+07:00
---

## Symptoms

- expected: Step 3 installs the vendored Recognize Anything package and `from model.lfgcm import TagGCM` succeeds in the same notebook kernel.
- actual: CompressAI imports succeed, then `model/lfgcm.py` fails at `from ram import get_transform`.
- error: `ModuleNotFoundError: No module named 'ram'`.
- timeline: First fresh Colab acceptance run after the complete CompressAI dependency fix.
- reproduction: On a fresh Python 3.13 Colab runtime, run notebook Steps 1–3; the cell installs `src/recognize-anything` editable in a pip subprocess and immediately imports `TagGCM` in the already-running kernel.

## Current Focus

bug_class: bohrbug
reasoning_checkpoint:
  hypothesis: "Step 3 causes `ModuleNotFoundError: ram` because editable installation occurs in a child pip process and the live notebook interpreter does not activate the new `.pth` metadata before importing `TagGCM`."
  confirming_evidence:
    - "The real vendored package installed successfully editable while the persistent parent still reported `find_spec('ram') == None`."
    - "Reprocessing site-packages in that same parent immediately resolved `ram` to the vendored `ram/__init__.py`."
    - "The Step 3 source imports `TagGCM` immediately after the child pip process and performs no live-kernel path activation."
  falsification_test: "The hypothesis would be false if a fresh persistent parent resolved `ram` immediately after the child editable install without reprocessing site metadata or adding the source root."
  fix_rationale: "Replace the editable install with a regular local package install into the already-active site-packages directory, invalidate import caches, and smoke-import `ram` before `TagGCM`. This works immediately in the current kernel and in later subprocesses without a restart-only `.pth` dependency."
  blind_spots: "The exact cell cannot be executed on a real Colab GPU here; verification will combine the Python 3.13 lifecycle reproduction, generated-cell contract tests, and existing dependency-contract tests."
  candidate_causes:
    - "code: child-process editable install is followed by a parent-process import without path activation (confirmed)"
    - "environment: Colab/Python dependency incompatibility or missing RAM source package (eliminated because discovery fails before dependency import and the same source resolves after path refresh)"
  and_gate: "no — the code-level path activation gap alone reproduces the failure outside Colab; no second config, environment, or data condition is required"
next_action: "Resolved locally; rerun Step 3 on a fresh Colab runtime for the environment-level acceptance check."

## Evidence

- timestamp: 2026-09-25
  checked: User traceback after the prior dependency fix.
  found: `compressai.losses` and `compressai.zoo` import successfully; failure occurs only when `model.lfgcm` imports top-level `ram`.
  implication: CompressAI closure is fixed; the remaining defect is vendored package discovery.
- timestamp: 2026-09-25
  checked: `src/recognize-anything/setup.py`, `setup.cfg`, package tree, and notebook install command.
  found: Packaging declares `packages = find:` and contains `ram/__init__.py`, but Step 3 installs it with `pip --no-deps -e` in a subprocess and imports it immediately in the existing kernel.
  implication: Package contents are valid; editable-path activation timing is the leading cause.
- timestamp: 2026-09-25
  checked: Complete notebook generator Step 3 and dependency-contract regression test.
  found: Step 3 performs the editable install in `subprocess.run(...)` and then imports `TagGCM` in the unchanged parent interpreter; the regression test checks only that source strings exist and never executes the same-process install/import lifecycle.
  implication: The current test gate cannot detect editable metadata that is invisible to an already-running kernel.
- timestamp: 2026-09-25
  checked: Recognize Anything packaging metadata.
  found: Distribution name and top-level import are both `ram`; setuptools package discovery is valid, so an absent `ram` spec after a reported successful install points to activation/visibility rather than a name mismatch.
  implication: The reproduction should isolate interpreter path refresh from package contents and dependency imports.
- timestamp: 2026-09-25
  checked: First isolated reproduction setup under `C:\tmp`.
  found: `python -m venv` failed with Windows `Access is denied` before pip or import discovery ran.
  implication: This attempt provides no evidence for or against the import hypothesis; retry under the writable workspace.
- timestamp: 2026-09-25
  checked: Inline `python -c` reproduction in the workspace virtualenv.
  found: PowerShell native argument handling stripped quotes from the multiline payload and Python stopped at a syntax error before the test ran.
  implication: This is a test-harness failure, not product evidence; use a temporary script file without changing the experiment.
- timestamp: 2026-09-25
  checked: File-based reproduction in a clean Python 3.13 virtualenv.
  found: `BEFORE` was `None`, but pip failed before installation because this Python 3.13 venv did not contain `setuptools.build_meta` and the command intentionally disabled build isolation.
  implication: Package absence was established, but import visibility was not tested; retry with inherited system setuptools while preserving the fresh interpreter.
- timestamp: 2026-09-25
  checked: Persistent parent-process reproduction with the real vendored package in a fresh Python 3.13 virtualenv inheriting setuptools.
  found: `BEFORE None`; pip successfully built and installed `ram-0.0.1` editable; `AFTER_EDITABLE None`; after `site.addsitedir(...)`, `find_spec('ram')` resolved to `src/recognize-anything/ram/__init__.py`.
  implication: The leading hypothesis is confirmed: editable installation succeeds, but its newly written path metadata is not activated in the already-running notebook kernel.
- timestamp: 2026-09-25
  checked: Spectrum-based fault localization eligibility.
  found: No behavior-level failing test with per-test coverage existed for the notebook install lifecycle; the current suite only inspected generated source strings.
  implication: SBFL is skipped; a focused regression contract must be added before the fix.
- timestamp: 2026-09-25
  checked: First invocation of the new focused regression test.
  found: Dotted unittest import failed because `tests/` is not a Python package; no test body executed.
  implication: This is a harness invocation error, not a result; execute the test file directly.
- timestamp: 2026-09-25
  checked: Regular local package install in an already-running Python process.
  found: Installing the vendored source with `--target` after the target path was already on `sys.path`, followed by `importlib.invalidate_caches()`, made `find_spec('ram')` succeed without restarting the interpreter.
  implication: A non-editable local install removes the `.pth` activation dependency that caused the Colab failure.
- timestamp: 2026-09-25
  checked: Generated notebook, install-order contract, and same-process regression suite.
  found: All 4 focused dependency tests passed; all generated notebook code cells parsed; `git diff --check` passed.
  implication: The generator, checked-in notebook, package install visibility, and immediate import ordering are covered.

## Eliminated

- Missing `ram/` source package: the vendored tree contains `src/recognize-anything/ram/__init__.py` and its model/data subpackages.
- CompressAI dependency failure: all explicit CompressAI smoke imports shown immediately before `TagGCM` succeeded.
- Colab-only dependency incompatibility: `ram` is absent at module-discovery time before its dependencies execute, and the real package becomes discoverable in the same interpreter as soon as its source root is activated.
- Package-name or source-path mismatch: after site metadata reprocessing, discovery resolves exactly to the repository's `src/recognize-anything/ram/__init__.py`.

## Resolution

- root_cause: Step 3 runs editable pip installation in a child process and imports `TagGCM` immediately in the unchanged notebook process; that process does not reprocess the newly created editable `.pth` metadata, so the vendored `ram` source root is absent from `sys.path` despite a successful install.
- fix: Replaced `pip install -e` with a forced regular local install of `src/recognize-anything`, invalidated import caches in the live kernel, explicitly smoke-imported `ram`, regenerated the notebook, documented the restart-free contract, and added a same-process install/discovery regression test.
- verification: 4 focused dependency-contract tests passed, including a real pip install into a path already present in the running interpreter; all notebook code cells parsed and `git diff --check` passed. Fresh Colab Step 3 remains the final environment-level acceptance check.
- oracle_type: specified — Step 3 explicitly promises that `TagGCM` is importable in the same notebook kernel immediately after dependency installation.
- files_changed: [`tools/build_colab_training_notebook.py`, `Wild_Diff_ICMH_Kgalagadi_Train.ipynb`, `tests/test_colab_dependency_contract.py`, `COLAB_TRAINING.md`]
