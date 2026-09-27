---
status: resolved
trigger: "Colab Step 2A fails before execution with `IndentationError: unexpected indent` at `local_changes = subprocess.check_output(...)`."
created: 2026-09-27
updated: 2026-09-27
---

## Symptoms

- expected: Step 2A checks the worktree and fast-forwards the Colab checkout.
- actual: Python rejects the generated cell before any Git command runs.
- error: `IndentationError: unexpected indent` on the `local_changes` assignment.

## Root Cause

The notebook generator embedded unescaped `\n` sequences inside its outer triple-quoted
cell template. Python converted them to real line breaks before `textwrap.dedent()` ran,
introducing zero-indentation fragments and preventing the rest of the cell from being
dedented.

## Resolution

- Escaped the two intended newline characters as `\\n` in the generator template.
- Regenerated `Wild_Diff_ICMH_Kgalagadi_Train.ipynb`.
- Added a regression test that compiles every generated code cell.

## Verification

- Focused contract tests: `6 passed`.
- Full suite: `92 passed, 2 skipped`.
- The generated Step 2A source begins at column zero and compiles successfully.
