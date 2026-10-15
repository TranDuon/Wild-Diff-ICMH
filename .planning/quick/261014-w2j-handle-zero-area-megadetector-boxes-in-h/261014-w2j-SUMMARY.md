---
phase: quick
plan: 261014-w2j
quick: 261014-w2j
status: complete
completed: 2026-10-15
implementation_commits: [dee980c]
requirements-touched: [INFRA-02, H2-04, H2-06]
experimental_status: live_h2_preflight_rerun_pending
---

# Quick 261014-w2j: Handle zero-area detector boxes

H2 ROI preflight now discards only finite boxes whose width or height is non-positive. Valid boxes remain in source order, while an all-degenerate but present record becomes an explicit empty frame. Malformed, non-numeric, non-finite, duplicate, and missing evidence still fails closed.

Verification: `tests/test_h2_roi_preflight.py` passed **21 tests** in the established CPU environment. No Drive access, Colab job, model download, detector inference, training, or evaluation was performed.

Implementation commit: `dee980c`.
