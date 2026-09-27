---
phase: "1"
slug: "khung-x-ng-end-to-end-t-i-thi-u-v-l-i-ch-n"
status: draft
nyquist_compliant: false
wave_0_complete: true
created: "2026-09-27"
---

# Phase 1 — Validation Strategy

## Test Infrastructure

| Property | Value |
|---|---|
| Framework | pytest |
| Config file | none |
| Quick run | `python -m pytest tests/data/test_audit_metadata.py tests/test_phase1_closeout.py -q` |
| Full suite | `python -m pytest tests/data tests/test_checkpoint_contract.py tests/test_phase1_completion_contract.py tests/test_results_registry.py -q` |
| Estimated runtime | < 20 seconds locally |

## Sampling Rate

- Sau mỗi task: chạy quick run.
- Sau mỗi wave: chạy full suite và `python -m compileall -q tools tests`.
- GPU L4 chỉ dùng cho checkpoint đã được người dùng chạy; closeout không khởi động job 2K.

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | Status |
|---|---|---|---|---|---|---|
| 1-01-01 | 01 | 1 | DATA-02, DATA-03, DATA-05 | unit | `python -m pytest tests/data/test_audit_metadata.py tests/data/test_split_check.py -q` | pending |
| 1-02-01 | 02 | 2 | PRE-03..06, INFRA-03..04, EVAL-01, EVAL-08 | unit/contract | `python -m pytest tests/test_phase1_closeout.py tests/test_phase1_completion_contract.py -q` | pending |

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|---|---|---|---|
| L4 throughput measurement | INFRA-03 | Colab GPU unavailable locally | Run Bước 10 after successful Bước 9 and inspect closeout JSON recommendation |

## Validation Sign-Off

- [x] Every implementation task has an automated check.
- [x] No watch mode.
- [ ] Colab Bước 10 creates the final Drive artifact.

