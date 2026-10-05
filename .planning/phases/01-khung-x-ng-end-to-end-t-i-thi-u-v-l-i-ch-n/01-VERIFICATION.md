---
phase: 1
status: passed
score: 14/14
verified: 2026-10-04
next_action: "Phase 2"
next_command: "$gsd-plan-phase 2"
---

# Phase 1 Verification

## Automated evidence

- 97 relevant tests passed; 2 skipped; no failures.
- Generated notebook code cells compile.
- Split positive path and deliberate sequence-leak negative path pass.
- Checkpoint and registry contracts are covered by automated tests.
- Decision coverage is 14/14.

## Human verification required

1. In the existing successful Colab runtime, run permanent **Bước 2A** to pull the new
   commit, then run only **Bước 10**.
2. Confirm the cell prints `PHASE 1 CLOSEOUT THÀNH CÔNG`.
3. Confirm these Drive files exist:
   - `MyDrive/wild_diff_icmh/results/phase1_metadata_KGA_A01.json`
   - `MyDrive/wild_diff_icmh/results/phase1_closeout_KGA_A01.json`
4. Report the printed 2K forecast and recommendation. The phase remains open until this
   real L4 artifact exists; no automatic 2K run is required.

## Live Colab evidence (2026-10-04, branch `ver2`, commit `140830a`, NVIDIA L4)

One runtime ran Bước 1→10 into the fresh run dir `runs/phase1_calib/A01/20261004-155451`.
`phase1_closeout_KGA_A01.json` reports every gate `passed: true`:

| Gate | Evidence |
|---|---|
| split_positive | 10,222 rows, no sequence in two splits |
| split_negative | deliberate leak of sequence `KGA:KGA_S1#B02#1#99` rejected |
| metadata_100 | EXIF datetime 100/100, location 100/100; 9/100 hour-proxy vs grayscale disagreements |
| checkpoint | compact, contract v2, 1 optimizer state, global_step 21 |
| resume | 20 → 21 |
| results_registry | `phase1_h1_smoke_KGA_A01`, 10 rows, idempotent |

Calibration: steady 5.53 s/optimizer step (136 batches, 16 validation runs excluded) vs
20.21 s/step wall clock; 2K steps ≈ 3.07 h ≈ 4.73 CU; measured session cost 1.37 CU
(Colab "Available" delta); recommendation `run_2k_allowed_not_started`. Environment:
Python 3.13.15, torch 2.11.0+cu130, Lightning 2.6.6, NumPy 2.1.3, pyiqa 0.1.15.post2.

Carried forward: a mid-run kill/resume (PRE-06) is demonstrated on the first H1 run
(rolling checkpoint every 50 steps).

