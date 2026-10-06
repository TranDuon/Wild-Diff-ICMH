---
quick_id: 261006-hxn
status: complete
date: 2026-10-06
commit: 4288862
---

# Summary — Phase 2 Eval notebook step 7 fix

## Root cause (from the Drive logs of 2026-10-05)

- `mbt2018` / `cheng2020-attn` are autoregressive: real entropy coding walks the latent
  position by position (~29 s/img at 1024 on L4, ~2.4 h per 300-image point).
- Scoring was deferred to step 8, and baseline reconstructions live on the local SSD by
  design, so when the runtime died at 01:58 (start of `mbt2018_ls1024_q3`), nothing had
  been scored and ~8 h of baseline work was lost. No `*.done.json` existed on Drive.

## Changes

- `tools/baselines/run_compressai_zoo.py`: `--rate auto|coded|estimated`. The autoregressive
  models default to one forward pass with bits = sum(-log2 likelihood), which is CompressAI's
  `--entropy-estimation` convention. The bits are logged as `estimated_bits`, and `run_info.json`
  carries `rate`. The hyperprior model keeps real coding.
- `tools/evaluate_kgalagadi.py`: falls back to `estimated_bits` when `data/<stem>` is absent,
  adds `rate_source` to each row and `rate` to the protocol.
- `tools/build_colab_phase2_notebook.py` → `Wild_Diff_ICMH_Phase2_Eval.ipynb`:
  - `score_archive()` is defined in step 4; steps 6 and 7 score each point right after
    coding it, and step 8 covers B0 plus any leftovers.
  - The bitstream tar is written before the done marker.
  - The time table now gives realistic figures.
  - Step 11 never raises: it records the session, runs `drive.flush_and_unmount()` and
    then `runtime.unassign()`, controlled by `AUTO_DISCONNECT`.
- `.planning/REQUIREMENTS.md` EVAL-14 records the estimated-rate protocol decision.

## Follow-up (same day, user direction)

The user's goal for this notebook is limited to two things: baselines to compare against, and
MegaDetector bboxes before and after decode. Compute units are no longer recorded.

- Baselines run on `BASELINE_DEV`, the largest image set B0 decoded (100 dev images at 512),
  so every comparison is paired. Before this change the baselines used 300 images while B0
  used 100.
- Both Phase 2 notebooks have their own step 1 (Drive only); the last cell flushes Drive and
  disconnects.
- Dropped the domain-stats step (already done in Prepare P2-4). The RD plot and the table focus
  on mAP, missed animals and hallucinated animals.
- Steps renumbered 1–10; estimated run time is ~1.5 h.

## Verification

- `python -m pytest -q`: 161 passed, 3 skipped.
- Torch venv: `tests/test_compressai_rate.py` passes. It runs the full code → evaluate path
  with a stub CompressAI model (CompressAI has no Windows wheel). The real models run only
  on Colab.
