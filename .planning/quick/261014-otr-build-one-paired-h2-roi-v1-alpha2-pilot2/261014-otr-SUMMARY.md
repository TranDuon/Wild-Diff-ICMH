---
quick: 261014-otr
status: complete
completed: 2026-10-14
implementation_commits: [c4a0373]
experimental_status: h2_gpu_pilot_pending
---

# One paired H2 ROI-V1 pilot in the existing Recovery notebook

Implementation and CPU verification are complete. The actual H2 GPU run,
visual mask approval, 30-image evaluation and phase acceptance are not. A local
commit alone does not update the user's open Colab repository; the final push
must succeed before they run the continuation cell.

## Delivered

- Separate H2 alpha2/background1, ROI-loss-only config and 200-step run. The
  runner checks the completed H1 low-LR recovery200 control, author-lambda2
  source, frozen dev30 IDs, manifests/tags, checkpoint provenance and resume
  identity before any training or comparison. It trains with a fresh optimizer
  and records H2-only ROI/background loss, mask fraction, rate terms and actual
  learning rates. Old H1 run/config/checkpoints are not changed.
- Real dataset resize/crop/flip and latent-mask overlay preflight with digest
  acknowledgement. A malformed or foreign source fails closed. After approval,
  only H2 is decoded/scored on the same30; cached H1/B0 are read, not rerun.
  Valid detector rows can be recovered after an interrupted final JSONL line
  while preserving the original file; corrupt interior rows still fail closed.
- Existing `Wild_Diff_ICMH_H1_Recovery.ipynb` rebuilt in place. Its new H2
  continuation uses the already-open Colab kernel, `/content` assets and
  detector environment, with explicit preflight/train/evaluate actions. The
  open GitHub-backed Colab page needs one once-only cell because pulling a
  changed notebook file does not modify a live open cell. No new notebook,
  runtime reset, H1 retrain, Xie recreation, alpha sweep, full-dev or test run.
- Narrow opt-in `preflight.asset_splits` in `train.py` permits H2 train/val
  staging without requiring held-out test image copies. Whole-manifest split
  leak checks remain; old H1 configs retain their behavior.

## Actual control and interpretation

On the frozen dev30, B0 lambda2 had 0.005990175 bits/pixel, detector AP
0.394392636 and6/20 positive frames missed. Completed H1 low-LR recovery200
had 0.005352521 bits/pixel, AP0.408348500 and the same6/20 missed. H1 used
about10.65% fewer bits at this measured point; neither result proves a
matched-rate curve win or a paper-level improvement. H2 has no result yet.

## Verification

- Full CPU suite before final hardening: **299 passed, 2 skipped**. After the
  train/val asset scope and interrupted detector-tail fixes, affected suite:
  **45 passed**. The canonical notebook builder completed successfully.
- The seven existing H1 config-chain files match their live H1 contract hashes
  after normalizing Windows CRLF to the Colab LF representation; no H1 config
  diff. This check guards scientific pairing but does not replace an L4 run.
- No Colab/Drive GPU job or benchmark was launched by this implementation.

## Handoff

After code reaches `main`, continue in the **same current Recovery notebook and
L4 runtime** using the once-only cell in `docs/H2_ROI_PILOT.md`. Read and
approve actual overlays before training. Run one bounded H2 200-step pilot,
then one H2-only dev30 evaluation; return H2 preflight/comparison, throughput
and loss CSV for review. Keep Phase 3/4 open until experimental gates pass.
