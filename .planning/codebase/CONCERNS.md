---
last_mapped_commit: working-tree
---
# Technical Concerns

**Analysis Date:** 2026-09-24

## Must Validate on Colab

- Lightning 2 compact-checkpoint resume has static coverage but no real GPU run yet.
- Actual peak VRAM and step time for L4 must be measured before all-site execution.
- Current 5 H1 epochs and 2 continuation epochs are starting budgets, not proven optima.
- RAM++ tag cache speed and output quality on Kgalagadi night images need measurement.

## Scientific Risks

- The comparison paper does not publish its split; exact numeric reproduction is impossible.
- Same-site train/test matches that paper's per-site goal but does not test unseen-site transfer.
- MegaDetector rectangles are pseudo-labels and can bias H2/foreground SSIM.
- H2 must be compared with the compute-matched H1-control, not only the earlier H1 checkpoint.
- H3 must include all tag/header/metadata bytes and cannot use ground-truth species prompts.

## Implementation Risks

- `inference_partition.py` extends a legacy bitstream without a formal version marker.
- H3 decode must be invoked with matching tag/metadata settings or parsing will disagree.
- Old notebooks contain historical compatibility patches and are not training authority.
- Legacy code outside the active DiffEIC path still contains old checkpoint loaders.

## Data Risks

- Kgalagadi has very few night images, so night-only confidence intervals may be wide.
- Several sites have few sequences; per-site validation/test estimates may be noisy.
- Drive small-file I/O is slow; training directly from mounted Drive can appear hung.

## Scope Discipline

- Do not run H3 training for L1/L2; it is an inference-only ablation.
- Do not enable H2 semantic weighting until cheap V1 shows signal.
- Do not claim superiority from one BPP point; build matched RD curves.

*Concern analysis: 2026-09-24*
