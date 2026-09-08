# Domain Pitfalls: Wild-Diff-ICMH

**Domain:** Domain-specialized generative (diffusion) image codec fine-tuning for camera-trap wildlife imagery, evaluated with machine-vision tasks, 2-person team, ~10 weeks, Google Colab Pro (~100 compute units/month, no dedicated GPU)
**Researched:** 2026-09-07
**Confidence:** MEDIUM overall (HIGH for the two most-damaging pitfalls, which are corroborated by the team's own risk register and standard camera-trap ML literature; LOW-MEDIUM for narrower claims sourced only from web search — flagged inline)

This file both (a) surveys pitfalls specific to this exact project shape and (b) **audits the team's own risk register** in `docs/ke-hoach-difficmh-wildlife-8-tuan.md` §4.3 (R0–R12), correcting entries that assumed a dedicated 24GB GPU running 24/7 (1,344 GPU-hours) when the real constraint is Colab Pro's ~100 compute units/month (~20h L4 or ~7.5h A100/month, ≈50h L4-equivalent for the whole project — roughly 5-6x less than the plan assumed).

---

## Critical Pitfalls

### Pitfall 1: Site/burst data leakage across train/val/test splits

**What goes wrong:**
Camera traps are bolted to a fixed point and fire in bursts (3-10 frames within seconds of a trigger). A single site accumulates thousands of images sharing an almost pixel-identical static background (same tree, same waterhole, same rock) across an entire season. If the split is done **per image** (or even per burst but not per site), the model — and every downstream detector/classifier used for evaluation — can trivially memorize the background texture rather than learning to compress/detect the animal. Every metric reported afterward (bpp, PSNR, mAP, species accuracy) becomes inflated and none of it transfers to a genuinely new camera. This is not a small effect: it is large enough to invalidate the entire 10-week study, because the core scientific claim ("H2 or H3 beats H1 fine-tuning") could just be "the model memorized which pixels are background at this specific site."

**Why it happens:**
Standard `sklearn.train_test_split` / random shuffling treats every image as an i.i.d. sample. Camera-trap corpora violate i.i.d. twice over: (1) burst frames are near-duplicates of each other (temporal leakage), (2) all images from one physical site share a background distribution across the whole deployment period, sometimes months (spatial/site leakage). A 90/10 random split silently puts frames from the same burst, or images from the same site take at different times, into both train and val. This is the single most common and highest-impact mistake in camera-trap ML — confirmed by the camera-trap literature and dataset design conventions (Snapshot Serengeti, Caltech Camera Traps, iWildCam): benchmarks such as iWildCam explicitly hold out entire camera **locations** for test, precisely because splitting by image (even with class-stratification) causes background memorization and inflated, non-transferable scores. [MEDIUM confidence — corroborated across multiple camera-trap papers and the iWildCam competition design]

**How to avoid:**
- Split at the **`location`/site ID** granularity, not image or even burst granularity: pick whole sites for val/test, never let a site appear in more than one split.
- Additionally collapse bursts: group images by `seq_id` (LILA/COCO Camera Traps schema field) or, if `seq_id` is absent, reconstruct pseudo-sequences by grouping frames from the same `location` within a short time window (~3 seconds, per LILA's own documented approach) — then assign the whole sequence to one split.
- Write `split_check.py` **before** building the split (not after) — this is explicitly called out as the team's own single line of defense for what they label R0, and it should run as an automated assertion, not a manual eyeball check:
  ```python
  def assert_no_leakage(train_meta, val_meta, test_meta):
      train_locs = {m["location"] for m in train_meta}
      val_locs   = {m["location"] for m in val_meta}
      test_locs  = {m["location"] for m in test_meta}
      assert train_locs.isdisjoint(val_locs), f"Leaked locations train/val: {train_locs & val_locs}"
      assert train_locs.isdisjoint(test_locs), f"Leaked locations train/test: {train_locs & test_locs}"
      assert val_locs.isdisjoint(test_locs), f"Leaked locations val/test: {val_locs & test_locs}"
      # burst/sequence check — belt and suspenders even though location split implies this
      train_seqs = {m["seq_id"] for m in train_meta if m.get("seq_id")}
      val_seqs   = {m["seq_id"] for m in val_meta if m.get("seq_id")}
      assert train_seqs.isdisjoint(val_seqs), f"Leaked sequences train/val: {train_seqs & val_seqs}"
  ```
  Run this as a CI/pre-commit-style gate that must pass before any training or eval job reads the split files, and persist the frozen split as `location_split.json` (site IDs per split) plus the literal image-ID lists — not just a script that regenerates a "random" split every run, since LILA source datasets get revised over time.
- CCT (Caltech Camera Traps) should be held out **entirely** from training and used only as an unseen-site generalization test (the team's plan already does this correctly — verify it stays true after any dataset re-sampling).
- When measuring the "cost of specialization" against COCO/Kodak (cross-domain eval), this leakage concern doesn't apply (different dataset entirely) — but the CCT-as-new-site generalization number is exactly the test that catches leakage if it was missed: if in-domain (Snapshot Serengeti val) numbers look great but CCT numbers collapse, suspect leakage in the SS split first.

**Warning signs:**
- Val/train metrics (bpp, PSNR, mAP) are implausibly close, or val metrics are *better* than train metrics.
- `split_check.py` assertion never actually gets run/wired into the pipeline (a script existing in the repo unused is not protection).
- A location/site ID appears in the file lists of more than one split — check by literally running `set(train_locations) & set(val_locations)` and confirming it prints `set()`.
- Detection/classification accuracy on the held-out CCT (new-site) eval is dramatically worse than on the Snapshot Serengeti val split (e.g., >15-20 mAP points) — this is the downstream symptom of a leak that inflated SS val numbers.
- Burst frames of the same animal instance appear split across train and val when manually spot-checking 20 random val images against their timestamps/site IDs.

**Phase to address:**
Data Foundation phase (equivalent to the plan's T1.1) — must be closed in Week 1, before any baseline number is trusted, and must be gated (not merely attempted) before the Week 2 training-loop go/no-go checkpoint. This is correctly identified as the highest-probability-of-silent-catastrophe risk in the team's own register (R0) and should stay the single highest-priority verification item in the roadmap.

---

### Pitfall 2: Colab Pro compute-unit exhaustion (project dies mid-run, not just mid-session)

**What goes wrong:**
The team's original 8-week plan (`docs/ke-hoach-difficmh-wildlife-8-tuan.md`) assumed **a dedicated 24GB GPU running 24/7 for 8 weeks = 1,344 GPU-hours**, against an estimated need of ~290h (a 4.6x safety margin) — its entire risk calculus (R10 "GPU contention in week 6", the 7-run training schedule, the 3-bitrate-point RD curve, weekly milestones) is built on that abundance. The actual environment is **Google Colab Pro, ~100 compute units/month**, which converts to roughly ~20h of L4 or ~7.5h of A100 per month, i.e. **~50h of L4-equivalent compute for the entire ~10-week project** — a **5-6x smaller budget** than assumed, not a rounding error. This single mis-scoped constraint, if not corrected in the roadmap, causes the project to run out of usable compute partway through Week 5-6 of the *original* plan while believing it still has budget, because the plan's mental model of "GPU-hours available" no longer matches reality (compute units, not wall-clock hours, are the scarce resource, and they do not reset until the next monthly billing cycle).

**Why it happens:**
Colab Pro is sold as a subscription with "priority access" language, which reads like unlimited-hours-per-session availability; teams moving from a dedicated-GPU mental model (where the constraint is calendar time) don't re-derive their run budget in compute-unit terms, and don't track burn rate against the monthly reset boundary. Compute-unit consumption also varies by GPU tier (A100 ≈ 5.4-7.5 CU/hr vs L4/T4 much cheaper), and Colab does not guarantee which tier you get on any given session (see Pitfall 6b), which makes naive "hours planned" arithmetic doubly wrong: the same *planned* wall-clock hours can cost wildly different compute-unit amounts depending on which GPU is silently assigned.

**How to avoid:**
- Reforecast the entire training/eval schedule in **compute units**, not GPU-hours, before committing to a run plan. Convert every planned run (7 training runs × ~20h/run in the original plan = ~140h alone) into an estimated CU cost using the *actual* tier likely to be assigned, and check the running total against ~100 CU/month × number of months in the project window (~2.3 months at 10 weeks → ~230 CU total ceiling, optimistically, and that assumes the team is willing to pay for multiple months of Colab Pro — confirm this budget assumption explicitly with the team, it is not stated in PROJECT.md).
- Cut scope **before** hitting the wall, not after: the plan's own H4 (task-aware SC loss) is already correctly marked out-of-scope for exactly this reason: it needs ≥60 GPU-hours the compute-unit budget does not have. Apply the same lens to the 7-run / 5-bitrate-point / 3-hypothesis plan — at 5-6x less compute, the realistic ceiling is closer to 1-2 full fine-tuning runs at reduced iteration count (thousands, not tens of thousands, of iterations) plus a handful of short ablations, not 7 full ~20h runs.
- Track compute-unit burn explicitly: log estimated CU cost per job in `results.jsonl` (add a `cu_estimate` field) so the team has a running total visible at any time, not just a GPU-hours tally.
- Treat "smoke test" (T2.3-equivalent) throughput measurement as **mandatory before locking any downstream schedule** — do not plan Week 3-8 run counts off the paper's or the plan's assumed throughput; measure actual it/s on whatever GPU tier Colab hands out that day and multiply out real CU cost per planned run before committing.
- Default to the cheapest viable GPU tier deliberately (don't chase A100) — L4 is far more CU-efficient per hour even if slower per-step, so total useful compute-hours per CU spent is often higher on L4. Only escalate to A100 for genuinely VRAM-bound stages.
- Buy Colab Pro+ or top up compute units explicitly if the 10-week timeline spans more than one monthly billing cycle and the team is relying on units resetting — do not assume units silently carry over or replenish mid-project without action.

**Warning signs:**
- Any single planned run's estimated hours, multiplied by that session's actual CU/hr rate, consumes a large fraction of remaining monthly units in one sitting — recompute the whole remaining schedule immediately if this happens.
- The Colab UI shows compute units below ~20 units with more than one training run still planned for the month — this is the point where the team must ruthlessly cut to the sacrifice order (H4 first, then bitrate points beyond 2, then ablation cells) rather than hoping units "come back."
- Smoke-test throughput (it/s measured in Week 2-equivalent) is markedly worse than the paper's Table 2 numbers (13.14s/image at 512² on A100, 50 DDIM steps) scaled to L4 — if the team is quietly assigned T4-class hardware, real throughput can be 2-4x worse than assumed, silently blowing every downstream time/CU estimate.
- A run is queued expecting A100 and Colab assigns a lower tier (see Pitfall 6b) — if this isn't detected and the run proceeds with the original A100-derived hyperparameter/time budget, the run will either time out mid-training or silently take 2-3x longer, consuming that much more of the monthly CU allotment for the same iteration count.

**Phase to address:**
This must be resolved **before** the roadmap commits to a specific number of training runs / bitrate points / hypotheses — i.e., during initial roadmap/phase planning, not discovered in Week 5-6 as a crisis. Concretely: the roadmap phase that plans H1/H2/H3 run counts should carry an explicit CU budget line derived from measured (not assumed) throughput, and every phase gate should include "did this consume more CU than budgeted" as an exit check, not just "did the metric improve."

---

## Fine-Tuning Failure Modes (Diffusion Codec)

### Pitfall 3: Catastrophic forgetting / learning-rate-driven prior destruction

**What goes wrong:**
Fine-tuning the codec encoder-decoder (and control module) on a narrow domain (camera-trap imagery) at too high a learning rate overwrites the pretrained representations fast enough that the model degrades on anything outside the fine-tuning distribution — including, in the worst case, degrading *within*-domain generation quality if the LR destabilizes the interaction between the (frozen) SD UNet prior and the (trainable) codec/control module. Since the control module's entire job is to steer the frozen SD prior correctly, an LR high enough to "overwrite" it can produce visibly broken decodes (artifacts, texture collapse) even on training-distribution images, not just OOD ones.

**Why it happens:**
The plan's own choice of `lr = 1e-5` (vs. the paper's stage-2 `5e-5`) reflects awareness of this risk in principle, but the real danger is subtler than "pick a low number": per-step forgetting has been shown to scale with the product of learning rate and √(current loss) — meaning high-loss batches (which are common early in fine-tuning, when the domain gap is largest) are disproportionately destructive even at a nominally "safe" LR. A single LR value chosen up front doesn't protect against this if early-training loss spikes are large. [MEDIUM confidence — general diffusion/deep-learning fine-tuning literature, not camera-trap/codec specific]

**How to avoid:**
- Use a warmup schedule (do not start fine-tuning at full target LR from iteration 0) to avoid large early-loss-driven forgetting steps.
- Log both the in-domain loss curve *and* a small fixed-size out-of-domain sanity set (e.g., 20-50 Kodak/COCO images) decoded at fixed checkpoints throughout training — if OOD reconstruction quality (PSNR/LPIPS) craters early, the LR is too high regardless of in-domain loss looking fine.
- Prefer the smallest LR that still shows measurable in-domain improvement within the iteration budget over a higher LR that converges faster — given the ~5-6x smaller-than-planned compute budget (Pitfall 2), there won't be spare runs to recover from a destroyed prior.
- Keep the very first checkpoint (iteration 0, i.e., the unmodified pretrained weights) as a permanent, never-overwritten reference artifact so "cost of specialization" comparisons are always possible even if a later run degenerates.

**Warning signs:**
- Training loss log line shows `l_simple` or `l_bpp` spiking sharply (order-of-magnitude jump) in the first few hundred iterations — a spike, not a dip, is the signature of LR-driven forgetting, not normal optimization noise.
- Decoded validation images at early checkpoints show visible degradation *worse* than the frozen pretrained checkpoint on the same input — this should never happen and is a hard stop signal.
- OOD sanity-set (Kodak/COCO) reconstruction PSNR drops by more than ~1-2dB within the first 10% of planned iterations, before in-domain gains have had time to accrue.

**Phase to address:**
H1 domain-adaptive fine-tuning phase (the plan's T3). The mitigation (OOD sanity checkpointing) should be built into the training harness during the training-infrastructure phase (T1.3/T2.3-equivalent), not added reactively after a run is already degraded.

---

### Pitfall 4: Degenerate rate collapse (bpp → 0 or bpp explodes)

**What goes wrong:**
A rate-distortion codec being fine-tuned with an imbalanced `λ_rate` vs `λ_dist`/`λ_sem`/`λ_diff` weighting, or a training instability, can collapse to a degenerate solution: either the entropy model drives bpp toward ~0 (the codec stops encoding meaningful information, relying entirely on the frozen SD prior/control module to hallucinate plausible-looking output from almost nothing) or bpp explodes (the rate loss term stops effectively constraining the entropy bottleneck, e.g., due to a numerical issue in the hyperprior). Either failure silently produces a model that looks like it's "training" (loss decreasing) while producing scientifically meaningless RD points.

**Why it happens:**
Multi-term losses (`l_bpp`, `l_guide`, `l_simple`, `l_semantic_weight` in `model/diffeic.py`) combine terms with very different natural scales and gradient magnitudes; a config change (e.g., swapping in ROI-weighted `L_dist` per H2, or a new `λ_rate` sweep point) can shift the effective balance without anyone noticing, especially at the extreme ends of the planned `λ_rate ∈ {2, 8, 32}` sweep. Fine-tuning from a converged pretrained checkpoint makes this worse, not better, because the entropy model starts near a specific operating point and a mis-set `λ_rate` at fine-tune time can push it off that point faster than it would during from-scratch training (where the loss landscape is being explored more broadly anyway).

**How to avoid:**
- Log bpp as its own explicit metric every N iterations (not just total loss) and set a hard sanity bound per `λ_rate` config, e.g., "bpp should stay within [X, Y] given this λ_rate" derived from the pretrained checkpoint's own operating point at that λ_rate as the starting reference.
- Log each loss term (`l_rate`, `l_dist`, `l_diff`, `l_sem`) **separately** in every training log line, not just the summed total — this is the only way to catch one term silently dominating or vanishing.
- When sweeping `λ_rate ∈ {2, 8, 32}`, run a short (few-hundred-iteration) smoke check at each extreme before committing a full run's compute budget, specifically watching for bpp collapsing toward 0 or diverging upward in the first few hundred steps.
- If ROI-weighting is added to `L_dist`/`L_sem` (H2), verify the *rate* loss term's relative magnitude hasn't implicitly shifted too, since normalization changes (e.g., `w_n / Σw_n` per the plan's H2 design) change gradient scale relative to `l_rate`.

**Warning signs:**
- `bpp` metric in the training log trends toward 0 while `l_simple`/reconstruction loss keeps decreasing (or stays flat) — the model has stopped using the bitstream and is being carried by the frozen generative prior. Decoded images at this point often still "look fine" perceptually (SD is generating plausible content) despite carrying almost no information — this is exactly the hallucination risk described in Pitfall 8, compounded.
- `bpp` grows without bound relative to the configured `λ_rate`'s expected operating range from the pretrained checkpoint.
- Any individual loss term (`l_rate`, `l_dist`, `l_diff`, `l_sem`) is more than ~2 orders of magnitude larger or smaller than the others in the same log line, without an explicit weighting justification.

**Phase to address:**
H1 fine-tuning phase and every subsequent phase that touches the loss function (H2 ROI-weighting, any `λ_rate` sweep). The per-term logging and bpp sanity-bound check should be built into the training harness once, during infrastructure setup, and reused by every later phase rather than re-implemented per-hypothesis.

---

### Pitfall 5: Fine-tuned model's bpp range doesn't overlap the baseline's — BD-rate becomes uncomputable

**What goes wrong:**
BD-rate (the standard RD-curve comparison metric) requires interpolating between rate-distortion points on two curves at *matched* distortion or rate values. If the fine-tuned model's achievable bpp range (given its λ_rate sweep) doesn't overlap the pretrained baseline's bpp range on the same eval set, BD-rate cannot be computed without extrapolation — and extrapolated BD-rate numbers are not meaningful/comparable, effectively an unreportable result after a large fraction of the compute budget has already been spent generating it. [MEDIUM confidence — general RD-curve/BD-rate methodology, confirmed by learned-compression literature which explicitly separates evaluation when bitrate ranges don't overlap]

**Why it happens:**
Domain specialization (H1 fine-tuning) is expected, and intended, to shift the operating point of the codec — that's the whole "PSNR should improve +1.0-2.5dB" hypothesis in the plan. But a large enough shift, especially combined with a λ_rate sweep chosen independently for the fine-tuned model vs. reused verbatim from the baseline, can push the fine-tuned curve's bpp range entirely above or below the baseline's, especially at the compute-constrained iteration counts this project can afford (undertrained models at extreme λ_rate values are more likely to land at unexpected bpp than fully-converged ones).

**How to avoid:**
- After the *first* short fine-tuning run at any given λ_rate, immediately measure actual achieved bpp on the eval set and compare against the baseline's bpp at the same λ_rate — do this before committing further compute to more λ_rate points.
- If a gap is detected, add an intermediate λ_rate point specifically chosen to land inside the missing overlap region, rather than waiting until the full RD curve is assembled (T6.3-equivalent) to discover the gap — the plan's own early-warning table already flags this exact failure mode ("Vùng bpp của model mới không chồng lấn baseline") and correctly recommends fixing it "at first detection, not at T6.3" — this is good practice and should be preserved as a standing warning-sign check, not just a one-off note.
- Keep at least one λ_rate configuration identical (same value, not just same nominal range) between the fine-tuned and baseline sweeps as an anchor point, rather than choosing sweep points for each independently.

**Warning signs:**
- The lowest bpp achieved by the fine-tuned model at its highest λ_rate is still higher than the baseline's bpp at the baseline's highest λ_rate (or vice versa) — check this as soon as the first 2-3 RD points exist per model, not after the full sweep.
- BD-rate computation throws a numerical warning about extrapolation, or the underlying interpolation library silently extrapolates linearly past the data range without warning (verify: log a hard assertion that all comparison points fall within the convex hull of both curves before trusting a BD-rate number).

**Phase to address:**
Should be checked incrementally starting from the H1 fine-tuning phase (as soon as the first RD point exists) and again at the multi-bitrate-point/ablation phase (T6-equivalent) before any BD-rate number is written into the report.

---

## ROI-Weighted Loss Pitfalls (H2)

### Pitfall 6: Background starvation paradox — prioritizing the animal makes detection worse

**What goes wrong:**
Weighting `L_dist`/`L_sem` toward the ROI (animal) region at high α starves the background region of bit budget, producing unnatural, "clumped" or repetitive background texture (since the generative prior fills in low-bit-rate background with plausible-but-wrong high-frequency detail). Camera-trap detectors are already known to be prone to false positives on moving foliage, shadows, and textured ground — feeding them background with generative-model artifacts (rather than natural camera noise/texture) can shift the false-positive distribution in ways the detector was never trained to reject, so **detection performance can get worse even though the animal region itself is better preserved**. This is the paradox explicitly named in the research question and already anticipated (correctly) in the team's own plan as R5, citing the paper's own Figure 3 as a precedent for this exact mechanism.

**Why it happens:**
Object detectors and classifiers are trained on the statistics of natural images, including natural background noise/texture correlations. A generative codec's background region at low effective bitrate is not "noisy" in the way a lossy DCT/wavelet codec's background would be (blocky, blurry) — it's "wrong" in a different, potentially more deceptive way: locally plausible, globally inconsistent generative texture, which detectors have no learned prior to discount the way they've learned to discount JPEG blockiness. High α thus doesn't just reduce background fidelity, it changes background fidelity's *failure mode* into one the eval task models are unprepared for.

**How to avoid:**
- Sweep α starting from lower values than the paper's own comparable settings might suggest, and treat "false positive rate on empty (no-animal) images" as a first-class metric tracked at *every* α, not just AP on animal images — the plan already recommends this ("theo dõi riêng false-positive rate trên ảnh rỗng"); this should be wired into the eval harness as a required column in `results.jsonl` from the start of H2, not added ad hoc.
- Report mAP split by object size (AP_small/medium/large) as the plan specifies — but also visually inspect a handful of decoded background regions at each α for artifact character (clumping, repetition, texture "melting") before trusting quantitative detector numbers; a paradoxical FP increase is much easier to diagnose with 5 minutes of visual inspection than by staring at aggregate mAP deltas.
- Don't default to "higher α = better" as an assumption anywhere in the analysis pipeline (e.g., don't auto-select the highest α for the "final" config without checking the empty-image FP metric).

**Warning signs:**
- Empty-image false-positive rate increases monotonically with α while AP on animal-containing images also increases — this is the paradox actually manifesting, not a wash; report both numbers, don't let an aggregate mAP improvement hide a FP regression.
- Visual spot-check of decoded background at high α shows repetitive/tiled texture patterns or texture that doesn't match the semantic content of the original background (e.g., grass rendered as an indistinct green smear, or leaf patterns that look "stamped").

**Phase to address:**
H2 ROI-weighted loss phase (T4-equivalent). The empty-image FP metric must be part of the eval harness *before* the α sweep begins, not retrofitted after results look surprising.

---

### Pitfall 7: Resolution mismatch between where loss is computed and where objects live in latent space

**What goes wrong:**
`L_dist` operates on the VAE latent (8x downsample) while `L_sem` (SC loss) operates on the SD UNet middle block (64x downsample) in the paper's default configuration. At 64x downsample, a mid-sized animal in a 256px crop occupies well under one latent cell — ROI-weighting `L_sem` at the middle block is then **mathematically almost meaningless** for this domain, since the weight mask itself, downsampled to that resolution, degenerates to near-uniform (the animal region rounds away to nothing distinguishable from background at the cell level). This is a subtle bug-shaped pitfall: the code will run, the loss will compute, gradients will flow, and nothing will error — but the ROI-weighting term contributes essentially nothing to the intended effect, silently wasting the implementation and analysis effort spent on H2's SC-loss variant.

**Why it happens:**
The paper's SC loss placement (middle block) was chosen for its original task-agnostic, general-image objective, not with small-object-dominant domains in mind. Naively porting `L_dist`'s ROI-weighting approach to `L_sem` without checking the actual latent resolution at the chosen tap point is an easy mistake, especially since the code change itself ("swap `1/N` for `w_n/Σw_n`") is trivially simple and doesn't force anyone to reconsider *where* in the network that sum is being computed.

**How to avoid:**
- Before implementing H2's SC-loss ROI-weighting, compute the actual downsample factor at the chosen tap point for the actual crop size in use (256² per this project's VRAM-constrained config, not the paper's 512²) and check the resulting cell count covered by a typical/small bounding box — the plan's own analysis already did this arithmetic correctly (Encoder Layer 9 at 32x downsample chosen over middle block at 64x, citing exactly this reasoning) and should be preserved as a documented design decision, not re-derived ad hoc later by someone who didn't read that section.
- Add a unit-test-style sanity check: given a known bounding box and crop size, assert the downsampled ROI mask at the chosen tap point has at least some minimum number of "hot" cells (e.g., ≥4) for a representative sample of training boxes — if a large fraction of masks degenerate to 0-1 hot cells, the tap point is too coarse for this domain's object sizes.
- Report, in the ablation/ROI analysis, the distribution of "cells covered by ROI mask" at whatever tap point is used, split by animal size bucket — this makes the resolution-mismatch limitation visible and quantified in the report rather than an unstated assumption.

**Warning signs:**
- A large fraction (spot-check ≥50) of training-time ROI masks, once downsampled to the SC-loss tap resolution, contain zero or one "hot" cell — the weighting term is then indistinguishable from unweighted for those samples.
- H2's SC-loss variant shows no measurable difference from H2's `L_dist`-only variant (V1 vs V2 in the plan's own ablation design) — this could be a legitimate negative result, but should first be checked against the mask-resolution sanity check above before being reported as "SC-loss ROI-weighting doesn't help," since a resolution-mismatch bug would produce exactly the same null result.

**Phase to address:**
H2 ROI-weighted loss phase (T4-equivalent), specifically before implementing the V2 (L_dist + L_sem) variant — verify the tap-point resolution choice against actual object-size statistics from the Phase 1 dataset analysis before writing the SC-loss weighting code.

---

### Pitfall 8: Mask/image crop misalignment — a mask silently not reaching the loss

**What goes wrong:**
ROI masks are generated (via SAM from bbox ground truth) at full-image resolution, then must be cropped/resized/aligned to match whatever training crop and latent downsample the loss actually operates on. A coordinate-system bug (off-by-one in crop offset, forgetting to apply the same random-crop transform to both image and mask, a flip/rotation augmentation applied to the image but not the mask, or an incorrect interpolation mode when downsampling a binary mask to latent resolution) silently produces a mask that no longer corresponds to the actual animal location in the crop — or, in the worst case, a mask that's accidentally all-zeros or all-ones for some/all samples, which makes the "ROI-weighted" loss degenerate back to the unweighted loss without any error being raised.

**Why it happens:**
This is a classic silent-failure bug class: shape-compatible tensors (a wrong mask is still a valid-shaped tensor) produce no exception, and the resulting loss values look plausible (still a number, still decreasing) even when computed against a misaligned or degenerate mask. It's especially easy to introduce when adding oriented/bbox-biased crop sampling (per the plan's own recommendation to bias crops toward containing animals) — the crop-sampling logic and the mask-generation logic are natural candidates to be written/maintained somewhat independently, increasing the chance their coordinate transforms drift out of sync.

**How to avoid:**
- Log `L_dist_roi` (loss restricted to ROI pixels) and `L_dist_bg` (loss restricted to background pixels) as **separate metrics** in every training log line, not just the combined weighted loss — this is exactly the check the team's own plan already lists as an early-warning signal ("`L_dist_roi` và `L_dist_bg` không tách biệt trong log" → "Mask không thực sự vào loss — dừng run, kiểm tra `mask_utils.py`"), and it should be implemented as a mandatory, always-on training log field from the very first H2 run, not something added after a run is already suspected of being broken.
- Add a periodic (every N iterations) visual dump: overlay the downsampled mask on the corresponding crop and save as an image artifact for manual inspection — cheap to build, and the single fastest way to catch a coordinate-system bug that a scalar metric might not reveal for many iterations.
- Assert, per-batch during a debug run, that `mask.sum() > 0` for some minimum fraction of samples in a batch that are known (from the crop-sampling metadata) to contain at least one bbox — a systematically all-zero mask on bbox-containing crops is the clearest possible signature of a broken pipeline.
- Unit-test the crop+mask transform pipeline in isolation with a synthetic image (e.g., a single white pixel at a known coordinate standing in for the mask) and verify the mask ends up at the geometrically correct location after the full augmentation chain, before ever running it against real data.

**Warning signs:**
- `L_dist_roi` and `L_dist_bg` are numerically identical or near-identical across training (the signature explicitly named in the plan's own early-warning table) — this means the mask is not actually distinguishing ROI from background in the loss computation.
- Visual mask-overlay dumps show the mask offset from, rotated relative to, or otherwise not covering the animal in the crop.
- The fraction of training crops (among those known to contain a bbox, per the oriented-sampling logic) with an all-zero downsampled mask exceeds a few percent.

**Phase to address:**
H2 ROI-weighted loss phase (T4-equivalent) — the split-loss logging and mask-overlay dump should be added to the training harness on day one of implementing ROI-weighting, not retrofitted after a run's results look suspicious. This is cheap insurance (a few lines of logging code) against a bug class that would otherwise be very hard to detect from aggregate metrics alone.

---

## Generative Hallucination in Scientific Imagery

### Pitfall 9: Diffusion decoder synthesizes plausible-but-fictional content — a scientific-integrity problem, not an aesthetic one

**What goes wrong:**
A diffusion-based decoder (as opposed to a deterministic transform-coding decoder like JPEG/VTM) can, at low bitrate, synthesize locally plausible detail that was never present in the source image — texture, edges, or in a worst case an animal-shaped blob in a background region, or a plausible-but-wrong pattern on an animal's coat. For a general perceptual-quality use case this is a known, accepted tradeoff (perception-distortion tradeoff). For ecological/scientific camera-trap data, this crosses into scientific-integrity territory: if a compressed image is later used (by an ecologist doing manual verification, or by a downstream classifier whose output is trusted) to assert presence/absence, count, species identity, or behavior, a hallucinated feature is not merely an aesthetic defect — it is a fabricated data point that could corrupt ecological records. This risk is structurally different from, and more severe than, ordinary lossy-compression artifacts, and the literature explicitly separates this out: generative/GAN/diffusion codecs are flagged in the literature as unsuitable, without explicit safeguards, for applications requiring strict fidelity such as medical imaging or forensics — camera-trap ecological monitoring belongs in the same category. [MEDIUM confidence — general generative-compression literature explicitly names medical/forensic fidelity requirements as the reference class of concern; camera-trap-specific discussion of this exact framing was not directly found in the search performed, so the analogy, while apt, is this project's own extension of the argument rather than a directly cited precedent]

**Why it happens:**
This is not a bug — it is the intended mechanism of the entire generative-codec paradigm (the Diff-ICMH paper's whole premise is using SD's generative prior to reconstruct plausible detail from a compact latent). At the ultra-low bitrates this project's use case targets (satellite/2G/LoRa-constrained field deployment), the generative prior necessarily does more of the "work" of reconstruction relative to the actual transmitted bits, which is exactly when hallucination risk is highest. Combined with domain specialization (fine-tuning on wildlife imagery makes the model *better* at generating plausible wildlife-looking content), the risk compounds: a specialized model may hallucinate more convincingly wildlife-plausible false detail than the general-domain baseline would, precisely because specialization is working as intended for texture/context but not necessarily for factual accuracy of what's actually in a given frame.

**How to avoid (this is primarily a *reporting and measurement* problem, not a fully solvable engineering one within this project's scope):**
- Measure and report a **hallucination/fidelity risk indicator**, even if imperfect: e.g., detector/classifier agreement between the *original* uncompressed image and the *decoded* image (does the detector find an animal in the decode that isn't in the original bbox ground truth, or a different species than ground truth, at a rate above the baseline codec's rate?), tracked specifically on **empty (no-animal) images** — a hallucinated animal appearing in an empty-image decode is the clearest, most measurable proxy for this risk, and the team's plan already gestures at this with its planned "tỉ lệ ảo giác trên ảnh rỗng" (hallucination rate on empty images) metric for T7 — this should be treated as a first-class, required metric, not an optional stretch analysis.
- Explicitly compare hallucination rate between the baseline (general-domain) and the domain-fine-tuned model — if specialization increases plausible-wildlife hallucination on empty images, that is a genuinely important, reportable finding about the cost of specialization, symmetric with the "cost of specialization" cross-domain PSNR tables the plan already commits to producing.
- Write a **substantive Limitations section entry** (not a boilerplate one-liner) that explicitly states: (1) this codec can synthesize plausible content not present in the source, (2) this is a structural property of generative decoding, not a fixable bug, (3) the measured hallucination-rate-on-empty-images number and what it does/doesn't tell you about hallucination risk on non-empty images (where hallucination is harder to measure directly since ground truth already contains an animal), (4) an explicit statement that this codec, as-is, should not be used as the sole basis for ecological record-keeping (presence/absence, counts, species ID) without independent verification against the original uncompressed capture, or at minimum that decoded images used for automated pipeline decisions should be flagged as coming through a generative reconstruction step.
- Do not rely on PSNR/LPIPS alone to characterize this risk — those metrics can look good even when a small, semantically important region (an animal-shaped hallucination in a corner of frame) is fabricated, since the metric is dominated by the much larger area of correctly-reconstructed background/context.

**Warning signs:**
- Detector run on decoded empty (verified no-animal) images returns any detection above a low confidence threshold, at a rate meaningfully higher for the fine-tuned/domain-specialized model than for the general baseline — this is the direct, measurable signature of the problem this pitfall describes.
- Visual inspection of a sample of decoded low-bitrate images (especially at the most aggressive λ_rate) shows texture or shapes that, on close comparison to the source, are clearly synthesized rather than reconstructed (e.g., a "second animal" that isn't in the original, or coat-pattern detail that doesn't match the source's actual pattern).

**Phase to address:**
The hallucination-rate-on-empty-images metric must be built into the eval harness during the foundational eval-harness phase (T1.2-equivalent) so it's available from the very first baseline run onward (giving a baseline hallucination rate to compare specialized models against), and the Limitations-section content should be drafted alongside the Method/Setup writing the plan already recommends starting mid-project (their own R9 mitigation), not left to the final week.

---

## Night IR Imagery Pitfalls

### Pitfall 10: Stable Diffusion has no real prior for monochrome infrared night imagery — text conditioning can cause physically-wrong color hallucination

**What goes wrong:**
Roughly half of camera-trap imagery (all night captures) is single-channel IR, replicated to 3 channels to feed the RGB-expecting pipeline. Stable Diffusion 2.1 was trained overwhelmingly on natural-color photography and has essentially no learned prior for "this is IR, render accordingly" — its default behavior when given content-descriptive text conditioning (e.g., a TGM-generated prompt describing "a zebra at a waterhole" without any explicit IR/monochrome qualifier) is to lean on its strong color-photography prior and can synthesize a plausible-looking **color** reconstruction of a scene that was actually captured in monochrome IR. This is physically wrong output, not just an aesthetic mismatch — the model is confidently generating information (color) that literally could not have been present in the sensor data. This is confirmed directly by literature working on SD-based IR modalities: general Stable Diffusion "struggles with the infrared modality due to limited infrared knowledge, and the generated output is far from a normal infrared image" without specialized adaptation. [MEDIUM confidence — from IR-colorization/generation literature specifically discussing SD's infrared limitations]

**Why it happens:**
The TGM's text conditioning is meant to activate the right region of SD's generative prior; if the prompt content-describes the scene ("zebra," "waterhole," "grassland") without also explicitly flagging the *modality* (IR, monochrome, night), SD has no signal telling it to suppress its default color-photography behavior, and will happily hallucinate colors consistent with "zebra in grassland" rather than the achromatic reality of the actual IR capture.

**How to avoid:**
- Always include an explicit modality qualifier in the prompt for night/IR images — the team's own H3 design already proposes exactly this (`illumination` as an L2 structured attribute, and a required prompt template like `"a monochrome infrared night camera trap photo of..."`), and this should be treated as a **hard requirement**, not one of several template options to A/B test casually: any prompt variant that omits the IR qualifier for a night image should be considered a known-bad configuration, not a legitimate ablation arm to report as competitive.
- Verify empirically, not just by design intent, that the modality qualifier actually suppresses color hallucination: decode a sample of night-IR images with and without the qualifier and check the output channel statistics — a correctly-behaving decode of a genuinely monochrome-source IR image should have near-zero chroma variance (R≈G≈B per pixel); measure this directly rather than assuming the prompt engineering worked.
- Since `illumination` (day-RGB vs night-IR) is available as ground-truth-quality metadata at the encoder (checkable directly from the image, per the plan's own zero-oracle-risk observation), there's no excuse for it to ever be missing from the conditioning signal — treat a night image being encoded/decoded without an explicit IR flag as a pipeline bug, not an edge case.
- Report reconstruction quality metrics **separately** for day-RGB and night-IR throughout (the plan already commits to this) — this pitfall is exactly the kind of failure that a pooled day+night metric would hide, since day images (where SD's color prior is appropriate) would pull the average up while night images silently fail.

**Warning signs:**
- Decoded night-IR images show visible color (non-gray) content when the source was genuinely monochrome — check by computing per-pixel channel variance (R vs G vs B) on decoded night images; near-zero variance is expected, non-trivial variance indicates color hallucination.
- The IR-qualifier prompt template is being A/B tested as one of several options with no clear expectation that it's mandatory — re-examine the prompt-template ablation design to ensure it isn't implicitly treating "no IR flag" as an acceptable baseline arm.
- Night-IR image bpp/quality metrics are being reported pooled with day-RGB metrics anywhere in the pipeline (even for a "quick check"), masking a night-specific failure.

**Phase to address:**
H3 domain-aware TGM phase (T5-equivalent) for the prompt-template fix, but the channel-variance verification check should be added to the eval harness during the foundational phase (T1.2-equivalent) so it's available to catch this even in the T2 baseline (pre-H3) decode, where the *original* Diff-ICMH TGM (using RAM++ tags, not yet domain-aware) is already likely to exhibit this failure mode and should be measured as a baseline data point.

---

### Pitfall 11: RAM++ tagging on IR images returns format descriptors ("black and white," "monochrome") instead of content tags

**What goes wrong:**
RAM++'s open-vocabulary tagging, when run on a night-IR image (which visually looks like a grayscale photograph regardless of its actual wildlife content), is prone to returning tags describing the *image's visual format* — "black and white," "monochrome," "night," "darkness" — rather than tags describing *what's actually depicted* (species, habitat, behavior). This directly undermines the TGM's purpose: the text conditioning fed to SD ends up describing the fact that the image is dark and gray, not that there's a leopard at a waterhole, which both wastes bit budget on low-entropy near-constant tags across every night image, and fails to give SD's generative prior the content signal it needs to reconstruct the actual scene well. The team's own plan diagnoses this exact failure mode already (§2.4) as the primary motivation for H3.

**Why it happens:**
RAM++ (and most vision-language tagging models) was trained predominantly on natural color photography; when confronted with an unusual visual domain (achromatic, high-contrast IR flash photography with a very different noise/texture signature than color photos), it falls back to describing the most visually salient, format-level properties it can confidently recognize (grayscale-ness, darkness) rather than the finer content it wasn't trained to reliably recognize in this modality. This isn't a bug in RAM++ so much as expected out-of-distribution behavior for a frozen, general-purpose tagger applied to a modality it wasn't specialized for. [LOW confidence — this specific behavior for RAM++ on IR camera-trap images specifically was not directly found in the literature search performed; it is inferred from (a) the team's own diagnosis in their plan, which is internally consistent with known VLM out-of-distribution behavior patterns, and (b) general knowledge of open-vocabulary taggers defaulting to salient low-level visual properties under distribution shift — treat this as a hypothesis to verify empirically in Week 1-2, not an established fact]

**How to avoid:**
- Empirically verify this failure mode directly and early: run RAM++ on a sample of ~50-100 night-IR images from the actual corpus in Week 1-2 and manually inspect the tag output — confirm (or refute) that format descriptors dominate over content tags before designing H3 around this assumption. If the hypothesis doesn't hold for this specific corpus, H3's motivating argument needs adjustment.
- Where content tags genuinely aren't reliably extractable from RAM++ on IR images, lean on the metadata-driven attributes the plan already correctly identifies as oracle-risk-free at the encoder (species from the bbox annotation's own ground-truth label when available for training-time conditioning experiments, habitat from site ID, season from EXIF timestamp) rather than depending on RAM++ content tags for night images specifically.
- Explicitly filter or down-weight format-descriptor tags ("black and white," "monochrome," "grayscale," "night," "darkness") from the L1 restricted vocabulary — since `illumination` is already captured as a dedicated structured L2 attribute, these tags are redundant *and* low-information when RAM++ emits them, and removing them frees bit budget for whatever content tags RAM++ *does* manage to extract.

**Warning signs:**
- Manual inspection of RAM++ tag output on a night-IR sample shows format-descriptor tags present in most/all images while content tags (species names, habitat descriptors) are largely absent or generic ("animal," "wildlife") compared to day-RGB tag output on the same corpus.
- Tag diversity (unique tag sets across different night images) is dramatically lower than tag diversity across day images of comparably varied content — near-identical tag sets across many different night images is the signature of the tagger falling back to format description.

**Phase to address:**
Should be empirically checked during the eval-harness/dataset-foundation phase (T1.2-equivalent), specifically because it's a stated premise underlying the H3 phase (T5) design — verifying it early (rather than assuming it and finding out in Week 5 that the premise doesn't quite hold for this corpus) protects H3's time budget.

---

## Google Colab Pro Operational Pitfalls

*(See also Pitfall 2 above for the compute-unit budget mis-scoping, which is the most damaging Colab-specific issue and is treated separately above given its severity.)*

### Pitfall 12: Session termination mid-run without resumability

**What goes wrong:**
Colab sessions (even Pro) can disconnect due to idle timeout, browser/network interruption, or the ~12h wall-clock ceiling regardless of remaining compute units, at any point including mid-iteration. A training run without robust, frequently-checkpointed resume support loses all progress since the last checkpoint — at the compute-unit budget this project actually has (Pitfall 2), losing even a few hours of a run is a meaningfully large fraction of total available compute, not a minor annoyance.

**How to avoid:**
- Checkpoint frequently relative to the *realistic* session length, not the theoretical 12h ceiling — assume disconnection can happen at any time and checkpoint at an interval (e.g., every 15-30 minutes or every N iterations) small enough that a worst-case loss doesn't meaningfully damage the CU budget.
- Verify resume actually restores full training state (model weights, optimizer state, LR scheduler state, iteration counter) by deliberately testing a kill-and-resume cycle in Week 2 (as part of the smoke-test), not assuming `train.py`'s `--resume_codec` flag works correctly for full-state resume until it's been proven — the plan's own audit already confirmed a resume flag exists in `train.py`, but existence of a flag is not the same as verified-correct full-state resume behavior.
- Store checkpoints to persistent storage (Drive) immediately, not just to the ephemeral Colab VM disk, so a disconnected session doesn't also lose the checkpoint itself.

**Warning signs:**
- After a resume, training loss at the resumed iteration count doesn't continue smoothly from where it left off (a visible discontinuity/spike) — this indicates optimizer state or LR schedule state wasn't correctly restored, even if the iteration counter itself looks right.
- A resumed run's logged iteration count doesn't match the checkpoint's — a classic silent-resume bug is restarting from iteration 0 while loading only the model weights, silently discarding optimizer momentum/schedule state (confirmed as a known category of PyTorch Lightning checkpoint-restore issue in the wider ecosystem — LR schedulers and callback state have documented history of lagging or resetting incorrectly on resume).
- GPU-hours or compute units consumed don't match the expected iteration-count progress — a sign that a run silently restarted from scratch one or more times without anyone noticing, burning CU budget on redundant work.

**Phase to address:**
Training infrastructure phase (T1.3/T2.3-equivalent) — the kill-and-resume test must be an explicit, verified pass/fail gate item before any long training run is trusted, not an assumption carried forward from "the flag exists."

---

### Pitfall 13: GPU tier assignment is non-deterministic — throughput measured on A100 does not predict L4 (or vice versa)

**What goes wrong:**
Colab does not guarantee which GPU tier (T4, L4, A100) a given session receives, even on Pro/Pro+ — assignment depends on availability and Google's internal allocation, and is documented (via user reports) to sometimes silently downgrade an A100 request to L4 with only a notification, not a hard block. If throughput (it/s, hours-per-run) is measured once on whatever tier happens to be assigned, then used to plan the rest of the project's schedule and compute-unit budget, every subsequent session that lands on a different (typically slower) tier invalidates that plan — runs take longer than budgeted, in both wall-clock and (since different tiers burn CU at different rates) compute-unit terms.

**How to avoid:**
- Log the actual assigned GPU (`nvidia-smi` output, or Colab's own reported GPU name) at the start of every session/run and record it in `results.jsonl` or an equivalent metadata log alongside throughput — never assume the tier from the previous session carries forward.
- Build the training/eval harness to auto-adapt (batch size, gradient accumulation steps) based on the detected GPU's VRAM at runtime, rather than hard-coding a config tuned for one specific tier — this also directly mitigates the OOM risk the team's own plan already flags (R2) as a real possibility if only ~24GB is assumed, when the actual assigned card in a given session might have less (T4-class) or more (A100) VRAM.
- Re-derive the CU-cost-per-run estimate (Pitfall 2) from the *actual* tier of the session about to run the job, not from a project-wide assumed-average tier — if a session comes up on A100, that's an opportunity to front-load the most VRAM/throughput-hungry planned work; if it comes up on a slower tier, defer that work rather than forcing it through inefficiently.

**Warning signs:**
- A run configured/tuned based on a previous session's A100 throughput numbers runs dramatically slower (2-4x) or OOMs on a subsequent session — check the assigned GPU first before assuming a regression in the code.
- `results.jsonl` (or training logs) don't record which GPU tier produced a given run's throughput/timing numbers — this makes it impossible to retroactively distinguish "the model got slower" from "the hardware got slower," and should be treated as a schema gap to fix immediately if found.

**Phase to address:**
Training infrastructure phase (T1.3/T2.3-equivalent) for the auto-adapting config and GPU-logging; ongoing operational discipline (log GPU tier every session) throughout every subsequent training/eval phase.

---

### Pitfall 14: Drive I/O bottleneck on many small files

**What goes wrong:**
A ~60K-image corpus, plus SAM-generated masks, plus per-checkpoint decoded outputs, stored as many individual small files on Google Drive (mounted into Colab) can produce severe I/O throughput bottlenecks — Drive's mount layer is not optimized for high-frequency small-file random access the way local SSD or a purpose-built dataset format (WebDataset, LMDB, tar shards) is. This can silently make data loading, not GPU compute, the actual bottleneck — meaning measured "throughput" during the Week 2 smoke test might reflect I/O-bound performance rather than true GPU-bound performance, further corrupting the compute-unit budget estimates in Pitfall 2 (a run that's I/O-bound burns wall-clock and CU without proportionally more training progress).

**How to avoid:**
- Package the training corpus (post-split, post-leakage-check) into a small number of large sharded archive files (e.g., WebDataset tar shards, or an LMDB/HDF5 store) rather than tens of thousands of loose image files on Drive, before any training run relies on it — this is best done once, during the data-foundation phase, and reused by every subsequent phase.
- Copy the active working subset to the Colab VM's local (ephemeral) disk at the start of a session rather than reading directly from the Drive mount for every training step, if the subset fits in local disk space.
- Measure and log data-loader throughput (images/sec delivered to the training loop) separately from GPU compute throughput during the Week 2 smoke test, so an I/O bottleneck is diagnosed as such rather than misattributed to "the GPU is slow" or "the model is expensive."

**Warning signs:**
- GPU utilization (`nvidia-smi` during a training step) is frequently well below 100% with no obvious compute-bound explanation — classic signature of a data-loading bottleneck.
- Measured throughput during the Week 2 smoke test is far below what the paper's Table 2 numbers (scaled for GPU tier and crop size) would predict, even after accounting for GPU tier differences.

**Phase to address:**
Data foundation phase (T1.1-equivalent) — decide the storage format before generating the final training corpus, not after Week 2's smoke test reveals a throughput problem.

---

## Evaluation Pitfalls

### Pitfall 15: Reporting differences within noise as real improvements

**What goes wrong:**
Given the small eval sets this compute budget likely forces (the plan's own T4.3 already proposes 400 images for fast decisions, and even the "final" numbers use only ~1,500 images), a metric delta between two configurations (e.g., H1 vs H1+H2, or one α value vs another) can easily be within the noise floor of eval-set sampling variance, random seed, or minor implementation differences — not a real effect of the mechanism being tested. Reporting such a difference as a finding (especially the central "H2 or H3 beats H1" claim the whole project's value proposition rests on) without quantifying uncertainty risks the project's central scientific claim being unfounded.

**How to avoid:**
- Compute bootstrap confidence intervals (resampling the eval image set with replacement, recomputing the metric, repeating hundreds of times) for every headline comparison, not just point estimates — this is standard practice for exactly this small-sample-size, few-runs-affordable scenario, and recent literature specifically recommends paired bootstrap protocols (matching per-image or per-seed deltas) precisely because small compute budgets can't afford many independent full runs to estimate variance the traditional way.
- When a competing configuration's point estimate falls inside another configuration's confidence interval, report the comparison as "not distinguishable given available eval set size" rather than declaring a winner — this is a more scientifically honest and, notably, exactly what the team's own plan already commits to in R7 ("báo cáo 'không phân biệt được' thay vì cường điệu").
- Increase eval-set size for any comparison that will become a headline claim in the report, even if it costs more compute — the plan's own two-tier design (400 images for fast iteration decisions, ~1,500 for final numbers) is a reasonable compromise, but the CI must still be computed and reported even on the larger set, not just the point estimate.
- Where possible, pair the comparison (same eval images, same random seed for stochastic decode) across the two configurations being compared, and bootstrap the *per-image paired difference* rather than bootstrapping each configuration's metric independently — paired bootstrap is substantially more statistically powerful for detecting small real effects against sampling noise.

**Warning signs:**
- A headline claim in the report ("H2 improves AP by X points") is stated without an accompanying confidence interval or significance measure.
- The point estimate difference between two configurations is smaller than what a quick back-of-envelope bootstrap CI (even a rough one) would produce as the CI half-width — if this is the case, that difference cannot be reported as a clear win.

**Phase to address:**
Should be built into the eval harness from the very first comparison (H1 vs baseline) in the foundational/H1 phase, and applied consistently through every later phase's headline comparisons (H2 ablation, H3 ablation, final RD curves) — retrofitting statistical rigor onto the final report in Week 8/10 after all runs are already "frozen" (per the plan's own results-freeze milestone) is too late to add more eval images if the CIs turn out too wide.

---

### Pitfall 16: Too few DDIM steps for headline numbers, or inconsistent DDIM step count between configurations

**What goes wrong:**
Diffusion decoding quality depends on the number of DDIM sampling steps used at inference; using fewer steps (for faster iteration during development, e.g., the plan's own suggestion of DDIM 20 steps for fast H2 α-selection decisions) produces measurably different, generally worse, reconstruction quality than the paper's reference 50-step setting. If this development-speed shortcut isn't clearly separated from the final numbers reported in the paper/report, or if different configurations being compared inadvertently use different step counts, quality/metric comparisons become confounded by an inference-time hyperparameter that has nothing to do with the actual hypothesis (H1/H2/H3) being tested.

**Why it happens:**
Under this project's compute budget, using fewer DDIM steps for fast iterative decisions (as the plan itself proposes for T4.3) is a reasonable, even necessary, tradeoff — but it creates a real risk that a "fast decision" step count silently becomes the step count used for a final reported number too, simply because re-running at full step count for every configuration that was fast-iterated on costs more compute than the schedule allows, or because someone forgets which script/config was used for a given number in `results.jsonl`.

**How to avoid:**
- Add `ddim_steps` as a mandatory field in the `results.jsonl` schema (the plan's own Phụ lục C checklist already flags this exact field as something to remember when setting up the schema) — this makes it possible to filter/audit whether any two numbers being compared actually used the same step count.
- Never compare numbers across different `ddim_steps` values as if they were equivalent — a fast-decision number (20 steps) can inform a go/no-go call during development, but must be re-measured at the final, consistent step count before being reported as a headline finding.
- Explicitly state the DDIM step count used for every reported number in the report/paper, and use a single consistent step count (the paper's reference 50, or an explicitly justified alternative) for every number that appears in a cross-configuration comparison table or RD curve.

**Warning signs:**
- Two numbers being compared in a table or figure have different `ddim_steps` values in `results.jsonl` — this should be a hard validation check on any figure-generation script (`make_all_figures.py`-equivalent) before it's allowed to plot a comparison.
- A "final" RD curve or ablation table includes numbers generated during the fast-iteration phase (e.g., using the T4.3 400-image/20-step protocol) without having been re-measured at the final protocol.

**Phase to address:**
Should be enforced structurally (schema validation) from the moment `results.jsonl`'s schema is defined in the data-foundation phase, and checked as a specific gate item before any figure/table generation for the final report.

---

### Pitfall 17: Dependency conflicts between wildlife task models corrupting each other's environments

**What goes wrong:**
The eval harness needs MegaDetector, a species classifier (SpeciesNet or similar), SAM, and possibly MegaDescriptor (re-ID ceiling experiment) — these come from different ecosystems/maintainers with their own PyTorch, torchvision, numpy, and CUDA version pins. Installing them into a single shared environment risks one package's install silently downgrading or upgrading a dependency another package needs, causing either an immediate crash or (worse) a silent numerical/behavioral change (e.g., a downgraded torchvision changing NMS behavior in a detector) that corrupts results without an obvious error message.

**Why it happens:**
This project deliberately spans multiple task-model ecosystems by design (detection + classification + segmentation + re-id, each best served by a different maintained package) on a compute-constrained setup where the team might be tempted to save setup time by sharing one environment across tools rather than isolating each. The team's own plan already correctly identifies this risk (R6, marked "Cao" probability) and commits to separate conda environments communicating via JSON as the mitigation — this is the right approach and should be preserved, not relaxed for convenience later in the project when time pressure increases.

**How to avoid:**
- Maintain fully separate environments per task model (as the plan already specifies) with pinned dependency versions (`environment.yml` per env, committed to the repo, not just "whatever installs today") — this is listed explicitly in the plan's own proposed repo structure (`envs/` with 5 environment files) and should not be collapsed into fewer environments even under Week 6 GPU-contention time pressure.
- Communicate between environments via a stable serialization format (JSON, as the plan specifies) rather than trying to import one tool's Python objects into another environment's process.
- Pin exact versions (not just "latest compatible") for each environment at the point they're first verified working, and treat any later re-install (e.g., after a Colab session reset wipes a non-persistent environment) as a risk that needs re-verification against a small fixed test case, not an assumption that "it'll install the same way again."
- Test load + a trivial inference call for each task-model environment immediately after setup (the plan's own Week 1 checklist already includes this) and re-verify after any Colab session that required reinstalling environments from scratch, since Colab VM resets (a new session almost always means these environments must be rebuilt, since they don't persist on the ephemeral VM disk).

**Warning signs:**
- A task model's output on a fixed, known test image changes between two sessions without any code change on the team's part — this indicates a silent dependency drift, not a data change.
- Environment install/setup for a task model produces "successfully installed" output but with visibly different resolved versions than a previous successful install for the same `environment.yml`.
- Colab environment setup consumes meaningfully more session time than expected because a conda solve is resolving conflicts or reinstalling large packages (e.g., PyTorch) repeatedly — this is both a correctness risk and a compute-unit-budget risk (Pitfall 2), since environment setup time on a Colab session burns wall-clock/CU without contributing to training/eval progress.

**Phase to address:**
Data/eval-harness foundation phase (T1.2-equivalent) — environment isolation must be designed and verified before the first cross-tool eval run, and the environment.yml files should be committed to the repo as source of truth from day one so re-creating an environment after a Colab session reset is a deterministic, one-command operation, not an ad hoc reinstall.

---

## Corrections to the Team's Own Risk Register (R0–R12)

The plan in `docs/ke-hoach-difficmh-wildlife-8-tuan.md` §4.3 is generally well-reasoned, but its probability/impact estimates and mitigation budgets were calculated against a **dedicated 24GB GPU running 24/7 for 8 weeks (1,344 GPU-hours)**, not the actual **Colab Pro ~100 compute-units/month (~50h L4-equivalent total)** constraint. This changes several risk assessments materially:

| Risk | Original assessment | Correction needed |
|---|---|---|
| **R0** (site/burst leakage) | Correctly identified as highest-impact ("huỷ toàn bộ giá trị 8 tuần"). | **Unchanged — this assessment is correct and should stay the #1 priority regardless of compute constraint.** See Pitfall 1 above for the full treatment. |
| **R1** (`train.py` is a stub) | Marked highest probability + highest impact, gating everything. | Per `PROJECT.md`'s own validated findings, this risk is **closed** — `train.py` is confirmed a working Lightning entrypoint with data module, resume support, and callbacks. This risk should be removed from the active register (already reflected correctly in PROJECT.md's Context section) — do not re-litigate it in the roadmap. |
| **R2** (OOM at 24GB) | Assumes a fixed 24GB ceiling. | **Needs correction**: the actual VRAM ceiling is *non-deterministic* per session (T4/L4/A100, see Pitfall 13) — mitigation must auto-adapt batch size/precision to whatever VRAM is detected at runtime, not assume a fixed 24GB target. A config tuned only for 24GB will OOM on a T4-class session. |
| **R3** (VAE ceiling too low) | Unaffected by compute reassessment. | Unchanged — this is an architectural/information-theoretic limit, not a compute-budget issue. |
| **R4** (H1 doesn't improve) | Unaffected by compute reassessment directly, but... | The diagnostic checklist ("check lr / crop animal-ratio / data / checkpoint loading in order") should have **learning-rate-driven catastrophic forgetting** (Pitfall 3) and **degenerate rate collapse** (Pitfall 4) added explicitly as checks, since these produce a similar symptom ("H1 doesn't improve") but require different diagnosis than the four causes originally listed. |
| **R5** (ROI paradox) | Correctly anticipated; treated as an expected finding, not a bug. | Unchanged and well-reasoned — but should be given a *dedicated, always-on metric* (empty-image false-positive rate) from the start of H2, not just "watched for" (see Pitfall 6). |
| **R6** (dependency conflicts) | Correctly marked high probability. | Unchanged — confirmed as a reasonable concern; add explicit pinned `environment.yml` + re-verification-after-session-reset discipline (Pitfall 17), since Colab's ephemeral VM makes this *recurring*, not a one-time Week-1 setup cost as the original plan implies. |
| **R7** (results within noise) | Correctly identified. | Unchanged — reinforce with bootstrap CI as standard practice from the first comparison onward, not just when a result "looks surprising" (Pitfall 15). |
| **R8** (scope creep to H4) | Correctly marked high probability/impact. | Unchanged, and now **easier to enforce**: given the ~5-6x smaller real compute budget, H4 (which needs ≥60 GPU-hours by the plan's own estimate) is not just "risky," it is **arithmetically close to impossible** within the real ~50h total project budget once H1/H2/H3 and eval overhead are accounted for. This should be stated as a hard scope boundary in the roadmap (already reflected correctly in PROJECT.md's Out of Scope section), not merely a soft "cut if time-constrained" item. |
| **R9** (report writing crunch) | Correctly identified with a good mitigation (write Method/Setup from mid-project). | Unchanged — this is a people/process risk, orthogonal to compute. |
| **R10** ("GPU contention in week 6," 85-105h needed vs 168h available) | **This entire risk is mis-scoped.** It assumes wall-clock contention between two people sharing one continuously-available GPU is the binding constraint. | **Replace with a compute-unit budget risk** (see Pitfall 2, which should be treated as the corrected, expanded version of R10): the binding constraint is not "do the two people's GPU-hour needs fit in a week's wall-clock," it's "does the *cumulative* compute-unit cost of the whole project's run plan fit in the monthly-resetting ~100 CU budget." The 85-105h figure itself should be recomputed in compute-unit terms against actual measured (not paper-assumed) throughput before the roadmap commits to it. |
| **R11** (disk space, ~236GB) | Assumes Google Drive 2TB, not a binding constraint. | Unchanged as a disk-space risk — Drive capacity is not the constraint per PROJECT.md's own Constraints section — but **add the I/O throughput issue** (Pitfall 14) as a related-but-distinct risk: even with ample *capacity*, many-small-files-on-Drive can bottleneck training *speed*, which is a compute-unit-budget risk, not a capacity risk. |
| **R12** (EXIF/metadata loss) | Correctly identified, correctly prioritized to Week 1. | Unchanged — good risk, good mitigation, keep as-is. |

**New risks not in the original R0-R12 register, surfaced by this research (see corresponding Pitfall sections above for full detail):**
- BD-rate uncomputable due to non-overlapping bpp ranges between fine-tuned and baseline models (Pitfall 5).
- Loss-term scale mismatch / degenerate rate collapse as training failure modes distinct from "doesn't improve" (Pitfall 4).
- SC-loss ROI-weighting resolution mismatch producing a silently-inert weighting term (Pitfall 7) — distinct from R5's paradox, this is a different failure mode (no effect at all, vs. a harmful effect).
- IR/night color hallucination from SD's missing infrared prior (Pitfall 10) — a physical-correctness issue, not covered by the original register's TGM-prompt risk framing.
- RAM++ format-descriptor tagging on IR images (Pitfall 11) as a hypothesis requiring empirical verification, not an assumed fact.
- Checkpoint-resume state-loss on Colab session termination (Pitfall 12) — distinct from R10's GPU-contention framing, this is about resumability correctness under *involuntary* session loss, which is common in Colab regardless of compute-unit availability.
- GPU-tier non-determinism corrupting throughput-based planning (Pitfall 13).
- DDIM-step-count inconsistency between fast-iteration and final numbers (Pitfall 16).

---

## "Looks Done But Isn't" Checklist

- [ ] **Split-leakage check:** `split_check.py` existing in the repo is not the same as it being *run and passing* as a gate before every training/eval job — verify it's actually wired into the pipeline, not just written.
- [ ] **Checkpoint resume:** a `--resume_codec` flag existing in `train.py` is not the same as verified full-state (optimizer + scheduler + iteration counter) resume — verify with an explicit kill-and-resume test, don't assume.
- [ ] **ROI mask reaching the loss:** a mask file existing and being loaded without error is not the same as the mask actually differentiating ROI from background in the loss computation — verify via separate `L_dist_roi`/`L_dist_bg` logging, not just "no crash."
- [ ] **BD-rate numbers:** an RD curve with points plotted is not the same as a *valid* BD-rate computation — verify the compared curves' bpp ranges actually overlap before trusting the interpolated number.
- [ ] **"Headline" metric improvements:** a point-estimate delta favoring the specialized model over baseline is not the same as a statistically supported finding — verify with a bootstrap CI before it goes in the report.
- [ ] **Night-IR handling:** a prompt template that mentions "infrared" is not the same as verified suppression of color hallucination — verify via channel-variance measurement on actual decoded output.
- [ ] **Environment reproducibility:** an `environment.yml` file existing in the repo is not the same as the environment being deterministically re-creatable after a Colab session reset — verify by actually tearing down and rebuilding at least once before relying on it during a time-constrained week.

---

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---|---|---|
| 1. Site/burst data leakage | Data Foundation (Week 1) | `split_check.py` runs as automated gate; CCT (new-site) eval doesn't collapse relative to in-domain val |
| 2. Colab compute-unit exhaustion | Roadmap/run-plan commitment (before Week 3) | Every planned run has a CU estimate from *measured* throughput; running CU total tracked against monthly ceiling |
| 3. Catastrophic forgetting / LR-driven prior destruction | H1 fine-tuning (Week 3) | OOD sanity-set (Kodak/COCO) PSNR tracked at every checkpoint; no early spike in loss terms |
| 4. Degenerate rate collapse | H1 fine-tuning + any loss-touching phase (H2) | Per-term loss logging; bpp sanity bounds per λ_rate |
| 5. BD-rate uncomputable (non-overlapping bpp) | H1 fine-tuning (first RD point) → multi-bitrate phase | bpp range overlap checked incrementally, not just at final RD-curve assembly |
| 6. ROI background-starvation paradox | H2 ROI-weighted loss (Week 4) | Empty-image false-positive rate tracked at every α; AP_small/medium/large reported, not just aggregate mAP |
| 7. SC-loss resolution mismatch | H2 ROI-weighted loss, before implementing SC-loss variant | ROI mask "hot cell" count distribution checked at chosen tap-point resolution before trusting null/positive result |
| 8. Mask/crop misalignment | H2 ROI-weighted loss (day one of implementation) | `L_dist_roi`/`L_dist_bg` logged separately; periodic mask-overlay visual dumps |
| 9. Generative hallucination (scientific integrity) | Eval-harness foundation (Week 1) → cross-analysis (Week 7) | Hallucination-rate-on-empty-images metric present from first baseline run; substantive Limitations section drafted mid-project |
| 10. IR color hallucination | H3 domain-aware TGM (Week 5); verification from Week 1 | Channel-variance check on decoded night-IR output; mandatory (not optional) IR qualifier in prompts |
| 11. RAM++ format-descriptor tags on IR | Eval-harness foundation (Week 1-2) | Manual tag-output inspection on ~50-100 night-IR images before designing H3 around the assumption |
| 12. Session termination / resume state loss | Training infrastructure (Week 1-2) | Explicit kill-and-resume test with loss-curve continuity check |
| 13. GPU tier non-determinism | Training infrastructure (Week 1-2); ongoing | GPU tier logged every session; config auto-adapts to detected VRAM |
| 14. Drive I/O bottleneck | Data Foundation (Week 1) | Sharded storage format decided before final corpus generation; data-loader throughput measured separately from GPU throughput |
| 15. Reporting noise as improvement | Eval-harness foundation (Week 1) → every headline comparison | Bootstrap CI computed for every reported delta; "not distinguishable" reported when CIs overlap |
| 16. DDIM step-count inconsistency | `results.jsonl` schema definition (Week 1) → figure generation | `ddim_steps` mandatory field; validation check blocks cross-step-count comparisons in figures |
| 17. Task-model dependency conflicts | Eval-harness foundation (Week 1) | Pinned `environment.yml` per tool; re-verified after every Colab session reset |

---

## Sources

- `D:/Diff_ICMH/docs/ke-hoach-difficmh-wildlife-8-tuan.md` — team's own 8-week plan, §0-5 and Appendices A-C, including the R0-R12 risk register (§4.3) that this document audits and corrects. [Primary source, HIGH confidence for what the plan itself states]
- `D:/Diff_ICMH/.planning/PROJECT.md` — validated project context, corrected compute-budget framing (Colab Pro ~100 CU/month vs. the plan's dedicated-24GB-GPU assumption). [Primary source, HIGH confidence]
- Camera-trap site/location-disjoint splitting convention and burst/sequence leakage — corroborated across multiple camera-trap ML papers found via web search (iWildCam-style location-disjoint splits; burst-sequence temporal-leakage discussion). [MEDIUM confidence, general web search, not a single authoritative citation]
- LILA BC / COCO Camera Traps metadata schema (`location`, `seq_id`, `sub_location`, `datetime`, ~3-second pseudo-sequence reconstruction when `seq_id` is absent) — from LILA BC dataset pages and related documentation found via web search. [MEDIUM confidence]
- Diffusion/deep-learning fine-tuning catastrophic forgetting and learning-rate scaling (`per-step forgetting ∝ LR × √loss`) — general fine-tuning literature (arXiv papers on loss-adaptive learning rates and catastrophic-forgetting mitigation), not codec-specific. [LOW-MEDIUM confidence, general fine-tuning literature applied to this project's specific codec context by inference]
- BD-rate non-overlapping-range evaluation practice — learned image compression literature (general practice of evaluating on intersecting sub-ranges when full-range comparison isn't possible). [MEDIUM confidence]
- ROI-weighted compression literature (background quality degradation under ROI weighting) — general learned-compression ROI papers found via web search; the specific link to detector false-positive increase is this project's own extension/hypothesis, consistent with but not directly confirmed by the surveyed literature, and independently corroborated by the team's own plan (R5) citing the Diff-ICMH paper's own Figure 3. [LOW-MEDIUM confidence for the FP-increase mechanism specifically]
- Generative/diffusion compression hallucination risk and its explicit flagging as unsuitable for strict-fidelity domains (medical/forensic) without safeguards — general generative-compression literature found via web search. [MEDIUM confidence for the general claim; LOW confidence for the direct camera-trap/ecological-data analogy, which is this document's own extension]
- Stable Diffusion's limited/absent infrared-modality prior — IR-image-generation/colorization literature explicitly discussing SD struggling with the infrared modality without specialized adaptation. [MEDIUM confidence]
- RAM++ format-descriptor tagging on IR images specifically — **not directly confirmed** in literature search performed; this is carried forward from the team's own plan's diagnosis (§2.4) as a hypothesis requiring empirical verification, not an established fact. [LOW confidence — flagged explicitly for early empirical verification]
- Google Colab Pro compute-unit costs, GPU-tier non-determinism (A100 request silently downgraded to L4), and session-limit behavior — Colab community/GitHub issue reports (`googlecolab/colabtools` issues) and third-party Colab-pricing comparison sources found via web search. [MEDIUM confidence — corroborated by multiple independent GitHub issue reports of the same A100→L4 downgrade behavior]
- PyTorch Lightning checkpoint-resume state-loss issues (optimizer/scheduler/callback state not fully restored) — `Lightning-AI/pytorch-lightning` GitHub issue tracker, multiple independent reported issues across versions. [MEDIUM confidence — this is the general PyTorch Lightning ecosystem's documented issue history, not confirmation that this specific project's `train.py` has the same bug; treat as a reason to test, not as proof of an existing bug]
- Bootstrap confidence intervals for small-sample ML evaluation, paired bootstrap protocols for detecting small real effects — recent (2026) arXiv methodology papers on evaluating small improvements under constrained compute budgets, found via web search. [MEDIUM confidence]
- MegaDetector / PyTorch-Wildlife installation and environment structure — official Microsoft documentation found via web search; specific dependency-conflict-with-SAM evidence was not directly found and is carried forward as the team's own already-correct R6 assessment rather than newly substantiated by this search. [LOW confidence for the specific conflict claim, MEDIUM for the general multi-ecosystem-environment-isolation best practice]

---
*Pitfalls research for: Wild-Diff-ICMH — domain-specialized generative image compression for camera-trap wildlife imagery*
*Researched: 2026-09-07*
