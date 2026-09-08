# Project Research Summary

**Project:** Wild-Diff-ICMH
**Domain:** Domain-specialized generative (diffusion) image compression for camera-trap wildlife imagery — academic research deliverable (ICMH: image coding for machines and humans)
**Researched:** 2026-09-07
**Confidence:** MEDIUM-HIGH

## Executive Summary

This project fine-tunes an existing, working Diff-ICMH/DiffEIC codebase (SD 2.1-prior generative codec) onto camera-trap wildlife imagery, then layers two specialization mechanisms — ROI-weighted loss (H2) and a domain-aware Tag Guidance Module (H3) — on top of plain domain fine-tuning (H1), with the goal of proving the specialization mechanisms contribute beyond fine-tuning alone. All four research passes converge on one dominant fact: **the real compute budget is ~50 GPU-hours total (Colab Pro, ~100 compute units/month ≈ 20h L4/month), not the ~1,344 GPU-hours the source 8-week plan assumed — a 5-6x gap.** Every decision about run count, bitrate points, eval-set size, DDIM steps, and even which loss injection point to use must be re-derived from this real number, not the plan's original arithmetic. Multiple researchers independently flagged this as the single fact most likely to sink the project if not corrected before phase/run planning locks in.

The recommended approach is: fix a short, concrete list of pre-flight code/dependency issues (numpy/pytorch_lightning/lpips version breaks, the Lightning 1.x to 2.x accelerator config break, a crash-unsafe checkpoint-resume path that silently discards optimizer state, checkpoint bloat from saving frozen SD weights, and a dataset/mask crop-misalignment trap) before any GPU spend; then exploit the project's biggest budget lever — H3's L1/L2 tiers (restricted tag vocabulary + real sensor metadata as conditioning) require zero training, since they only change the decode-time text prompt on an already-trained checkpoint — to fill out ablation cells and extra bitrate/config points that would otherwise cost dedicated runs. Architecture research confirms the codebase is not a stub (a risk the source plan over-weighted) and pinpoints exact line numbers for every injection point (H2's loss reweighting, H3's tag pipeline, the checkpoint/resume gap). Feature research confirms the report's credibility floor (RD curves, additive ablation table with bootstrap CI, cost-of-specialization table on both general-domain and held-out-site data, day/night split reporting, hallucination-on-empty-images metric) and flags that the closest prior art (Disney/ETH's region-adaptive diffusion codec, arXiv:2604.01122) must be cited to keep the ROI-weighting novelty claim defensible and narrowly scoped.

The two risks that can end the project outright are (1) site/burst data leakage across train/val/test, which would silently invalidate every downstream number, and (2) compute-unit exhaustion mid-project from planning against the wrong budget. Both require hard gates (automated assertions, budget checkpoints tied to measured, not assumed, throughput), not just documented awareness. Every other finding in this research pass is secondary to getting these two gates, plus the pre-flight code fixes, right before Phase 1 GPU work begins.

## Key Findings

### Recommended Stack

The repo's requirements.txt will not clean-install on current Colab (Python 3.12, torch ~2.11) as pinned — independent of any architectural question. Confirmed breaks: numpy==1.23.1 (no Python 3.12 wheel), pytorch_lightning==1.5.0 (predates Python 3.12, and is a frozen/deprecated package alias — must move to the unified lightning>=2.6 package), lpips==0.1.4 (hard-errors on current torchvision's removed pretrained=True kwarg), and xformers==0.0.22 (wrong ABI for torch 2.x). None of these are architectural risks — they are a ~1-2 hour dependency-upgrade pass that must land before the Week-1/Week-2 smoke test.

**Core technologies:**
- **PyTorch / Python**: whatever Colab currently preinstalls (verify live each session, do not pin) — fighting Colab's runtime causes CUDA/driver mismatches
- **lightning (unified package) >=2.6**, imported as lightning.pytorch — replaces the dead pytorch_lightning==1.5.0 pin; also requires deleting a dead LightningCLI import in train.py and rewriting accelerator: ddp to accelerator: gpu, devices: 1 (no distributed strategy needed — Colab gives one GPU per session)
- **xformers** — reinstall from the torch-matched index at the start of every session (Colab's torch version can shift on quarterly runtime rollouts)
- **CompressAI 1.2.8**, **pyiqa 0.1.15.post2** (replaces standalone lpips+DISTS — single actively-maintained metrics package), **bjontegaard 1.3.0** (canonical BD-rate), **pycocotools 2.0.11**
- **MegaDetector V6** (via PytorchWildlife), **SpeciesNet 5.0.5**, **SAM 2.1** (recommended upgrade over the plan's SAM ViT-H — faster, higher mIoU, same Apache-2.0 license), **MegaDescriptor** (via wildlife-tools, used only to measure the re-ID Nyquist-limit claim, not as a product feature)
- **3 isolated conda/pip environments** (detect / species / segment) — confirmed real dependency conflicts between MegaDetector's and SpeciesNet's ultralytics/yolov5 pins; communicate via JSON on disk only, never shared imports

**Colab Pro compute reality (community-measured, MEDIUM confidence, re-verify Week 1):** L4 approx 1.71 CU/hr (~58.5h from 100 CU) is the best tier for this project (24GB VRAM, matches the paper's assumed card); A100 40GB approx 5.40 CU/hr (~18.5h) burns budget ~3.16x faster per hour — reserve for late VRAM-bound runs only. This confirms (rather than contradicts) PROJECT.md's own "~50h L4" figure. Background execution (surviving a closed tab) is a Pro+-only feature — do not assume it. Session hard cap ~24h, idle timeout ~90 min (does not trigger during active training).

### Expected Features

**Must have (table stakes) — the credibility floor for the report:**
- Site-level (ideally sequence/burst-level) train/val/test split with automated leak assertion — before anything else
- RD curves at >=3-4 bitrate points (4 is the JVET/BD-rate convention; 3 + disclosed quadratic BD-rate fallback is acceptable if budget forces it) for Original/+H1/+full, scored on detection+species+segmentation
- Additive ablation table (Original -> +H1 -> +H1+H2 -> +full) **with bootstrap CI on every cell**, not just the H2 AP table the source plan originally scoped CI for
- Cost-of-specialization table on both axes: general-domain (Kodak/COCO) and held-out-site (CCT) — do not cut this under schedule pressure, it is one of two tables that jointly answer the project's Core Value question
- Day-RGB / night-IR split reporting on every headline metric (never pool — pooling hides exactly the effect H1 is trying to demonstrate)
- Empty-image false-positive rate AND hallucination-on-empty-images rate, across the full bitrate sweep — elevate from nice-to-have to required numbered tables; both reuse the same MegaDetector inference pass already needed for mAP (cache raw confidence scores once, derive 3 metrics from it)
- Species-level AND coarser group/taxonomic-fallback accuracy (connects directly to the VAE Nyquist-limit finding)
- VTM/BPG anchor curve (contextualizing only, CPU-only, never competes with GPU-bound work for calendar time)
- Reproducibility package: pinned envs, results.jsonl + figure-generation scripts, split manifests, fine-tuned-delta-only checkpoints (not full re-hosted SD weights), model/dataset card, a minimal smoke test

**Should have (differentiators):**
- ROI-weighted L_dist/L_sem calibrated to wildlife's extreme foreground/background bit ratio — narrowly scoped novelty (specific architecture + spatial resolution + wildlife bit-redundancy argument + data/compute-scarce regime), explicitly distinguished from prior art (TLIC, and especially Disney/ETH's arXiv:2604.01122 spatially-varying diffusion codec, which must be cited)
- Domain-restricted tag vocabulary (L1) + real (non-predicted) sensor metadata as text-conditioning side-information (L2) — near-zero bit overhead, near-zero training cost, and a genuinely clean no-oracle-risk argument since most fields (illumination, season, habitat) are literal sensor metadata rather than predicted attributes

**Defer (v2+):** second neural-codec baseline (TransTIC/ELIC), FID at larger sample size, H3-L3 (spatial grid + count conditioning — the one TGM tier needing control-module retraining), additional lambda_rate points beyond the minimum.

**Explicit anti-features (do not attempt):** claiming individual re-ID capability (physically blocked by VAE downsampling below Nyquist for stripe/spot patterns — prove the ceiling quantitatively instead), claiming to beat VTM on PSNR (structurally impossible for a generative codec, contradicts the project's own framing), cherry-picked qualitative figures without failure cases, treating SAM pseudo-GT masks as true segmentation ground truth.

### Architecture Approach

The codebase is a working PyTorch Lightning pipeline, not a stub (the source plan's biggest feared risk is closed). The two loss injection points for H2 are precisely located: loss_guide (L_dist, model/diffeic.py:973-977, operates on the 8x-downsampled VAE latent) and loss_semantic (L_sem, lines 980-1013, must move its tap point from the default middle-block sl_loc: mid at 64x downsample — where a mid-sized animal degenerates to under 1 grid cell, making ROI-weighting meaningless — to sl_loc: enc_9 at 32x downsample). Both are 1-line changes (swap uniform mean for a weighted mean), require no new trainable parameters, and no VRAM increase. H3's L1/L2 tiers plug in as a wrapper around TagGCM.extract_tag() (model/lfgcm.py) with zero changes to RAM++ itself and zero training required, since they only change the CLIP-encoded text conditioning at decode time.

**Major components (new work, layered onto the existing pipeline):**
1. src/data/ (acquire/normalize/split + split_check.py hard assertion) — the true root dependency of the whole project; every other component depends on this being correct first
2. dataset/wildlife_licdataset.py — new mask-aware dataset class; owns the critical fix for the crop/mask alignment trap (existing random_crop_arr makes independent internal random choices per call — calling it separately for image and mask silently produces misaligned pairs; fix by stacking image+mask into one array before cropping)
3. src/losses/roi_loss.py — pure, GPU-independent weighted-loss helper functions, unit-testable without a running training job
4. src/eval/eval_harness.py + results.jsonl — single source of truth; wraps MegaDetector/SpeciesNet/SAM/MegaDescriptor inference, appends metric rows, feeds make_all_figures.py

### Critical Pitfalls

1. **Site/burst data leakage across splits** — the highest-impact, most-likely silent-catastrophe risk. A random or image-level split lets the model memorize stationary-camera backgrounds, inflating every downstream metric. Mitigation: split at site AND sequence/burst granularity, gate with an automated split_check.py assertion that runs before any training/eval job reads split files, run this in Week 1 before any GPU spend is trusted.
2. **Colab compute-unit exhaustion from planning against the wrong budget** — the source plan's 7-run/5-bitrate-point/3-hypothesis schedule assumed ~290h needed against ~1,344h available; real budget is ~50-58h L4-equivalent for the entire project. Mitigation: reforecast every planned run in compute units (not GPU-hours) using measured, not assumed, throughput from the Week 2 smoke test; track running CU burn in results.jsonl; have a pre-committed sacrifice order (H4 first, then bitrate points beyond 2-3, then ablation cells) rather than discovering the wall in Week 5-6.
3. **Crash-unsafe checkpoint resume** — train.py's existing resume path is weights-only (load_state_dict, not trainer.fit(ckpt_path=...)); optimizer state, LR schedule position, and global_step silently reset on every restart. Given Colab sessions can die every few hours, every resume today is actually a lossy warm-start. Must patch before the first real (non-smoke-test) training run — this is a small code change (~0.5-1 day) but sits on the critical path for everything downstream.
4. **Checkpoint bloat from saving frozen SD/VAE/RAM++ weights** — DiffEIC does not override on_save_checkpoint to strip frozen submodules, so every checkpoint is multi-GB instead of a few hundred MB; combined with the default every_n_train_steps: 10000 (too coarse for ~2-4h Colab sessions) and save_top_k: -1 (keep-everything), this both wastes Drive I/O throughput and risks losing large chunks of compute on a crash. Fix both the stripping override and the checkpoint cadence (drop to every 500-1000 steps) in the same pass as the resume fix.
5. **ROI-weighting the wrong resolution / background starvation paradox (H2-specific)** — applying ROI-weighting to L_sem at the default middle-block tap point is nearly meaningless at this domain's object sizes (already covered above); separately, aggressively prioritizing the animal region at high alpha can starve background bits enough that the generative prior fills it with plausible-but-wrong texture, which can increase detector false-positive rates even as animal-region quality improves. Mitigation: track empty-image false-positive rate as a first-class metric at every alpha from the start of H2, don't assume higher alpha is strictly better.

## Implications for Roadmap

Based on combined research, suggested phase structure:

### Phase 1: Pre-flight fixes + data foundation
**Rationale:** Everything else is blocked on (a) the codebase actually running on Colab's real environment and (b) a leak-proof split existing. Both are cheap, both are catastrophic if skipped, and both were independently flagged by 3 of the 4 researchers as belonging before any GPU spend.
**Delivers:** Dependency upgrades (numpy/lightning/lpips/xformers), train.py Lightning 2.x config fixes, ckpt_path crash-safe resume, checkpoint-stripping on_save/on_load_checkpoint override, checkpoint cadence fix; LILA BC data download (bbox-annotated Snapshot Serengeti subset + CCT held-out), site/sequence-level split with split_check.py gate, SAM 2.1 mask generation (pseudo-GT val + ROI train masks).
**Addresses:** Table-stakes reproducibility requirements; the site-level-split table-stakes item from FEATURES.md.
**Avoids:** Pitfalls 1 (data leakage), 3 (crash-unsafe resume), 4 (checkpoint bloat) above.

### Phase 2: Training/eval infrastructure + baseline
**Rationale:** Before any real fine-tuning run, the team needs (a) a measured (not assumed) throughput/CU-burn number to calibrate every later budget decision, and (b) the eval harness + results.jsonl schema in place so the unmodified pretrained baseline (the load-bearing counterfactual for the whole ablation table) can be scored immediately.
**Delivers:** WildlifeLICDataset (with the crop/mask-alignment fix), session-start Drive-shard-copy script, smoke-test training run (~2K iters) with real CU-burn measurement, eval_harness.py skeleton wired to MegaDetector/SpeciesNet/SAM/pyiqa, results.jsonl writer, unmodified-pretrained-checkpoint baseline eval (day/night split from day one).
**Uses:** Stack elements — pyiqa, CompressAI, bjontegaard, MegaDetector V6, SpeciesNet, SAM 2.1.
**Implements:** eval_harness.py + results.jsonl architecture component; the reforecast-in-compute-units step from Pitfall 2.
**Research flag:** Colab GPU-tier assignment is non-deterministic — the smoke test result here should drive a hard go/no-go recalibration of every subsequent phase's run-count budget, not just inform Phase 2 itself.

### Phase 3: H1 domain fine-tuning
**Rationale:** H1 is the prerequisite baseline that both H2 and H3-L3 build on top of, and it is itself the row-0-vs-row-1 comparison the core ablation table needs.
**Delivers:** Fine-tuned codec+control-module checkpoint on the wildlife domain; first real RD point(s); OOD sanity-check logging (Kodak/COCO decoded at fixed intervals to catch catastrophic-forgetting/LR-driven prior destruction early); per-loss-term logging (bpp/dist/diff/sem separately, to catch rate collapse).
**Avoids:** Catastrophic forgetting from too-high LR; degenerate rate collapse; BD-rate range non-overlap with baseline (check incrementally, not just at the end).

### Phase 4: H2 ROI-weighted loss
**Rationale:** Builds directly on H1's checkpoint; the architecture research pinpoints this as a 1-2 line code change with no VRAM cost, so the phase's compute cost is dominated by the alpha-sweep runs, not implementation effort.
**Delivers:** roi_loss.py weighted-mean helpers; L_dist reweighting at VAE-latent resolution; L_sem reweighting moved to sl_loc: enc_9 (32x); mask-overlay visual debug dumps and split L_dist_roi/L_dist_bg logging (mandatory from day one, not retrofitted); empty-image false-positive-rate tracking across the alpha sweep.
**Avoids:** Resolution mismatch, the background-starvation paradox, and the mask/crop misalignment silent-failure bug class.

### Phase 5: H3 domain-aware TGM (L1/L2, then optionally L3)
**Rationale:** L1/L2 need zero training and can be validated against the pretrained checkpoint immediately — this phase can start in parallel with Phase 3/4's GPU-bound work, not after it. This is the single biggest budget lever in the project; the roadmap should schedule it to run concurrently with H1/H2 GPU time, not sequentially after.
**Delivers:** Curated ~256-tag wildlife vocabulary (L1), structured metadata conditioning (illumination/habitat/season, L2) wired into TagGCM.extract_tag, decode-time prompt-swap experiments layered onto H1/H2 checkpoints for near-free ablation-table cells; IR-modality color-hallucination check (verify near-zero chroma variance on night-IR decodes); RAM++-on-IR tag-quality empirical check.
**Research flag:** Empirically verify (Week 1-2, cheap, CPU-only) whether RAM++ actually degrades to format-descriptor tags on IR images before committing H3's motivating argument to the report — this is a stated but unverified premise.

### Phase 6: Ablation, cost-of-specialization, RD/BD-rate, and report artifacts
**Rationale:** These are synthesis deliverables that read from results.jsonl only — no new GPU work beyond filling any remaining ablation-table cells (exploiting H3-L1/L2's zero-training property to close gaps cheaply) and the final bitrate-point runs.
**Delivers:** Additive ablation table with bootstrap CI on every cell; cost-of-specialization table (Kodak/COCO + CCT); RD curves + BD-rate/BD-accuracy (bjontegaard); hallucination-on-empty-images and species/group accuracy tables; VTM anchor curve (scheduled opportunistically on CPU throughout, not blocking this phase); Limitations section; reproducibility package; make_all_figures.py.
**Avoids:** Reporting noise as signal (bootstrap CI mandatory on every headline number, paired where possible).

### Phase Ordering Rationale

- Pre-flight fixes and the site-level split gate are strict prerequisites for every other phase — corrupting either one invalidates all downstream numbers, which is why architecture and pitfalls research both independently place them at the very front.
- H3-L1/L2's zero-training property means it should be scheduled to run in parallel with H1/H2's GPU-bound phases, not treated as a sequential later phase — the roadmap should explicitly note this as a scheduling opportunity, not just a feature note.
- eval_harness.py and make_all_figures.py skeletons should be built early (against the baseline/sparse data) rather than at the end, so they are never last-week work and so async per-checkpoint eval (via a lightweight STATUS.json poll, not manual handoff) is possible from Phase 2 onward.
- Compute-unit budget recalibration is not a one-time Phase 1 event — it must recur at every phase boundary, since GPU tier assignment is non-deterministic session to session and can silently invalidate a schedule calibrated on one session's throughput.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 4 (H2):** the exact alpha range to sweep and whether V2 (L_dist+L_sem) meaningfully outperforms V1 (L_dist-only) is an open empirical question — the resolution-mismatch pitfall means a null result here could be a real finding or a masking bug, and needs a dedicated verification step (mask-coverage sanity check) before being reported either way.
- **Phase 5 (H3):** the RAM++-on-IR tag degradation premise needs empirical verification before the phase's design is finalized; also flagged as MEDIUM confidence that the L1+L2 novelty combination (restricted vocab + real sensor metadata + diffusion codec) is unclaimed prior art — recommend one targeted literature re-check close to the report-writing phase, not now.
- **Phase 6:** BD-rate computability depends on the fine-tuned and baseline bpp ranges overlapping — this is a per-phase-4/5 incremental check, not something to discover only when assembling Phase 6's final tables.

Phases with standard patterns (skip deep research-phase):
- **Phase 1:** the dependency-upgrade fixes and split-leakage assertion are fully specified already (exact package versions, exact line numbers) by this research pass — implementation, not research, is needed.
- **Phase 2:** the eval harness wiring (MegaDetector/SpeciesNet/SAM/pyiqa) follows well-documented install paths already verified against PyPI/GitHub directly.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | MEDIUM-HIGH | Library versions verified directly against PyPI/GitHub (HIGH); Colab compute-unit rates are community-measured, not officially published by Google (MEDIUM) — must be re-measured empirically in Week 1 |
| Features | MEDIUM-HIGH | Grounded in established ICM/ICMH and camera-trap-ML community conventions (HIGH); a few novelty claims for H2/H3 are MEDIUM — absence of found prior art is not proof of absence, especially for the H3 L1+L2 combination |
| Architecture | HIGH for codebase claims (verified by reading actual files line-by-line); MEDIUM for the Colab Drive I/O throughput quantification (web-sourced) |
| Pitfalls | MEDIUM overall; HIGH for the two most damaging pitfalls (data leakage, compute-unit exhaustion), corroborated by both the team's own risk register and standard literature; LOW-MEDIUM for narrower claims (e.g., RAM++'s specific IR-tagging failure mode is an inferred hypothesis, not directly confirmed in literature — flagged for Week 1-2 empirical verification) |

**Overall confidence:** MEDIUM-HIGH

### Gaps to Address

- **RAM++-on-IR tag degradation** (LOW confidence, inferred not confirmed) — verify empirically in Week 1-2 by running RAM++ on ~50-100 night-IR sample images before H3's design commits to this premise.
- **EXIF datetime survival in LILA's redistributed JPEGs** (MEDIUM confidence, not explicitly documented by LILA) — verify on the first 100 downloaded images before designing H3's L2 layer around EXIF; JSON datetime/location fields are a confirmed-present fallback with zero redesign cost if EXIF is stripped.
- **Actual Colab compute-unit burn rate for this project's specific workload** (MEDIUM confidence, community-measured, fluctuates by region/demand) — must be measured directly in the Week 2 smoke test and used to recalibrate the entire remaining schedule, not assumed from the community figures cited here.
- **H3 L1+L2 novelty scoping** (MEDIUM confidence) — recommend one targeted literature re-check on metadata-conditioned generative compression close to the report-writing phase before finalizing the novelty claim in the report.
- **Wellington/Idaho/Missouri camera-trap datasets' bbox coverage** (not confirmed in this pass) — not required for MVP scope (Snapshot Serengeti + CCT suffice), but flagged as unverified if a third domain is later considered for T7-style generalization analysis.

## Sources

### Primary (HIGH confidence)
- Repo-internal, read directly: requirements.txt, train.py, configs/train_diffeic.yaml, configs/model/diffeic.yaml, model/diffeic.py, model/lfgcm.py, dataset/licdataset.py, dataset/data_module.py, utils/image/common.py, inference.py
- .planning/PROJECT.md — Core Value, Requirements, Constraints, Key Decisions, GPU budget derivation
- docs/ke-hoach-difficmh-wildlife-8-tuan.md — original 8-week plan, risk register R0-R12, acceptance criteria
- LILA BC dataset pages (https://lila.science/datasets/) — dataset sizes, licenses, download mechanics
- InterDigitalInc/CompressAI (https://github.com/InterDigitalInc/CompressAI), PyPI (https://pypi.org/project/compressai/) — v1.2.8
- Lightning-AI GitHub discussion #17095 (https://github.com/Lightning-AI/pytorch-lightning/discussions/17095) — package naming/maintenance
- NeurIPS Paper Checklist Guidelines (https://neurips.cc/public/guides/PaperChecklist)
- Region-Adaptive Generative Compression with Spatially Varying Diffusion Models (arXiv:2604.01122) (https://arxiv.org/abs/2604.01122) — closest prior art to H2, must be cited

### Secondary (MEDIUM confidence)
- Chris McCormick — Colab GPUs Features and Pricing (http://mccormickml.com/2024/04/23/colab-gpus-features-and-pricing/) — community-measured CU/hr rates
- Working with huge datasets in Google Colab and Google Drive (https://satyajitghana.medium.com/working-with-huge-datasets-800k-files-in-google-colab-and-google-drive-bcb175c79477) — Drive I/O ~300x slowdown on many-small-file random access
- agentmorris camera-trap-ml-survey (https://agentmorris.github.io/camera-trap-ml-survey/) — site-level split conventions, empty-image handling
- When the Codec Hallucinates (CHI 2026) (https://doi.org/10.1145/3772318.3790293) — generative-compression hallucination as a named data-integrity failure mode

### Tertiary (LOW confidence)
- RAM++ IR-tagging format-descriptor fallback behavior — inferred hypothesis, not directly confirmed in literature; verify empirically Week 1-2

---
*Research completed: 2026-09-07*
*Ready for roadmap: yes*
