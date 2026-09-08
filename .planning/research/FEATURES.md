# Feature Research: Domain-Specialized Generative Image Compression (Wild-Diff-ICMH)

**Domain:** Academic research deliverable — neural/generative image codec domain-specialization study (image coding for machines and humans, ICMH), applied to wildlife camera-trap imagery
**Researched:** 2026-09-07
**Confidence:** MEDIUM-HIGH (grounded in ICM/ICMH literature conventions, camera-trap ML community standards, and BD-rate/JVET common test conditions; a few novelty claims in Differentiators are MEDIUM confidence — absence of found prior art is not proof of absence)

**Reading note:** "Users" in the template below means **reviewers / the research community**, not end users. "Table stakes" = what makes the report *credible as research*, not what makes a product sell. Every item below carries a GPU-cost note against the project's real budget: **~50 GPU-hours TOTAL on Colab Pro (≈20h L4 or ≈7.5h A100/month × ~2.5 months), a 5–6× cut versus the original 8-week plan's ~290h assumption** (per `PROJECT.md` Context section). This is the single most important constraint shaping every recommendation below — cheap-credible-version guidance is not optional, it is the difference between finishing and not finishing.

---

## Feature Landscape

### Table Stakes — General ICM/ICMH Paper Conventions (Q1)

These are non-negotiable for the report to read as a real compression paper, independent of the domain-specialization angle.

| Feature | Why Expected | Complexity / GPU Cost | Notes |
|---------|--------------|------------------------|-------|
| **RD curves, ≥4 bitrate points per configuration** | JVET/VTM common test conditions use exactly 4 QP points (22/27/32/37) as the field-standard minimum for a defensible RD curve; the Bjøntegaard piecewise-cubic BD-rate fit is defined for 4 points and becomes numerically unstable/requires a lower-order (quadratic) fallback below that. Diff-ICMH's own paper sweeps `λ_rate ∈ {2,4,8,16,32}` (5 points). Reviewers of compression papers reject 2–3-point curves as "not a curve." | **HIGH — this is the single largest GPU line item.** Each point that requires its own fine-tune run costs the same as a full H1 training run. | **Cheapest credible version:** target exactly 4 λ_rate points, not 5. Fully fine-tune only 2 of them (e.g. λ=8 mid, λ=32 low-rate) as real training runs; obtain the other 2 via short/low-iteration fine-tunes (2–4K iters instead of 20–30K) rather than full runs, and disclose the asymmetric training budget per point in the report. If even that doesn't fit, 3 points + quadratic BD-rate is an acceptable, explicitly-disclosed fallback — do not silently report BD-rate as if 4 points were used. |
| **Standard metric set: PSNR, MS-SSIM, LPIPS, DISTS (+ FID where feasible)** | This is the metric set every generative/perceptual codec paper since HiFiC (Mentzer et al. 2020) and CDC (Yang & Mandt 2023) reports: two full-reference distortion metrics (PSNR, MS-SSIM) plus two learned perceptual metrics (LPIPS, DISTS) to capture the perception-distortion tradeoff that generative codecs are explicitly trading against. | LOW — no extra GPU, these are all post-hoc metrics computed on already-decoded images (CPU/light-GPU, minutes not hours). | FID is the outlier: it needs a reasonably large, stable sample count to not be noisy (field convention is inconsistent — Kodak papers report FID on only 24 images as a "soft convention" despite known instability). **Cheapest credible version:** report LPIPS/DISTS as primary perceptual metrics (stable at small N, reference-based); report FID only as a secondary/exploratory number computed on the largest feasible eval subset, with an explicit small-sample caveat in Limitations rather than silently presenting it as authoritative. |
| **Machine-task metrics per the ICM/ICMH protocol: mAP (AP_s/m/l split), classification top-1 (+group-level fallback), segmentation mIoU** | This is literally the protocol Diff-ICMH itself uses (frozen, off-the-shelf task models run on decoded images, no task-specific fine-tuning — that IS the paper's "without any task-specific adaptation" claim). A follow-up paper that changes this protocol would be changing the thing being tested. | LOW-MEDIUM — task-model inference is cheap relative to codec training/decoding (MegaDetector/SpeciesNet/SAM inference is seconds/image on GPU, and can run on CPU in a pinch). | Already correctly scoped in `PROJECT.md` Active requirements. **Reuse note:** MegaDetector inference needed here is the *same* inference needed for the empty-image hallucination-rate metric (Anti-Features section) — batch them together, don't run MegaDetector twice. |
| **BD-rate AND BD-accuracy, dual-axis** | ICM/ICMH convention (confirmed in current literature, e.g. FR-ICMH/PR-ICMH BD-rate reporting) is to report BD-rate on the human/distortion axis (bpp vs PSNR or MS-SSIM) *and* BD-accuracy on the machine axis (bpp vs mAP/top-1/mIoU) — a single BD-rate number on PSNR alone would be judged as reporting only half the ICMH story. | LOW — pure post-processing on `results.jsonl`, no GPU. | Directly compatible with the project's "results.jsonl is the single source of truth, all figures generated" decision — good, keep it. |
| **Mandatory baseline: unmodified pretrained Diff-ICMH checkpoint, evaluated on the wildlife domain with zero fine-tuning** | This is the **load-bearing baseline** for the entire project — without it, "we specialized a codec" has no counterfactual and the core reviewer question ("what's different from just fine-tuning?") cannot even be posed, let alone answered. | LOW — decode-only, no training. This should be the *first* GPU spend of the project (it already is, per plan T2.1). | Already correctly identified as `results.jsonl` baseline row and Active requirement. |
| **Mandatory baseline: traditional codec anchor (VTM or at minimum BPG/JPEG)** | Even papers that explicitly disclaim beating VTM on PSNR (correctly, per the project's own decision) still plot a VTM curve — it contextualizes *how much* is being traded away, and reviewers expect to see the gap quantified, not asserted. Absence of any traditional-codec curve reads as avoidance. | LOW-MEDIUM — VTM encoding is CPU-only (slow per-image but no GPU needed); can run in background on CPU while GPU is busy with training. | **Cheapest credible version:** VTM curve on a *small* fixed eval subset (e.g. the same 150–300 images used for other decision-making evals) is sufficient — it is a contextualizing anchor, not a competitive comparison, so it does not need the full eval set or multiple bitrate points beyond what's needed to place it relative to the generative-codec curve. |
| **Optional-but-expected baseline: a second neural/ICM codec (e.g. TransTIC, ELIC)** | Reviewers *prefer* to see the paper isn't cherry-picking its one comparison point, but this is not universally mandatory — many single-lab domain-specialization papers (including most wildlife-ML papers) compare only to their own base architecture. | HIGH if attempted (a second codec's own training/inference pipeline). | **Recommendation: explicitly scope OUT**, exactly as the project's Key Decisions already do ("Baseline so sánh là Diff-ICMH gốc, không phải VTM"). State the omission as a disclosed Limitation ("we compare against the unmodified base model as the primary counterfactual; benchmarking against other ICM codecs is future work") rather than silently omitting it — this converts a possible reviewer complaint into a pre-empted, defensible scope decision. |

### Table Stakes — Domain-Specialization Evidence (Q2)

The project's own Core Value statement already identifies the central reviewer question correctly. This section extends and hardens that with what the literature requires as *evidence*, not just claims.

| Feature | Why Expected | Complexity / GPU Cost | Notes |
|---------|--------------|------------------------|-------|
| **Additive ablation table (Original → +H1 → +H1+H2 → +H1+H2+H3/full)** | This is the standard way domain-adaptation/fine-tuning papers demonstrate a mechanism contributes beyond the trivial baseline (train-on-new-data). Reviewers of fine-tuning papers routinely reject work that shows only "fine-tuned beats not-fine-tuned" without isolating *which* added mechanism did the work. | HIGH — each row is a separately trained configuration; this table is the most expensive single artifact in the project (already Active requirement, already correctly identified in team's 6 acceptance criteria #3). | **This is where the "L1/L2 TGM needs zero training" fact from the plan is the single biggest budget lever in the whole project** — the +H3(L1/L2) ablation row can be added at near-zero marginal GPU cost since it only changes the text prompt at decode time on an *already-trained* +H1(+H2) checkpoint. Exploit this aggressively: run the training-required rows (Original, +H1, +H1+H2) for real, and get several H3-variant columns "for free" by decoding the same checkpoint with different prompt templates. |
| **Statistical uncertainty on every ablation/RD number (bootstrap CI)** | Ablation deltas between configs are frequently small (fractions of a mAP point, tenths of a dB) and reviewers now routinely ask "is this within noise?" A table of point estimates with no CI is the single most common reason a fine-tuning/ablation paper draws a "did you actually measure a difference?" review comment. | LOW — CI is a resampling computation on already-computed per-image metrics, no extra GPU. | **Gap vs. the team's existing plan:** the plan only calls out bootstrap CI for the H2 AP_s/m/l table (T4 milestone). Extend this requirement to **every** table in the report (RD curves, BD-rate table, ablation table, cost-of-specialization table), not just one. Cheapest way to afford this: keep eval-set sizes fixed and modest (150–400 images) but always compute and report the CI — CI cost is ~free, it's the underlying eval decode that's expensive, so this is "more information for the same GPU spend," a pure win. |
| **Cost-of-specialization table: general-domain eval (Kodak/COCO) AND held-out-site eval (CCT)** | This is precisely what separates a credible specialization study from a cherry-picked one, and reviewers of domain-adaptation work reject papers that report only in-domain gains without measuring catastrophic forgetting and out-of-distribution generalization. Already correctly identified as team's 6 acceptance criteria #4 — this is the **best-designed part of the existing plan; do not cut it under schedule pressure.** | MEDIUM — decode-only (no new training) on two additional eval sets, once per trained checkpoint. | **Cheapest credible version:** Kodak (24 images) is free/tiny; use it as-is despite small-N (it's the field's own accepted convention for this exact purpose — "does PSNR/LPIPS on Kodak get worse" is a *relative* comparison across your own checkpoints, not an absolute claim, so small-N is less damaging here than for FID). COCO subset and CCT held-out-site can both be capped at ~150–200 images with bootstrap CI rather than the plan's original 500. |
| **Isolating the mechanism, not just the dataset switch** | This is the reviewer question itself, verbatim from `PROJECT.md`. The literature answer is always: show that the *loss/architecture change* (H2/H3), not merely *more/different training data* (H1), moves a metric that plain fine-tuning does not — which is exactly what the additive ablation table provides. No additional artifact is needed beyond the ablation table + cost-of-specialization table done correctly; the failure mode is doing one without the other (gains only shown in-domain, or mechanism not isolated from dataset). | — | This is a synthesis point, not a new deliverable: **the ablation table alone is necessary but not sufficient — reviewers reject specialization claims that show H2/H3 helping in-domain without also showing H1-alone already captured most of the gain** (i.e., without the ablation table, you cannot rule out that fine-tuning alone did everything). Both tables together are what answers the question; report them adjacent to each other in the paper, ideally cross-referenced. |
| **Explicit failure-mode disclosure when a mechanism does NOT help** | The project's own criterion #2 already states this correctly: *"Nếu không đóng góp, phải giải thích được vì sao"* (if it doesn't help, explain why). This is good practice, not a fallback — papers that report every proposed component helping are viewed with more suspicion than papers that report one component being a wash and explain the likely mechanism (e.g. α too aggressive → background texture corruption → detector false-positive increase, which the plan already anticipates for H2). | LOW — this is a writing/analysis deliverable, not compute. | Keep exactly as planned; this is a genuine strength of the existing acceptance criteria. |

**Why reviewers reject specialization papers (synthesized from domain-adaptation/fine-tuning literature conventions):**
1. Cannot separate "new mechanism" from "new dataset" — no additive ablation, or ablation is 2 rows (before/after) instead of a decomposed chain.
2. No cost-of-specialization measurement — claims a free lunch (all upside, no downside quantified).
3. Point estimates without uncertainty on small deltas — reviewer cannot tell signal from noise.
4. Claiming a mechanism (e.g. ROI-weighted loss) is novel when it is a straightforward reapplication of a well-published idea (see Differentiators — this is a real risk for H2 specifically, addressed below).
5. Ablation table has empty/missing cells for combinations the paper didn't have time to run — reviewers read gaps as cherry-picking. (This is exactly what the team's own T6 milestone guards against: *"bảng ablation không còn ô trống"* — keep that gate.)

---

### Table Stakes — Wildlife / Camera-Trap Community Conventions (Q3)

| Feature | Why Expected | Complexity / GPU Cost | Notes |
|---------|--------------|------------------------|-------|
| **Site-level (not image-level) train/val/test splits, with an automated leak check** | Confirmed field standard: iWildCam splits by camera location specifically to prevent background memorization; the broader camera-trap-ML survey literature identifies random/image-level splits as *the* most common methodological error inflating reported accuracy, because a stationary camera's background is itself a strong (spurious) predictive signal. | LOW — this is a data-engineering task, no GPU. | Already correctly implemented in the project (`split_check.py` as a CI-style assertion, `location_split.json`) — this is **above** what many published camera-trap papers actually do (the survey notes even 2026-era papers inconsistently apply this). Consider going one step further per the survey: split at **sequence/burst level**, not just site level, since a single triggered burst (several near-duplicate frames) can leak across train/val if only site is checked but burst membership isn't — worth a one-line assertion addition to `split_check.py` if not already covered. |
| **Empty-image false-positive rate, reported across a threshold sweep, not a single operating point** | MegaDetector's own published FPR is domain-dependent (commonly cited ~3–5%, higher with background motion e.g. wind-blown vegetation); the community convention is to sweep confidence threshold and report an FNR–FPR tradeoff curve, not a single number, precisely because a single threshold is not comparable across papers/datasets. | LOW-MEDIUM — reuses MegaDetector inference already required for detection metrics; the sweep is free (just don't threshold until analysis time — keep raw confidence scores in `results.jsonl`). | **Elevate this from "early-warning signal" (as currently framed in the plan's risk table) to a first-class required deliverable table**, because H2's own risk profile (background texture corruption on foliage/shadow at high α) makes this exactly the metric most likely to move. Cheapest version: this is nearly free since it's a re-analysis of already-collected detection scores, not new inference. |
| **Day-RGB / night-IR split reporting on all headline metrics** | Already correctly identified as the team's own criterion #6, and independently justified by the domain physics already documented in `PROJECT.md` (IR is single-channel replicated to 3 channels — chroma carries ~0 information at night, a fundamentally different signal statistics regime from day-RGB). Pooling day+night would average away exactly the effect (H1's bimodal-illumination argument) the project is trying to demonstrate. | LOW — reporting/slicing cost only, the underlying inference is shared with the pooled numbers. | Correctly planned; no change recommended. Just ensure `illumination` is a required field in `results.jsonl` schema from day one (already flagged in the plan's Appendix C checklist — good). |
| **Species-level accuracy AND a coarser group/taxonomic-fallback accuracy** | SpeciesNet's own convention (taxonomic rollup / fallback to a coarser rank when fine-grained confidence is low) is now a de facto community standard for handling irreducible fine-grained ambiguity in camera-trap imagery. | LOW — reporting only, reuses classifier output already needed for species-level numbers (just also compute at a coarser taxonomy level). | **Domain-specific synthesis point, add this even though not explicit in the current plan:** this project has an unusually strong reason to report both levels — the VAE Nyquist-limit finding (stripe/spot patterns below 2 cells/cycle at 8× downsample) predicts species-level accuracy will degrade for pattern-dependent species specifically, while family/group-level accuracy should remain robust since gross shape/size/color survive compression. Reporting both levels turns a potential weakness (species confusion) into a *predicted and confirmed* physical-limit finding, consistent with the project's own "measure the physical ceiling, don't argue about it" philosophy (T1.4). |
| **How LILA/Snapshot Serengeti results are conventionally reported** | Per Dryad/Nature Scientific Data documentation, Snapshot Serengeti is reported at 61-category species level with sequence-level consensus labels and per-camera-location metadata; papers using it are expected to respect the sequence structure (a "sequence" = one trigger event, several frames) rather than treating frames as i.i.d. | LOW | Reinforces the site/sequence-level split point above — treat "sequence" as the leakage-prevention unit, not just "site." |

---

### Differentiators (Q4)

**Novelty must be stated narrowly and defensibly — the literature has moved fast on ROI-adaptive generative compression in particular, and an unscoped novelty claim is a real risk to this project's core contribution.**

| Feature | Value Proposition | Complexity / GPU Cost | Notes — What's Actually New vs. Published |
|---------|---------------------|------------------------|--------------------------------------------|
| **ROI-weighted `L_dist` + `L_sem` inside the Diff-ICMH SD-prior codec, calibrated to wildlife's extreme foreground/background bit ratio** | Domain-specific mechanism directly targeting the project's strongest information-theoretic argument (animals <5–8% of frame area, camera-stationary background redundancy). | MEDIUM — one-line loss change per the project's own analysis (Chi tiết 2), no added params/VRAM; the *cost* is the training runs needed to validate it, not the mechanism itself. | **NOT novel as a general idea.** Confirmed prior art: (1) **TLIC** (arXiv:2401.08154, 2nd place CLIC 2024/DCC 2024) — ROI-weighted distortion + bit allocation in a learned codec, adversarial-loss texture generation; (2) **"Region-Adaptive Generative Compression with Spatially Varying Diffusion Models"** (arXiv:2604.01122, Apr 2026, Disney Research/ETH group — Relic, Azevedo, Zhang, Mandt, Gross, Schroers) — a diffusion codec that spatially varies its *denoising process itself* per an importance map and feeds the map into the entropy model, explicitly framed as going beyond prior "post-hoc guidance" ROI methods. **This second paper is the closest prior art and should be cited and distinguished explicitly in the report**, or a reviewer familiar with the space will flag it as scooping the idea. The defensible novelty here is narrower than "ROI-weighted loss in a diffusion codec": it is (a) applying ROI weighting to the *specific* Diff-ICMH architecture's dual loss terms at their *correct* spatial resolutions (VAE latent for `L_dist`, SD Encoder Layer 9 not middle block for `L_sem` — a domain-informed architectural choice the project has already derived correctly), (b) the wildlife-specific bit-redundancy argument and empirical AP_s/m/l characterization of where it helps/hurts, and (c) doing this credibly in a data-scarce, compute-scarce fine-tuning regime rather than large-scale training. State the claim at this precision, not at "we invented ROI-weighted generative compression." |
| **Domain-restricted vocabulary + structured real-metadata conditioning (TGM L1/L2) as bitstream side-information** | Near-zero bit overhead (~8–120 bits/image vs ~113 bits baseline) for a large gain in conditioning specificity, with an unusually clean "no oracle risk" argument for most fields (illumination/season/habitat are literal sensor metadata, not predicted) — a genuinely distinctive property of this domain versus most others where side-info must be predicted. | L1/L2: **near-zero GPU** (no training required, decode-only prompt swap — the single cheapest high-value experiment in the whole project). L3 (spatial grid + count): requires control-module retraining, correctly flagged by the team as the expensive tier. | **MEDIUM confidence this combination is novel** — could not find a directly matching prior work combining (i) a domain-restricted tag vocabulary, (ii) real (non-predicted) sensor metadata as text-conditioning side-information, and (iii) a generative/diffusion image *codec* specifically (as opposed to unconditional text-to-image generation, which has extensive metadata-conditioning literature). Absence of found prior art is **not proof of novelty** — recommend one targeted literature check specifically on "metadata-conditioned generative compression" and "sensor side-information neural codec" closer to the writing phase (T5–T6), before the novelty claim is finalized in the report. |
| **Real sensor metadata (timestamp/site-ID/illumination) as conditioning side-information, distinct from predicted/learned tags** | Removes the "oracle trap" that plagues most attribute-conditioning work in other domains (where the attribute must be predicted at encode time and errors compound); genuinely domain-specific advantage worth foregrounding as a contribution in its own right, not just a TGM implementation detail. | Same as above (bundled with L2). | Frame this as a **methodological point applicable beyond this project** (any sensor-network compression problem with reliable side-channel metadata could reuse this pattern) — this generalization argument strengthens the differentiator beyond "we did TGM for wildlife." |

---

### Anti-Features — What This Project Should Deliberately NOT Do (Q5)

| Feature | Why It Seems Appealing | Why It's Actually Problematic | Alternative / What To Do Instead |
|---------|--------------------------|-------------------------------|-------------------------------------|
| **Claiming individual animal re-identification (re-ID) capability** | Would be a highly compelling headline result (individual tracking from compressed camera-trap streams) and re-ID is a hot topic in wildlife ML (MegaDescriptor etc.). | **Physically blocked by the VAE, not by the codec.** Per the project's own correct derivation: an 8× spatial downsample reduces a zebra's ~5–8px stripe period to 0.6–1 latent cell, below the 2-cells/cycle Nyquist minimum — information destroyed at VAE *encode*, unrecoverable even at bpp=∞/lossless bitstream. Claiming re-ID capability would be an easily falsifiable claim a reviewer with basic signal-processing background will reject on inspection, independent of any experiment run. | Reframe the target explicitly as **"presence, location, pose, and coarse appearance sufficient for detection + species classification,"** and *prove* the re-ID ceiling with a quantitative MegaDescriptor experiment (already planned, T1.4) rather than asserting it — a measured negative result stated plainly is more credible than an unaddressed gap. |
| **Claiming to beat VTM-18.2 on PSNR at matched bpp** | Would read as a strong, simple headline number. | Structurally impossible for a generative codec at typical operating points — this is the well-established perception-distortion tradeoff (generative/diffusion codecs trade PSNR for perceptual quality/generative plausibility; VTM optimizes PSNR directly). Diff-ICMH's own paper (Figure 7) already shows this gap for the base model; claiming to close it post-specialization would contradict the mechanism the project itself is using. A reviewer will immediately check this and read an unsubstantiated claim as a red flag on the rest of the paper's rigor. | Correct framing already adopted in `PROJECT.md`: **compare against the unmodified pretrained Diff-ICMH on the same domain**, not VTM. Keep VTM only as a contextualizing anchor curve (Table Stakes section above), explicitly disclaiming the PSNR race in the report's framing paragraph — this pre-empts the objection rather than inviting it. |
| **Cherry-picked qualitative figures with no failure cases shown** | Best-looking reconstructions make for a more persuasive-looking paper at a glance. | Standard, near-universal reviewer critique of generative-model papers; undermines trust in *all* quantitative claims once a reviewer notices selective figure choice (a single suspicious figure can retroactively cast doubt on tables the reviewer didn't have time to scrutinize). | The project's own planned `failure_taxonomy.md` (6 failure modes, T7.2) is exactly the right countermeasure — **use a systematic, non-hand-picked sampling protocol** for qualitative figures: e.g., stratified random sample across bitrate/day-night/species-group, plus explicitly the worst-N examples by a fixed metric (not "worst-looking to the eye"), shown *alongside* best-case examples in the same figure grid. |
| **Shipping a generative codec that can hallucinate content in ecologically empty scenes without measuring/disclosing it** | Generative priors filling in plausible detail is the entire mechanism that makes the codec work at low bitrate — it is not something that can be "turned off," and it's easy to treat it as purely an image-quality question. | This is a **data-integrity problem specific to scientific/ecological imagery, not merely an aesthetic one**, and is now an actively named concern in the compression literature: a CHI 2026 paper ("When the Codec Hallucinates: User Perceptions of Miscompressed Images") studies exactly this user-trust failure mode; neural-compression hallucination has also been shown to mislead deepfake detectors, and generative compression is explicitly flagged as unsuitable for medical imaging/forensics without fidelity guarantees for the same reason. For camera-trap ecology specifically: a codec that invents an animal in a verified-empty frame, or invents/removes species-diagnostic features (stripe count, spot pattern, horn shape, litter size) that were not actually in the source, would corrupt the downstream scientific record (occupancy models, population counts, presence/absence data feeding conservation decisions) — a much higher-stakes failure than a blurry photo. | **This is a required table-stakes disclosure, framed here as an anti-feature to avoid *ignoring*:** measure a **hallucination-on-empty-images rate** (run the codec on a verified-empty ground-truth subset at each bitrate point, run MegaDetector on the decoded output, report the fraction that now trigger a detection above threshold) — this is nearly free, since it reuses MegaDetector inference already needed elsewhere and needs no new training. The project's plan already includes this exact metric ("tỉ lệ ảo giác trên ảnh rỗng," T7.1–T7.2) — **elevate it from a qualitative appendix item to a required numbered table with the same CI treatment as every other metric**, and discuss it explicitly under Limitations as a data-integrity caveat for any operational deployment, not just an image-quality footnote. |
| **Treating pseudo-GT SAM masks as if they were true pixel-level segmentation ground truth** | Simplifies reporting — "segmentation mIoU" reads as a clean, standard number. | No pixel-level ground truth exists for this domain (no "Cityscapes of wildlife"); presenting SAM-from-bbox pseudo-GT as if it were true GT overstates precision and misrepresents the segmentation numbers' actual error bars (SAM's own bbox→mask errors propagate silently into the reported mIoU). | Already correctly flagged in `PROJECT.md` Out of Scope / Constraints. Keep the explicit "pseudo-GT" labeling on every segmentation table/figure caption, not just once in a Limitations paragraph — repetition at the point of use prevents a reader skimming past the caveat. |

---

### Reproducibility Artifacts (Q6)

What a credible reproducibility package contains for this kind of work, per NeurIPS-checklist-era community norms and the specific constraints of this project:

| Artifact | Why Required | GPU/Effort Cost | Notes |
|---|---|---|---|
| **Exact commands + pinned environment per experiment** | NeurIPS-checklist-era convention: "exact command and environment needed to reproduce results" is now the baseline expectation, not a bonus. | LOW (engineering time, no GPU) | Already planned (5 `environment.yml` files). Good — keep pinned versions, not floating `>=`. |
| **`results.jsonl` as single source of truth + `make_all_figures.py`** | Prevents the classic failure mode of numbers in tables silently drifting from numbers in figures; also *is* the reproducibility artifact for all downstream analysis (bootstrap CI, BD-rate, ablation) without re-running any GPU job. | LOW | Already an excellent decision in the project's Key Decisions — this is above what many published papers actually provide. Keep the schema versioned (a schema change mid-project should bump a `schema_version` field, not silently alter meaning of old rows). |
| **Data split manifests + automated leak check (`split_check.py`)** | Directly reproducible proof of no site-leakage — reviewers/replicators cannot verify a leakage claim from prose alone; a runnable assertion script is what actually earns trust here. | LOW | Already planned; recommend running it in CI (or at minimum, before every training run) so leakage can never silently reappear after a data-pipeline edit. |
| **Fine-tuned deltas only, not full re-hosted SD 2.1 checkpoint** | Storage/bandwidth efficiency; SD 2.1 base weights, VAE, and RAM++ are frozen and already public — re-hosting them wastes the 2TB Drive budget and complicates license/attribution. | LOW | Release only codec `E_c/D_c` + control-module weights + (if L3 attempted) the modified TGM head, with a clear README stating which public base checkpoint they must be combined with. |
| **Model/dataset card documenting known limitations** | Standard for responsible ML releases; also directly operationalizes the project's own "Limitations ≥1 page" acceptance criterion. | LOW (writing) | Must explicitly cover: VAE Nyquist/re-ID ceiling, pseudo-GT SAM mask caveat, hallucination-on-empty rate, day/night imbalance, oracle-vs-predicted metadata fields (occupancy only), and the compute-driven scope cuts (crop size, eval-set size, DDIM steps) versus the original paper's setup. |
| **Minimal end-to-end smoke test on a tiny fixed sample** | Lets a reviewer or future user verify the pipeline runs and reproduces *one* number in minutes, without needing the full 50-GPU-hour budget themselves — this is the single highest-leverage trust-building artifact for a compute-constrained project, since nobody else will have 50 spare Colab-Pro hours to fully replicate the report. | LOW (a few minutes GPU once, to validate the script itself) | **Not explicit in the current plan — recommend adding.** E.g. "run `smoke_test.sh` on 5 bundled images, confirm bpp/PSNR/mAP match `smoke_test_expected.json` within tolerance." |
| **BD-rate/CI computation scripts, not just final tables** | Lets a reader/reviewer re-derive every table number from `results.jsonl` without re-running any model — separates "we computed this once, trust us" from "here is the exact, re-runnable derivation." | LOW | Natural extension of `make_all_figures.py`; keep BD-rate and bootstrap-CI logic in versioned scripts, not notebook cells that were run once and discarded. |

---

## Feature Dependencies

```
split_check.py + location_split.json (site/sequence-level split)
    └──requires-first──> ALL training runs, ALL eval (no leakage guard = no credible numbers downstream)

SAM pseudo-GT masks (val) + ROI masks (train)
    └──requires──> H2 training runs (ROI-weighted loss needs the masks to weight)
    └──requires──> segmentation mIoU eval

Original pretrained Diff-ICMH baseline (decode-only, no training)
    └──requires-first──> ALL ablation table rows (it IS row 0)
    └──requires-first──> cost-of-specialization table (it is the pre-specialization reference point)

H1 (domain fine-tune) checkpoint
    └──requires──> H2 checkpoint (H2 = H1 + ROI-weighted loss, trained from/alongside H1 config)
    └──requires──> H3-L3 checkpoint (L3 needs control-module retraining on top of a fine-tuned base)
    └──enhances──> H3-L1/L2 (these are decode-only prompt swaps on ANY trained checkpoint — near-zero marginal cost, can be applied to H1, H2, or full)

MegaDetector inference (frozen, off-the-shelf)
    └──shared-by──> detection mAP metric
    └──shared-by──> empty-image false-positive-rate metric
    └──shared-by──> hallucination-on-empty-images rate metric
    (run once per decoded image set, reuse the scores three ways — do not re-run per metric)

≥4 λ_rate trained/derived points
    └──requires──> RD curves
    └──requires──> BD-rate / BD-accuracy tables

Ablation table (additive H1→H2→H3 chain) + Cost-of-specialization table (Kodak/COCO + CCT held-out-site)
    └──jointly-answer──> the core reviewer question ("what's different from plain fine-tuning?")
    (neither alone is sufficient — see Table Stakes Q2 synthesis note)

VTM anchor curve (CPU-only, no GPU dependency)
    └──independent-of──> everything above; can be produced any time, in parallel, on CPU while GPU is busy
```

### Dependency Notes

- **The site-level split guard is the true root dependency of the whole project.** Every other number is retroactively worthless if leakage is discovered late — this is why it belongs at T1, before any GPU spend, exactly as currently planned.
- **H3-L1/L2 being decode-only is the project's biggest hidden budget lever.** It can be layered onto *any* already-trained checkpoint (H1, H2, or full) for near-zero marginal GPU cost, which is why it should be exploited to fill out ablation-table cells and bitrate points that would otherwise require dedicated training runs.
- **The ablation table and cost-of-specialization table are complementary, not redundant** — cutting either one under schedule pressure breaks the answer to the project's own Core Value question, even though each individually looks like a "nice-to-have extra table." If forced to cut only one experiment class late in the schedule, do not cut either of these before cutting bitrate points (3 points + disclosed quadratic BD-rate is an acceptable degradation; a missing ablation or missing cost-of-specialization table is not).
- **MegaDetector inference is a shared resource across three separate metrics** (detection mAP, empty-image FP rate, hallucination rate) — architect the eval harness to compute and cache raw MegaDetector confidence scores once per decoded image, then derive all three metrics from that cache. This is a straightforward implementation efficiency but easy to miss if the eval harness is built metric-by-metric instead of inference-batch-first.
- **VTM anchor curve has no GPU dependency at all** — schedule it opportunistically on CPU during any GPU-busy period; it should never compete with GPU-bound work for calendar time.

---

## MVP Definition

Reframed against the project's own 6 acceptance criteria (`docs/ke-hoach-difficmh-wildlife-8-tuan.md` §5.1) — validated as correct and extended with the gaps identified above.

### Launch With (v1) — the credibility floor, cannot ship without these

- [ ] Site-level (ideally sequence-level) split + automated leak check — **before anything else**
- [ ] RD curves, ≥3–4 bitrate points, for Original / +H1 / +full, on detection + species classification + segmentation (team's own criterion #1, with the 3-vs-4-point fallback disclosed if budget forces it)
- [ ] Additive ablation table (Original→+H1→+H2/H3→full) **with bootstrap CI on every cell** — extends criteria #2/#3 with the CI requirement identified above
- [ ] Cost-of-specialization table on BOTH axes: general-domain (Kodak/COCO) AND held-out-site (CCT) — team's own criterion #4, unchanged, do not cut
- [ ] Day-RGB / night-IR split reporting on all headline metrics — team's own criterion #6, unchanged
- [ ] Empty-image false-positive rate AND hallucination-on-empty-images rate, both across the bitrate sweep — elevates existing plan items to required, numbered deliverables
- [ ] Species-level AND group-level accuracy reporting — new addition, cheap, connects directly to the VAE-ceiling finding
- [ ] Substantive Limitations section (VAE/re-ID ceiling, pseudo-GT caveat, hallucination rate, oracle-vs-predicted metadata, compute-driven scope cuts vs. original paper setup) — team's own criterion #5, extended with explicit compute-cut disclosure
- [ ] VTM anchor curve (contextualizing only, not competitive) — new addition per Table Stakes Q1, cheap (CPU)
- [ ] Reproducibility package: pinned envs, `results.jsonl` + figure scripts, split manifests, fine-tuned-delta checkpoints, model card, smoke test

### Add After Validation (v1.x)

- [ ] Second neural-codec baseline (TransTIC/ELIC) — only if H1–H3 land early and GPU budget remains; otherwise stays a disclosed Limitation
- [ ] FID at a larger, more statistically stable sample size — only if eval budget allows beyond the LPIPS/DISTS-primary strategy
- [ ] H3-L3 (coarse spatial grid + count conditioning) — the one TGM tier that requires control-module retraining; attempt only after L1/L2 signal is confirmed positive (per plan, this is the intended de-risking order already)
- [ ] Additional λ_rate points beyond the minimum 4, if the H3-L1/L2 zero-training lever frees up enough GPU budget

### Future Consideration (v2+) — explicitly out of this project's scope

- [ ] H4 task-aware SC loss (DINOv2/BioCLIP) — already correctly deferred by the team (contradicts task-agnostic framing, ≥60 GPU-hours the project doesn't have)
- [ ] A second neural-codec baseline done rigorously (own training pipeline, not just inference) — future work, not v1.x
- [ ] Individual re-ID revisited at higher VAE resolution / different architecture — only meaningful if the underlying VAE downsample factor changes, which is out of scope by design (frozen architecture constraint)
- [ ] Field deployment validation (actual satellite/2G/LoRa bandwidth trial) — this project validates the codec, not the deployment; a natural next-milestone item, not this one

---

## Feature Prioritization Matrix

| Feature | Research Value | GPU Cost | Priority |
|---------|-----------------|----------|----------|
| Site-level split + leak check | HIGH (blocks everything) | LOW | P1 |
| Unmodified pretrained baseline eval | HIGH (load-bearing counterfactual) | LOW | P1 |
| Additive ablation table + bootstrap CI | HIGH (answers Core Value question) | HIGH | P1 |
| Cost-of-specialization table (Kodak/COCO + CCT) | HIGH (answers Core Value question, other half) | MEDIUM | P1 |
| RD curves + BD-rate (4 points target) | HIGH (table stakes for any compression paper) | HIGH | P1 |
| Day/night split reporting | HIGH (domain-required, cheap) | LOW | P1 |
| Empty-image FP rate + hallucination rate | HIGH (data-integrity, cheap via shared MegaDetector inference) | LOW-MEDIUM | P1 |
| Species-level + group-level accuracy | MEDIUM-HIGH (connects to VAE ceiling finding) | LOW | P1 |
| H3-L1/L2 domain vocab + metadata conditioning | HIGH (differentiator, near-zero cost) | ~ZERO | P1 |
| VTM anchor curve | MEDIUM (expected, contextualizing only) | LOW (CPU) | P1 |
| Reproducibility package (env, split manifests, results.jsonl, model card, smoke test) | HIGH (credibility + team's own criteria) | LOW | P1 |
| H2 ROI-weighted loss (V1: `L_dist` only) | HIGH (core differentiator) | MEDIUM | P1 |
| H2 ROI-weighted loss (V2: + `L_sem`@EncLayer9) | MEDIUM-HIGH (strengthens differentiator) | MEDIUM | P2 |
| FID at large sample size | LOW-MEDIUM (nice completeness, unstable at small N anyway) | MEDIUM | P2 |
| H3-L3 spatial grid + count conditioning | MEDIUM (highest-ambition TGM tier) | HIGH | P2 |
| Second neural-codec baseline | MEDIUM (nice-to-have, not mandatory per literature) | HIGH | P3 |
| H4 task-aware SC loss | LOW for this project (contradicts framing) | VERY HIGH | Out of scope (P3/future) |

**Priority key:**
- P1: Required for the report to be credible as domain-specialization research
- P2: Strengthens the contribution, add if H1–H3-core land on schedule
- P3: Genuinely optional / future work, explicitly disclosed as out of scope rather than silently omitted

---

## Sources

**ICM/ICMH evaluation conventions:**
- [Explicit Residual-Based Scalable Image Coding for Humans and Machines](https://arxiv.org/html/2506.19297) — BD-rate/BD-acc dual-axis reporting convention in current ICMH literature
- [Rate-Distortion in Image Coding for Machines](https://arxiv.org/pdf/2209.11694)
- [Bjøntegaard Delta (BD): A Tutorial Overview of the Metric, computation](https://arxiv.org/pdf/2401.04039) — 4-point minimum, piecewise-cubic fit
- JVET common test conditions (QP 22/27/32/37 as field-standard 4-point convention) — [vvenc encoder performance wiki](https://github.com/fraunhoferhhi/vvenc/wiki/Encoder-Performance), [bjontegaard PyPI](https://pypi.org/project/bjontegaard/)
- `docs/NeurIPS-2025-diff-icmh-*.pdf` and `docs/Diff_ICMH__NeurIPS_2025___Camera_Ready_Appendix.pdf` — the direct precedent this project extends (own λ_rate sweep, own metric choices, Appendix B.1 eval protocol, Appendix A.2 ablation setup) — HIGH confidence, primary source, referenced via `docs/ke-hoach-difficmh-wildlife-8-tuan.md` summary

**Domain-specialization / fine-tuning ablation conventions:**
- [Domain-Aware Fine-Tuning: Enhancing Neural Network Adaptability (AAAI)](https://arxiv.org/abs/2308.07728) — ablation-table conventions for isolating fine-tuning mechanisms from dataset effects
- General domain-adaptation literature convention on catastrophic-forgetting / OOD cost measurement (synthesized across search results; MEDIUM confidence on specific reviewer-rejection reasoning, HIGH confidence on the "measure both in-domain gain and OOD cost" convention itself)

**Camera-trap / wildlife ML conventions:**
- [Everything I know about ML and camera traps (agentmorris survey)](https://agentmorris.github.io/camera-trap-ml-survey/) — HIGH confidence, actively maintained community survey; site-level splits, empty-image handling, taxonomic-fallback (SpeciesNet-style) conventions
- [The iWildCam 2021 Competition Dataset](https://arxiv.org/pdf/2105.03494) — site-level split-by-camera-location convention
- [Snapshot Serengeti, Scientific Data (Nature)](https://www.nature.com/articles/sdata201526) — dataset structure, sequence-level consensus labeling, per-location metadata
- [Filtering Empty Camera Trap Images in Embedded Systems](https://arxiv.org/pdf/2104.08859) — empty-image false-positive handling
- MegaDetector empty-image false-positive rate (~3–5%, background-dependent) — synthesized from search results, MEDIUM confidence (no single canonical benchmark paper found; consistent with community-reported figures across multiple secondary sources)

**Differentiators / ROI-adaptive compression prior art:**
- [Region-Adaptive Generative Compression with Spatially Varying Diffusion Models (arXiv:2604.01122, Apr 2026)](https://arxiv.org/abs/2604.01122) — HIGH confidence, directly fetched abstract; closest prior art to H2, must be cited and distinguished
- [TLIC: Learned Image Compression with ROI-Weighted Distortion and Bit Allocation (arXiv:2401.08154, DCC 2024)](https://arxiv.org/abs/2401.08154) — HIGH confidence; established ROI-weighted-distortion prior art predating this project
- [ROI-based Deep Image Compression with Implicit Bit Allocation](https://arxiv.org/pdf/2511.08918)
- [Learned Image Compression for Vision-Language-Action Models](https://arxiv.org/html/2606.16253) — task-guided bit allocation without external spatial priors, adjacent line of work

**Hallucination / data-integrity in generative compression:**
- [When the Codec Hallucinates: User Perceptions of Miscompressed Images (CHI 2026)](https://doi.org/10.1145/3772318.3790293) — HIGH confidence, directly names and studies this failure mode
- Neural-compression hallucination misleading deepfake detectors, and generative compression flagged as unsuitable for medical imaging/forensics without fidelity guarantees — synthesized from search results, MEDIUM confidence (multiple consistent secondary mentions, no single primary source fetched in full)
- [CoD: A Diffusion Foundation Model for Image Compression](https://arxiv.org/pdf/2511.18706), [CADC: Content Adaptive Diffusion-Based Generative Image Compression](https://arxiv.org/pdf/2602.21591) — current diffusion-codec landscape context

**Reproducibility conventions:**
- [NeurIPS Paper Checklist Guidelines](https://neurips.cc/public/guides/PaperChecklist) — HIGH confidence, primary/canonical source for ML reproducibility expectations
- [Improving Reproducibility in Machine Learning Research (NeurIPS 2019 Reproducibility Program report)](https://arxiv.org/pdf/2003.12206)

**Project-internal sources (primary, HIGH confidence):**
- `D:/Diff_ICMH/.planning/PROJECT.md` — Core Value, Requirements, Constraints, Key Decisions (GPU budget: ~50h total, 5–6× cut vs. original plan)
- `D:/Diff_ICMH/docs/ke-hoach-difficmh-wildlife-8-tuan.md` — full 8-week plan, §5.1 six acceptance criteria (verified and extended above), architecture analysis (§1), H1–H4 hypothesis design (§2), risk register, Appendix A original-paper parameter reference, Appendix C startup checklist

---
*Feature research for: Wild-Diff-ICMH — domain-specialized generative image compression for camera-trap wildlife imagery*
*Researched: 2026-09-07*
