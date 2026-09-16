<!-- GSD:project-start source:PROJECT.md -->

## Project

**Wild-Diff-ICMH**

Chuyên biệt hoá **Diff-ICMH** — codec nén ảnh dựa trên generative prior của Stable Diffusion 2.1, phục vụ đồng thời máy và người — cho **miền ảnh bẫy ảnh động vật hoang dã (camera trap)**. Dự án fine-tune codec + control module từ checkpoint tác giả trên dữ liệu bẫy ảnh, rồi thêm hai cơ chế chuyên biệt hoá: **ROI-weighted loss** ưu tiên bit cho vùng có động vật, và **domain-aware Tag Guidance Module** khai thác metadata sẵn có của bẫy ảnh (ánh sáng ngày/đêm IR, sinh cảnh theo site, mùa từ EXIF).

Người dùng cuối là các nhà sinh thái học vận hành mạng bẫy ảnh trong khu bảo tồn: hàng triệu ảnh mỗi mùa, đường truyền vệ tinh/2G/LoRa vài chục kbps, cần ảnh vừa đủ cho pipeline máy (phát hiện — định loài — đếm) vừa đủ cho chuyên gia kiểm tra thủ công.

**Core Value:** **Chứng minh được rằng cơ chế chuyên biệt hoá (H2 ROI-weighted loss hoặc H3 domain-aware TGM) đóng góp vượt trên fine-tuning thuần (H1)** — tức trả lời được câu hỏi phản biện "đóng góp của các bạn khác gì với việc fine-tune một model có sẵn trên dataset khác?". Nếu mọi thứ khác thất bại, điều này phải đứng vững.

### Constraints

- **Compute**: Colab Pro ~100 units/tháng ≈ ~50h L4 hoặc ~19h A100 cho cả dự án — Đây là tài nguyên khan hiếm nhất, nhỏ hơn giả định của kế hoạch gốc ~5–6 lần. Mọi run phải vừa ngân sách này.
- **Compute**: Session Colab ngắt sau ~12h và có thể ngắt bất kỳ lúc nào — **Mọi training run bắt buộc phải resume được từ checkpoint**; không có run nào chạy liền mạch 15–24h như kế hoạch gốc giả định.
- **VRAM**: GPU Colab (L4 24GB / A100 40GB tuỳ phiên) — Crop 256² thay vì 512² của paper; batch nhỏ + gradient accumulation để giữ effective batch khớp paper.
- **Timeline**: 9–10 tuần từ 07/09/2026, chậm nhất **~16/11/2026** — Trễ là phải cắt phạm vi theo thứ tự hy sinh, không kéo lịch.
- **Nhân lực**: 2 người — TV-A (Data & Evaluation Lead), TV-B (Model & Training Lead). Giao diện giữa hai luồng là **hợp đồng thư mục**, không chờ nhau qua tin nhắn.
- **Lưu trữ**: Google Drive 2TB — đủ cho corpus 60K + checkpoint + ảnh decode; không phải ràng buộc.
- **Dữ liệu**: Không có nhãn segmentation cấp pixel cho miền bẫy ảnh — Phải dùng pseudo-GT sinh bằng SAM từ bbox, và ghi rõ ảnh hưởng trong Limitations.
- **Kiến trúc**: SD 2.1 UNet, VAE, RAM++ **đóng băng** — Chỉ được can thiệp vào codec `E_c`/`D_c`, control module, hàm loss, và nội dung điều kiện text `c`.

<!-- GSD:project-end -->

<!-- GSD:stack-start source:research/STACK.md -->

## Technology Stack

## 1. Recommended Stack

### 1.1 Core Training Stack

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| **PyTorch** | **2.11.0** (Colab's current preinstalled default as of the 2026.07 runtime; 2.9.0/2.10.0 on older Colab snapshots) | Deep learning framework | This is what Colab actually ships — do not fight it by pinning an old torch. PyTorch's own latest stable is 2.13.0 (July 2026), but **you should use whatever Colab preinstalls**, not pip-install a different major version, to avoid CUDA/driver mismatches on a runtime you don't control. Verify with `import torch; print(torch.__version__, torch.version.cuda)` at the start of every session — it can change between Colab runtime rollouts (quarterly cadence: Jan/Apr/Jul/Oct). Confidence: MEDIUM (Colab's runtime-version-FAQ page lists exact historical versions per quarter; current-quarter version must be verified live). |
| **Python** | **3.12** (Colab's current default, per 2026.07 runtime: Python 3.12.13) | Runtime | Non-negotiable — set by Colab, not by you. This alone breaks several of the repo's current pins (see §6). |
| **Lightning** | **`lightning` package (unified), ≥2.6, import as `lightning.pytorch`** — NOT `pytorch_lightning==1.5.0` as currently pinned | Training loop orchestration | `pip install pytorch-lightning` is a deprecated alias frozen at old releases; `pip install lightning` is the actively maintained unified package (Fabric + PyTorch Lightning + supporting libs), current release 2.6.1 (Jan 2026). Lightning 1.5.0 (the repo's pin) predates Python 3.12 support entirely and will not resolve/install cleanly against torch 2.11. See §6 for the concrete migration diff. Confidence: HIGH (PyPI + official Lightning-AI GitHub discussion on package naming). |
| **xformers** | Install via `pip install -U xformers` **from the PyTorch index**, matched to the installed torch build — do not pull a bare PyPI wheel | Memory-efficient attention in the frozen SD 2.1 UNet (only real lever for fitting 256² crops + control module in Colab VRAM) | xFormers has migrated to PyTorch's stable API/ABI so binaries built for torch 2.10+ generally work across later patch releases, but you must match major torch version. The repo has `xformers==0.0.22` **commented out** in requirements.txt — this version predates torch 2.x entirely and must not be installed as-is. Confidence: MEDIUM (xFormers PyPI page + multiple Colab-specific GitHub issues confirm recurring version-mismatch pain — treat as "verify at session start," not "pin and forget"). |
| **CompressAI** | **1.2.8** (released June 25, 2025) | Learned-codec building blocks: entropy bottleneck, hyperprior, rANS entropy coder, rate-distortion utilities | Actively maintained by InterDigitalInc; the repo's pin (`compressai==1.2.4`) is two minor versions behind but should still install — the real risk is CompressAI's pip install can silently drag in its own preferred torch/CUDA build if not installed with `--no-deps` against an already-present torch. Confidence: HIGH (verified live via PyPI). |
| **einops, kornia, omegaconf, timm, transformers** | Bump to latest within each library's `0.x`/`4.x` line rather than the exact repo pins | Tensor reshaping, differentiable CV ops, config management, RAM++/CLIP backbones | The repo's pins (`kornia==0.7.0`, `timm==0.9.7`, `transformers==4.41.0`) are from ~2023-2024 and were never validated against torch 2.11/Python 3.12. None of these have known hard incompatibilities with recent torch, but pin ranges rather than exact versions to let pip resolve a working combination against Colab's torch. Confidence: MEDIUM (no direct evidence of breakage found, but not verified either — treat as a Week-1 smoke-test item, consistent with the source plan's own T1.3/T2.3 tasks). |

### 1.2 Supporting Libraries (Perceptual/IQA metrics)

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| **pyiqa** | **0.1.15.post2** (actively updated through at least June 2026 — new metrics added monthly) | Unified toolbox: PSNR, SSIM, LPIPS, DISTS, NIQE, MUSIQ, FID-adjacent metrics, all with a consistent `pyiqa.create_metric()` API | **Replace the repo's `lpips==0.1.4` + separate DISTS package entirely with pyiqa.** One dependency instead of three, actively maintained, and it wraps LPIPS/DISTS internally with the deprecation warnings already fixed (the standalone `lpips` 0.1.4 package calls `torchvision.models.vgg16(pretrained=True)`, which is deprecated since torchvision 0.13 and hard-errors on newer torchvision — this is a **concrete, verified break** on the repo's current pin). Confidence: HIGH (PyPI + GitHub issue thread on the `pretrained=True` removal). |
| **torch-fidelity** | 0.4.0 | FID / IS / KID between real and reconstructed image distributions | Use for the RD-curve FID column; `torchmetrics.image.FrechetInceptionDistance` also works and depends on this package under the hood — pick one, don't install both metric stacks. |
| **torchmetrics** | latest `1.x` (`[image]` extras) | Alternative/cross-check path for LPIPS, DISTS, FID inside a Lightning-style logging loop | Use only if you want metrics auto-logged as `pl.LightningModule` metrics during validation; otherwise pyiqa alone is sufficient and lighter. |
| **bjontegaard** | 1.3.0 (PyPI: `pip install bjontegaard`) | BD-rate / BD-PSNR computation between RD curves | This is the standard, actively maintained Python BD-rate library — implements cubic-spline, PCHIP, and Akima interpolation; PCHIP variant matches the canonical Excel/VCEG reference implementation to 10 decimal places. Use this over hand-rolled BD-rate scripts. Confidence: HIGH (PyPI + cited validation against Bjøntegaard's reference method). |
| **pycocotools** | **2.0.11** | COCO-style mAP, AP_small/medium/large breakdown | Standard. Note the area thresholds are fixed at the COCO convention (small <32², medium 32²-96², large >96² **pixel²**, measured in the *evaluation image's* pixel space) — since camera-trap animals are frequently <32px on a side even at 1024×768, expect most detections to land in AP_small, which is exactly the bucket the source plan's H2 hypothesis needs reported separately. |

## 2. Camera-Trap Datasets (LILA BC)

### 2.1 Download mechanics (all three mirrors are free, no account/approval needed)

# Google Cloud (gsutil) — convert https://storage.googleapis.com/... to gs://...

# AWS S3 — public bucket, no credentials needed

# Azure (azcopy) — no SAS token needed for public containers

### 2.2 Dataset-by-dataset numbers

| Dataset | Total images | Bbox-annotated subset | License | Approx size | Notes |
|---|---|---|---|---|---|
| **Snapshot Serengeti** (all seasons) | **~7.1M images**, ~2.65M sequences, seasons 1-11 | **~78,000 images with ~150,000 bounding boxes** — bboxes are distributed as a **separate JSON file layered on top of the full-season image set**, not tied to one specific season; treat this ~78K-image subset as your primary bbox-labeled corpus | Community Data License Agreement (permissive variant) — free for research/commercial reuse with attribution | Per-season zips range widely: Season 1 ≈242GB, Season 5 ≈596GB, Season 7 ≈636GB, Season 11 ≈479GB (full 11-season corpus is several TB) | **Do not download all 11 seasons.** Download only the seasons that intersect the ~78K bbox-annotated image list, plus a modest random sample of additional (unboxed) images if you want unsupervised/self-training signal. 61 species categories, ~76% of images empty. |
| **Caltech Camera Traps (CCT)** | 243,100 images, 140 camera locations, SW USA | ~66,000 bounding box annotations, 21 species categories | CDLA-permissive | Tens of GB (subset-dependent) | The source plan's intent — CCT as a **held-out generalization/new-site eval set, never in train** — is directly supported by this bbox coverage; ~66K boxes is plenty for a few-hundred/thousand-image eval split. |
| **Wellington Camera Traps** | 270,450 images, 187 locations, NZ | Species labels; bbox coverage not confirmed in this pass — verify on the dataset page before depending on it for H2 mask generation | CDLA-permissive | Not confirmed | Good candidate for an *additional* held-out cross-domain eval set (different continent/species mix from Serengeti) if T7's "cost of specialization" analysis wants a third domain. Not required for MVP scope. |
| **Idaho Camera Traps** | ~1.5M images | Species + some bbox labels (coverage not confirmed here) | CDLA-permissive | Large (hundreds of GB if taken whole) | Oversized relative to project needs; only worth touching for a small stratified sample, not as a primary corpus. |
| **Missouri Camera Traps** | ~25,000 images, 20 species | **8,892 bbox-annotated images** (mostly vehicles and birds) | CDLA-permissive | ~10GB | Small, tractable, has bboxes — a reasonable secondary/smoke-test dataset if Serengeti download is slow, but not mentioned in the source plan; treat as optional. |

### 2.3 What metadata survives where

- **`datetime` per image**: present in the **COCO Camera Traps JSON** (`images[].datetime`) for all LILA camera-trap datasets — this is the reliable, always-available source. The original Snapshot Serengeti data-collection pipeline derived camera-active date ranges from image EXIF at the source, which suggests EXIF datetime *was* present in the original camera output, but LILA's redistributed, re-processed JPEGs are **not guaranteed to retain EXIF** after LILA's own reprocessing/repackaging pipeline. **Action: verify empirically on the first 100 downloaded images** (`exiftool` or `PIL.Image._getexif()`) before designing H3's L2 tag layer around EXIF — this is exactly risk R12 in the source plan, and it is correctly flagged there as a Week-1, not Week-5, check. If EXIF is stripped, fall back to the JSON `datetime`/`location` fields, which are confirmed present.
- **`location`/`seq_id`**: JSON only, per-image, always present — this is what your site-disjoint split (`split_check.py` in the source plan) must key off, not any embedded image metadata.
- **Habitat/site-level attributes**: not distributed by LILA at all — these must be manually annotated per site ID (a one-time, small lookup table) as the source plan already assumes for H3's `habitat` structured attribute.

## 3. Wildlife Task Models ("machine consumers" for eval)

### 3.1 Recommended models and install paths

| Task | Model | Install | Version/Status | Confidence |
|---|---|---|---|---|
| Detection (animal/person/vehicle) | **MegaDetector V6** | `pip install PytorchWildlife` (pulls MDv6 automatically; weights download on first run) — or `git clone microsoft/MegaDetector && pip install -e .` for the standalone CLI / fine-tuning path | V6 restructured around Ultralytics YOLOv9/v10/RT-DETR variants; the flagship compact variant is **50x smaller than V5** (2.3M params vs V5's 139.9M) at comparable accuracy — directly relevant given the ~50h budget: MDv6-compact inference is materially cheaper per image than V5. Community standard for camera-trap detection. | HIGH |
| Species classification | **SpeciesNet** | `pip install speciesnet` | Current PyPI release **5.0.5** (June 2026), Google's official ensemble classifier (EfficientNetV2-M backbone + MegaDetector for cropping), >2000 output labels including higher-taxa fallbacks and "blank"/"vehicle". This is the correct current answer to "Google's 2025 release" in the question — SpeciesNet is the production successor to the older internal classifiers. | HIGH |
| Individual re-ID (used only for the VAE-limit experiment, T1.4 — NOT for a re-ID feature) | **MegaDescriptor** via **`wildlife-tools`** | `pip install wildlife-tools` (companion repo `wildlife-datasets` for dataset loaders) | MegaDescriptor is a Swin-Transformer-based foundation model for animal re-ID, released in S/M/L flavors on HuggingFace Hub; outperforms generic CLIP/DINOv2 embeddings on wildlife re-ID benchmarks. Use strictly to *measure* the Nyquist-limit claim in the source plan (not to build a product feature) — one inference pass over a handful of striped/spotted animal crops is enough. | HIGH |
| Segmentation (pseudo-GT from bbox, and val-time comparison) | **SAM 2.1** (`facebookresearch/sam2`) — recommended over SAM 1 | `git clone facebookresearch/sam2 && pip install -e .`, checkpoints downloaded separately; use `SAM2ImagePredictor` with box prompts for single images (SAM2's image mode is a strict superset of SAM1's box-prompt workflow) | SAM 2.1 (Sept 2024 checkpoint refresh) is **Apache 2.0 licensed**, achieves higher mIoU than SAM1 at 1-click/1-box prompts while being ~6× faster. Given the ~50h budget and that mask generation is a one-time preprocessing pass (not per-training-step), SAM 2.1's speed advantage matters more here than its video capabilities (which this project doesn't need). | HIGH |
| Segmentation, lightweight alternative if disk/VRAM-constrained | **MobileSAM** | `pip install mobile-sam` or clone `ChaoningZhang/MobileSAM` | ~5× faster and ~7× smaller than FastSAM, comparable quality to SAM1 at a fraction of the compute — a fallback if SAM2.1's ~2.4GB checkpoint set is too much disk/VRAM pressure alongside the other 4 task-model environments. Not needed unless disk becomes the binding constraint (source plan already budgets ~500GB, which should be sufficient). | MEDIUM |

### 3.2 Dependency conflicts — mandatory environment isolation

## 4. Compression Evaluation Tooling

| Purpose | Library | Why |
|---|---|---|
| Codec building blocks / entropy coding | CompressAI 1.2.8 | Already in repo; bump minor version |
| Perceptual metrics (LPIPS, DISTS, NIQE, MUSIQ) | **pyiqa 0.1.15.post2** | Single actively-maintained package, replaces repo's standalone `lpips` (which has a live torchvision-deprecation break) |
| FID/KID | torch-fidelity 0.4.0 (or `torchmetrics[image]`) | Standard, pick one |
| BD-rate/BD-PSNR | bjontegaard 1.3.0 | Canonical PCHIP implementation, matches Excel/VCEG reference |
| Detection mAP + size breakdown | pycocotools 2.0.11 | Standard COCOeval; AP_small/medium/large thresholds are fixed COCO convention (32²/96² px²) |

## 5. Google Colab Pro Compute Reality (2026)

### 5.1 Compute-unit burn rates (Colab Pro, $9.99/month = 100 CU)

| GPU | CU/hour (community-measured) | Hours available from 100 CU | Notes |
|---|---|---|---|
| T4 | 1.19-1.96 (reports vary) | ~51-84h | Cheapest option; likely too slow per-iteration for 256² diffusion training to be worth the extra hours |
| **L4** | **~1.71** | **~58.5h** | **Best available tier for this project** — 24GB VRAM (matches the paper's assumed 24GB card), modern Ada architecture, better $/iteration than T4 for diffusion workloads |
| A100 40GB | 5.40 | ~18.5h | Matches PROJECT.md's own "~19h A100" estimate almost exactly — use this as a cross-check that the project's compute math is sound |
| A100 80GB | 7.52 | ~13.3h | Not worth it unless VRAM (not compute) is the binding constraint, which it shouldn't be at 256² crops |

### 5.2 Session limits and background execution

- **Colab Pro**: session hard cap ~24h; **idle timeout ~90 minutes** if the tab has no interaction AND no code is actively executing — a training loop that is actively running does not count as idle, so a live `trainer.fit()` call should survive past 90 minutes on its own. Background execution (continuing after the browser tab is closed) is a **Pro+ feature, not base Pro** — on base Pro, keep the tab open (or use a keep-alive trick) for the duration of a run.
- **Colab Pro+** ($49.99/mo, 500 CU): adds background execution and higher GPU-priority, but is **not what this project has budgeted** (PROJECT.md specifies "Colab Pro" — confirm with the user before assuming Pro+ economics).
- **Practical implication**: every training run MUST be resumable from a checkpoint (the project's own constraint, correctly identified in PROJECT.md), because (a) idle disconnects are possible even mid-run, (b) the 24h session cap means any run needing more than one day of wall-clock GPU time necessarily spans multiple sessions, and (c) with only ~50-58h total budget spread over 9-10 weeks, sessions will naturally be short (a few hours at a time), not the paper's assumed 15-24h unbroken runs.

### 5.3 Checkpoint/resume pattern (mandatory, not optional)

## 6. Training Stack: Where the Repo's Pins Break on Colab, and the Concrete Fix

### 6.1 Confirmed breaks

| Pin in `requirements.txt` | Why it breaks on Colab (Python 3.12) | Fix |
|---|---|---|
| `numpy==1.23.1` | NumPy 1.23.x has **no published wheel for Python 3.12** (Python 3.12 wheels start at NumPy 1.26.x) — pip will attempt a from-source build and fail without build tools, or simply fail to resolve | Bump to `numpy>=1.26,<2.0` (stay below 2.0 to avoid breaking the many old libs — kornia 0.7.0, timm 0.9.7, etc. — that still assume the NumPy 1.x C-API) |
| `pytorch_lightning==1.5.0` | Predates Python 3.12 support by ~2 years; will fail to resolve/install against a modern torch. Separately, `pytorch-lightning` (the split-name package) is a frozen/deprecated alias — new fixes only land in the unified `lightning` package | Switch to `lightning` package ≥2.x, `import lightning.pytorch as pl` |
| `xformers==0.0.22` (commented out, but implied needed for VRAM budget) | Built against a torch 1.x/early-2.x ABI; incompatible with torch 2.11 | Install the torch-index xFormers build matching the installed torch major version at session start, not a pinned version number |
| `lpips==0.1.4` | Internally calls `torchvision.models.vgg16(pretrained=True)` — `pretrained=` was deprecated in torchvision 0.13 and is **removed** (hard error, not just a warning) in the torchvision versions that ship alongside torch 2.11 | Replace with `pyiqa` (wraps LPIPS/DISTS with the modern `weights=` API already fixed) |

### 6.2 `train.py` / `configs/train_diffeic.yaml` — the Lightning 1.x → 2.x API breaks, with exact fixes

### 6.3 Net assessment

## 7. RAM++ / Recognize Anything

- **Current state**: `xinyu1205/recognize-anything` on GitHub, weights on HuggingFace Hub (`xinyu1205/recognize-anything-plus-model`). No PyPI package — install via `git clone` + `pip install -e .` or direct dependency on the repo path, consistent with the project's existing `src/recognize-anything/` vendored copy.
- **Tag vocabulary format**: RAM++'s default vocabulary is a fixed list of 4,585 tags (English common nouns/concepts), each with an LLM-generated textual description used to build the tag embedding at inference time (`generate_tag_des_llm.py` in the repo, calling an LLM API to produce per-tag descriptions). Tags are recognized via similarity between image features and these tag-description embeddings, **not** via a closed classification head — this is exactly why restricting/extending the vocabulary is tractable without retraining.
- **Restricting to a ~256-tag wildlife vocabulary**: this is a **supported, first-class RAM++ use case**, not a hack. The `inference_ram` function accepts an explicit custom tag list (the repo's own examples pass arbitrary tag lists like `["house", "car", "pig"]`), and the open-set recognition path works by comparing image features against whatever tag-description embeddings you provide. Concretely:
- **Extending with structured attributes (illumination/habitat/season/occupancy)**: these are **not RAM++ tags at all** in the source plan's design (L2 layer) — they're appended to the text conditioning string alongside RAM++'s output tags, sourced from metadata/EXIF/a lightweight classifier, not from RAM++ itself. This is consistent with what's technically supported: RAM++ produces a tag list, and the TGM (Tag Guidance Module) downstream is what assembles the final text prompt fed to SD 2.1 — extending the prompt-assembly logic outside RAM++ requires no changes to RAM++ itself.
- **Caveat on maintenance**: this is a research-lab GitHub repo (Tsinghua/OPPO), not a corporate SDK — expect to read source and adapt scripts rather than rely on polished public API stability. Given the project already vendors a working copy (`src/recognize-anything/`, confirmed present per PROJECT.md's Validated section), do not re-clone from upstream; patch the vendored copy in place and track the diff.

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| SAM 2.1 for mask pseudo-GT | SAM 1 (ViT-H), as the source plan specifies | If SAM2.1's dependency footprint conflicts with the segmentation env's other packages (SAM2 requires a somewhat newer torch than SAM1 tolerates) — SAM1 remains a safe fallback, just slower and marginally less accurate at box-prompt tasks |
| pyiqa for all perceptual metrics | Standalone `lpips` + `DISTS-pytorch` + `torch-fidelity`, kept as 3 separate packages | If you need pyiqa's LPIPS/DISTS numbers cross-checked bit-for-bit against the exact reference implementations (pyiqa reimplements rather than vendors some metrics) — for a report needing citation-exact reproducibility, keep the original `richzhang/PerceptualSimilarity` LPIPS install alongside pyiqa as a sanity check |
| L4 as the target GPU tier for training | A100 40GB | Only for short, VRAM-bound or wall-clock-critical runs late in the project (e.g., the final full-config run for the report's headline numbers) — A100's ~3.16× CU burn rate vs L4 means every A100-hour costs ~3 L4-hours of budget, so reserve A100 sparingly, not as the default tier |
| `lightning` unified package | Keep `pytorch_lightning==1.5.0`, run in an isolated env with an old NumPy | Not recommended at all for this project — Lightning 1.5 cannot install on Colab's Python 3.12 without either building NumPy from source or using a non-default older Python (which itself requires a custom Colab runtime setup, wasting setup time this project doesn't have) |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| `pytorch_lightning` (split package name) as pinned in requirements.txt | Frozen/deprecated alias, no longer receives fixes; version 1.5.0 predates Python 3.12 wheel availability | `lightning` (unified package), ≥2.6 |
| `numpy==1.23.1` pin | No Python-3.12 wheel exists | `numpy>=1.26,<2.0` |
| Standalone `lpips==0.1.4` as the only perceptual-metric dependency | Hard-errors on current torchvision due to removed `pretrained=True` kwarg | `pyiqa` (wraps a fixed version internally) |
| Original SAM (ViT-H) as the *only* segmentation option, if avoidable | Slower and lower-mIoU than SAM 2.1 at the exact box-prompt task this project needs, for no offsetting benefit (both are Apache/permissive-licensed, SAM2.1 has no feature the project needs less of) | SAM 2.1 |
| Downloading full multi-season Snapshot Serengeti zips (several TB) | Wildly oversized relative to the ~78K-image bbox-annotated subset this project actually needs, and will blow past the "≥500GB free disk" budget on download alone if done naively | Download only the season zips intersecting the bbox JSON's image list, plus a bounded random sample for unlabeled/self-training use |
| Assuming Colab Pro+ economics (background execution, 500 CU) | PROJECT.md specifies base Colab Pro (100 CU, no guaranteed background execution) — planning around Pro+ features that may not be provisioned will produce a schedule that silently assumes 5× the actual compute budget | Base Colab Pro assumptions throughout: tab must generally stay open/active, ~50-58h L4 total, resumable checkpointing every 1-2k steps as a hard requirement, not a nice-to-have |
| A single shared conda/pip environment for MegaDetector + SpeciesNet + SAM + MegaDescriptor | Confirmed real dependency conflicts (competing `ultralytics`/`yolov5` version pins across MegaDetector and SpeciesNet) | 3 isolated environments (detect/species/segment), communicating via JSON on disk — exactly as the source plan's R6 mitigation already specifies |

## Stack Patterns by Variant

- Increase batch size / reduce gradient-accumulation steps to use the extra VRAM productively, but budget the session's wall-clock time knowing it burns CU ~3.16× faster than L4 — don't let an A100 session run as long as an L4 session would.
- Swap SAM2.1 for MobileSAM in the segmentation env (smaller checkpoint), and delete decoded-image caches from prior eval rounds once their numbers are committed to `results.jsonl` — the source plan's own R11 mitigation.
- Fall back entirely to the JSON `datetime`/`location` fields for H3's L2 structured-attribute layer (season/illumination-adjacent timing) — these are confirmed present regardless of EXIF status, so this fallback carries zero redesign risk, just a data-source swap.

## Version Compatibility

| Package A | Compatible With | Notes |
|-----------|-----------------|-------|
| `lightning>=2.6` | `torch==2.11.0` (Colab current) | Also requires deleting the dead `LightningCLI` import and rewriting `accelerator`/`gpus` config keys per §6.2 |
| `numpy>=1.26,<2.0` | `torch==2.11.0`, `kornia==0.7.0`, `timm==0.9.7` | Stay below NumPy 2.0 — several of the repo's other pinned-era libraries assume the NumPy 1.x C-API; NumPy 2.0's ABI break is a separate, avoidable risk not worth taking on top of the Lightning migration |
| `xformers` (torch-index build) | Must match installed `torch` major.minor exactly | Reinstall xformers fresh at the start of any session where Colab's torch version changed (quarterly Colab runtime updates can silently bump torch) |
| `pyiqa==0.1.15.post2` | `torch==2.11.0`, `torchvision` (current) | Actively updated through mid-2026, tracks current torchvision's `weights=` API |
| `speciesnet==5.0.5` | Python `>=3.9,<3.14` | Compatible with Colab's Python 3.12 |
| `compressai==1.2.8` | `torch>=1.7` per its own stated floor, but **install with care** — its pip metadata has been reported to try to pull its own preferred torch/CUDA combination | Install torch first (accept Colab's preinstalled version), then `pip install compressai --no-deps`, then manually verify/install compressai's remaining non-torch deps |

## Sources

- [LILA BC — Data Sets](https://lila.science/datasets/) — dataset index, confirmed live
- [LILA BC — Snapshot Serengeti](https://lila.science/datasets/snapshot-serengeti/) — image counts, bbox subset size, license, per-season sizes (HIGH confidence, fetched directly)
- [LILA BC — FAQ](https://lila.science/faq/) — azcopy/gsutil/aws download command syntax, confirmed public/no-requester-pays buckets (HIGH)
- [LILA BC — Caltech Camera Traps](https://lila.science/datasets/caltech-camera-traps) — image/bbox counts (HIGH, via web search snippet)
- [LILA BC — Missouri Camera Traps](https://lila.science/datasets/missouricameratraps/) — image/bbox counts (MEDIUM, via search snippet)
- [microsoft/MegaDetector — GitHub](https://github.com/microsoft/megadetector) + [Releases](https://github.com/microsoft/MegaDetector/releases) — V6 architecture/size claims (HIGH)
- [microsoft/Pytorch-Wildlife — GitHub](https://github.com/microsoft/Pytorch-Wildlife) — `pip install PytorchWildlife` install path (HIGH)
- [google/cameratrapai — GitHub](https://github.com/google/cameratrapai) + [speciesnet — PyPI](https://pypi.org/project/speciesnet/) — version 5.0.5, June 2026, Python constraint (HIGH, fetched directly)
- [WildlifeDatasets/wildlife-tools — GitHub](https://github.com/WildlifeDatasets/wildlife-tools) — MegaDescriptor install path (HIGH)
- [facebookresearch/sam2 — GitHub](https://github.com/facebookresearch/sam2) — SAM 2.1 license (Apache 2.0), checkpoint notes (HIGH)
- [ChaoningZhang/MobileSAM — GitHub](https://github.com/chaoningzhang/mobilesam) — size/speed comparison vs FastSAM (MEDIUM)
- [InterDigitalInc/CompressAI — GitHub](https://github.com/InterDigitalInc/CompressAI) + [PyPI](https://pypi.org/project/compressai/) — version 1.2.8, June 2025 (HIGH, fetched directly)
- [bjontegaard — PyPI](https://pypi.org/project/bjontegaard/) — version 1.3.0, PCHIP validation claim (HIGH)
- [pyiqa — libraries.io](https://libraries.io/pypi/pyiqa) — version 0.1.15.post2, active 2026 releases (MEDIUM, via aggregator not direct PyPI fetch)
- [pycocotools — libraries.io](https://libraries.io/pypi/pycocotools) — version 2.0.11 (MEDIUM)
- [richzhang/PerceptualSimilarity — GitHub issue #108](https://github.com/richzhang/PerceptualSimilarity) — confirmed `pretrained=True` deprecation break (HIGH)
- [Lightning-AI/pytorch-lightning — GitHub discussion #17095](https://github.com/Lightning-AI/pytorch-lightning/discussions/17095) — `pytorch-lightning` vs `lightning` package naming/maintenance status (HIGH)
- [PyTorch Lightning 2.0 versioning docs](https://lightning.ai/docs/pytorch/2.0.0/versioning.html) — `accelerator`/`gpus` deprecation-to-removal timeline (HIGH)
- [Google Colab — runtime version FAQ](https://research.google.com/colaboratory/runtime-version-faq.html) — Python 3.12.13 / PyTorch 2.9.0-2.11.0 across 2026 quarterly runtimes (HIGH, fetched directly, but current-quarter value must be re-verified live)
- [Chris McCormick — Colab GPUs Features & Pricing](http://mccormickml.com/2024/04/23/colab-gpus-features-and-pricing/) — community-measured CU/hr rates for T4/L4/A100 as of March 2026 (MEDIUM — unofficial, self-reported, explicitly noted to fluctuate)
- [googlecolab/colabtools — GitHub issues on background execution](https://github.com/googlecolab/colabtools/issues/5950) — Pro vs Pro+ background-execution behavior and reported reliability issues (MEDIUM)
- [numpy/numpy — GitHub issue #23808](https://github.com/numpy/numpy/issues/23808) and related — NumPy 1.23 lacking Python 3.12 wheels (HIGH)
- [xinyu1205/recognize-anything — GitHub](https://github.com/xinyu1205/recognize-anything) — tag vocabulary structure, custom-tag inference support (MEDIUM-HIGH)
- Repo-internal: `D:/Diff_ICMH/requirements.txt`, `D:/Diff_ICMH/train.py`, `D:/Diff_ICMH/configs/train_diffeic.yaml` — verified directly by reading the files (HIGH, primary source)

<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->

## Conventions

Conventions not yet established. Will populate as patterns emerge during development.
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->

## Architecture

Architecture not yet mapped. Follow existing patterns found in the codebase.
<!-- GSD:architecture-end -->

<!-- GSD:skills-start source:skills/ -->

## Project Skills

| Skill | Description | Path |
|-------|-------------|------|
| "source-command-gsd-mempalace-recall" | "Recall decisions, patterns, and surprises from MemPalace before planning" | `.agents/skills/source-command-gsd-mempalace-recall/SKILL.md` |
| "source-command-gsd-ns-context" | "codebase intel \| map graphify docs learnings mempalace" | `.agents/skills/source-command-gsd-ns-context/SKILL.md` |
| "source-command-gsd-ns-ideate" | "exploration capture \| explore sketch spike spec capture" | `.agents/skills/source-command-gsd-ns-ideate/SKILL.md` |
| "source-command-gsd-ns-manage" | "config workspace \| workstreams thread update ship inbox" | `.agents/skills/source-command-gsd-ns-manage/SKILL.md` |
| "source-command-gsd-ns-project" | "project lifecycle \| milestones audits summary" | `.agents/skills/source-command-gsd-ns-project/SKILL.md` |
| "source-command-gsd-ns-review" | "quality gates \| code review debug audit security eval ui" | `.agents/skills/source-command-gsd-ns-review/SKILL.md` |
| "source-command-gsd-ns-workflow" | "workflow \| discuss plan execute verify phase progress" | `.agents/skills/source-command-gsd-ns-workflow/SKILL.md` |
| "source-command-gsd-review-backlog" | "Review and promote backlog items to active milestone" | `.agents/skills/source-command-gsd-review-backlog/SKILL.md` |
| "source-command-gsd-workstreams" | "Manage parallel workstreams — list, create, switch, status, progress, complete, and resume" | `.agents/skills/source-command-gsd-workstreams/SKILL.md` |
<!-- GSD:skills-end -->

<!-- GSD:workflow-start source:GSD defaults -->

## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:

- `/gsd-quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd-debug` for investigation and bug fixing
- `/gsd-execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->

<!-- GSD:profile-start -->

## Developer Profile

> Profile not yet configured. Run `/gsd-profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
