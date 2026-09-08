# Architecture Research: Wild-Diff-ICMH

**Domain:** Domain-specialized generative image compression (camera-trap wildlife) — added onto an existing working PyTorch Lightning codebase
**Researched:** 2026-09-07
**Confidence:** HIGH for all codebase claims (verified by reading the actual files line-by-line); MEDIUM for the Colab I/O quantification (web-sourced, provider-tier `websearch`)

This document supersedes Appendix B of `docs/ke-hoach-difficmh-wildlife-8-tuan.md` where the plan's assumptions diverge from what the code actually does. Every claim below cites a real file and line range from the repo as it exists today.

---

## 0. Three findings that change the plan

These were not in `docs/ke-hoach-difficmh-wildlife-8-tuan.md` and materially affect build order and risk:

1. **`train.py`'s "resume" is not PyTorch Lightning's crash-safe resume.** `train.py:73-82` loads a checkpoint via a hand-rolled `load_state_dict(model, torch.load(...), strict=False)` — this restores *weights only*. It never passes `ckpt_path=` to `trainer.fit()` (`train.py:88`), so optimizer state (both the main `AdamW` and the auxiliary entropy-bottleneck optimizer), LR scheduler position, and `global_step` are **silently reset to zero on every restart**. On a single 24h A100 box this doesn't matter. On Colab, where a session may die every few hours, this means every "resume" is actually a warm-start-from-weights, not a continuation — `max_steps: 300001` and `every_n_train_steps` counters restart from 0 each time, and Adam's momentum state is thrown away every crash. **This must be patched before any real H1 training run** (see §4).
2. **Default PyTorch Lightning checkpoints will include the frozen SD 2.1 UNet, frozen VAE, and (if TGM is enabled) frozen RAM++ weights**, because `DiffEIC` (in `model/diffeic.py`) does not override `on_save_checkpoint`/`on_load_checkpoint` to strip frozen submodules (confirmed: no such method exists anywhere in `model/*.py` or `ldm/models/diffusion/ddpm.py`). A full checkpoint is therefore several GB (frozen SD UNet ~3.4GB fp32 + frozen VAE ~0.3GB + trainable control module + codec + two optimizer states), not the few-hundred-MB delta one would expect. With `configs/train_diffeic.yaml`'s current `save_top_k: -1`, every checkpoint is kept forever — on a Colab session with limited time-per-session, writing multi-GB files to a Drive mount repeatedly is a real throughput problem, not a storage-quota problem (2TB Drive is plenty of *space*; it is not fast).
3. **The dataset and mask must share crop coordinates, and no existing utility supports this.** `utils/image/common.py:random_crop_arr` (lines 33-54) picks a random resize scale *and* a random crop offset internally, with no way to extract or inject those coordinates. Calling it twice — once for the image, once for a mask — produces two independently-cropped views of two different regions. This is the single most likely silent-correctness bug in the whole project (see §1.2).

---

## 1. Codebase Intervention Map

### 1.1 Loss composition — `model/diffeic.py`, `p_losses` (lines 934-1024)

`p_losses(self, x_start, cond, t, noise=None)` is the one function where every loss term is combined. Concretely, in call order:

| Loss term | Line(s) | What it computes | Current spatial handling | ROI-weighting injection point |
|---|---|---|---|---|
| `loss_simple` (diffusion ε-loss) | 952-961 | `get_loss(model_output, target, mean=False).mean([1,2,3])` — MSE/L1 over the UNet's noise prediction, in **noise space**, not latent-reconstruction space | Already per-pixel then uniformly averaged over `[1,2,3]` (C,H,W) | **Not the H2 target.** This is the diffusion training loss, not the codec distortion loss. Leave alone. |
| `loss_bpp` / rate loss | 963-971 | `cond['bpp']` — computed upstream in `get_input` (line 830) from the entropy-model likelihoods, already a scalar per-batch | Scalar, no spatial dimension survives | Not directly ROI-weightable without a rate-allocation term (plan's V3, out of scope for MVP) |
| **`loss_guide` — this IS `L_dist`** | 973-977 | `c_latent = cond['c_latent'][0][:,:4,:,:]` (decoded/compressed latent, 4 channels) vs `x_start` (clean VAE latent). `loss_guide = self.get_loss(c_latent, x_start)` — **`get_loss` defaults to `mean=True`**, so this call already collapses to a scalar before `p_losses` sees it. | Uniform average over B,C,H,W (8× downsample from pixels: 256px crop → 32×32 latent) | **Primary H2 injection point.** Change the call to `self.get_loss(c_latent, x_start, mean=False)` (returns `(B,4,H,W)`), then replace the `.mean()` in `p_losses` with a weighted mean using a mask `M` at the SAME 32×32 (8×) resolution: `loss_guide = (per_pixel_loss * W).sum() / (W.sum() * per_pixel_loss.shape[1])` where `W_{i,j} = 1 + (α-1)·M_{i,j}` per the plan's Eq. |
| **`loss_semantic` — this IS `L_sem`** | 980-1013 | Only computed if `self.l_semantic_weight != 0`. Pulls spatial feature maps via `self.model.diffusion_model.get_encode_features(...)` (defined in `ldm/modules/diffusionmodules/openaimodel.py:814-841`, returns `features_enc` = list of encoder block outputs, and `features_mid` = middle-block output). Selects `sl_loc` (`'mid'` → 64× downsample from pixels, i.e. 4×4 for a 256px crop; `'enc_N'` → encoder block N, e.g. `enc_9` is ~32× downsample). Calls `self.get_loss_semantic(sl_x_ori, sl_x, self.sl_metric)`. | `get_loss_semantic` (lines 1130-1148) does `F.cosine_similarity(x, y, dim=1)` → shape `(B,H,W)` (channel dim collapsed), then `loss = (1 - cos_sim); return loss.mean()` — **uniform spatial average, confirming the plan's Eq. 1 claim that the sum-over-n structure is already there.** | **Secondary H2 injection point — exactly a 1-line change** as PROJECT.md claims. In `get_loss_semantic`, accept an optional `weight_map` of shape `(B,H,W)` matching `x`/`y`'s spatial size, and replace `loss.mean()` with `(loss * weight_map).sum() / weight_map.sum()`. **Config default `sl_loc: mid` (64× downsample) must be changed to `sl_loc: enc_9` (32× downsample) for H2 to have any effect** — at `mid`, a mid-sized animal in a 256px crop occupies <1 grid cell, matching the plan's information-theoretic argument in PROJECT.md. |

Also present and directly relevant:

- **`l_semantic_weight`** — constructor param (`model/diffeic.py:609`), config default `0.0` in `configs/model/diffeic.yaml:35`. Must be set nonzero to activate `L_sem` at all; today it is effectively disabled.
- **`preprocess_semantic_model`** (line 628) and **`preprocess_tag_model`** (line 629) — instantiated from `preprocess_semantic_config`/`preprocess_tag_config` in the model config; both are `enabled: false` by default (`configs/model/diffeic.yaml:127,135`). Their outputs `c_semantic`/`c_tag` **overwrite or concatenate onto `c`** inside `get_input` (`model/diffeic.py:800-826`). This is the exact point where the conditioning text `c` — which is what H3's TGM changes — enters the training/inference pipeline. `preprocess_tag_model` is `TagGCM` (`model/lfgcm.py:689-780`); see §1.5.
- **`c_ucg_rate`** (constructor param, default `0.1`, `configs/model/diffeic.yaml:25`) — classifier-free-guidance dropout, applied at `model/diffeic.py:801-819`: with probability `c_ucg_rate` during training, `drop_cond = True` and the semantic/tag conditioning is zeroed (`c_semantic = torch.zeros_like(...)`) or blanked (`c_tag = [''] * len(c_tag)`). This is orthogonal to H2/H3 development but must not be disturbed — it's what makes CFG sampling (`c_cfg_scale: 3.0`, used in `sample_log`, lines 886-901) work at inference.

**Net effect for H2 implementation**: two call sites in `p_losses` (line 977 for `L_dist`, line 1011 inside `get_loss_semantic` for `L_sem`), plus one new constructor param (`roi_alpha`) and one new required key flowing through `cond` (the downsampled ROI mask — see §1.2 for how it gets there). No VRAM increase, no new trainable parameters, confirming the plan's characterization.

### 1.2 Dataset layer — `dataset/licdataset.py` + `dataset/data_module.py`

**Current `LICDataset.__getitem__` (`dataset/licdataset.py:30-69`):**
```python
gt_path = self.paths[index]                      # flat list, one path per line
pil_img = Image.open(gt_path).convert("RGB")
if self.crop_type == "random":
    pil_img_gt = random_crop_arr(pil_img, self.out_size)   # <-- internal random resize+crop
img_gt = (pil_img_gt / 255.0).astype(np.float32)
img_gt = augment(img_gt, hflip=self.use_hflip, rotation=self.use_rot)  # <-- also internally random
target = (img_gt * 2 - 1).astype(np.float32)      # jpg key: [-1,1], this is x_start after VAE encode
source = img_gt.astype(np.float32)                # hint key: [0,1], this is `control` fed to codec E_c
return dict(jpg=target, txt="", hint=source)
```
`self.paths = load_file_list(file_list)` (`utils/file.py:5-13`) is a flat list of image paths, one per line in a `.list` text file (e.g. `datalists/train.list`, referenced from `configs/dataset/lic_train.yaml:5`). **There is no mask, no bbox, no metadata field anywhere in this pipeline today.**

**The correctness trap, made concrete:** `utils/image/common.py:random_crop_arr` (lines 33-54) does its own `random.randrange(...)` calls for both the resize scale (`smaller_dim_size`) and the crop offset (`crop_y`, `crop_x`) *inside the function*, with no seed control and no return value exposing what it chose. If a new `MaskAwareLICDataset` calls `random_crop_arr(pil_img, ...)` for the image and separately calls it again (or any equivalent function) for the mask, **the two calls will independently pick different resize scales and different crop windows** — the returned mask will not correspond to the returned image region at all. This is a silent bug: shapes match, training runs, loss goes down, and the ROI weight map is uncorrelated noise relative to the actual animal location. It will not throw an error; it will just make H2 measure nothing.

**What a mask-aware dataset class must add** (new file, e.g. `dataset/wildlife_licdataset.py`, subclassing or replacing `LICDataset`):

1. **Factor crop-coordinate selection out of `random_crop_arr` into a shared step.** Either (a) write a new `random_crop_coords(img_size, out_size, min_frac, max_frac) -> (resize_scale, crop_y, crop_x)` that returns the chosen parameters instead of applying them, then apply the same `(resize_scale, crop_y, crop_x)` to both the image array and the mask array via a shared `apply_crop(arr, resize_scale, crop_y, crop_x, out_size, interp)` helper (nearest-neighbor interpolation for the mask, bicubic for the image); or (b) stack image and mask as an `(H, W, 4)` array (RGB + mask channel) before cropping, run the existing crop/augment pipeline once on the stacked array, then split channels back apart after. **(b) is simpler and reuses `augment()` (`utils/image/common.py:58-125`) unmodified** since `augment` already accepts and transforms an arbitrary channel-count `ndarray` — same hflip/rotation decision applies to all channels because it's one `random.random()` call shared across the whole array. This is the recommended approach: fewer new code paths, same random-state guarantee.
2. **Load the mask aligned to `gt_path`.** Simplest contract: mask files live in a parallel directory tree (`data/wildlife/masks/{split}/{same_stem}.png`, single-channel, 0/1 or 0-255), derived from the file list path by string substitution, OR a second file list (`file_list_mask`) with one-to-one line correspondence to `file_list` — the second is safer against directory-structure assumptions and matches how `load_file_list` already works (just add a second call). Return an additional dict key, e.g. `dict(jpg=target, txt="", hint=source, mask=mask_arr)`.
3. **Handle "no mask available" gracefully** (empty images, or animals outside any provided bbox) — return an all-ones or all-zeros mask consistently (all-ones = "no reweighting for this sample", NOT all-zeros, since `W = 1 + (α-1)·M` and an all-zero mask correctly degrades to uniform weighting — this is the safe default and should be preferred over skipping the sample).
4. **Downsampling the mask to latent resolution happens in the model, not the dataset.** The dataset should return the mask at full crop resolution (e.g. 256×256); `DiffEIC.get_input` (or a thin wrapper around it) is responsible for `F.interpolate(mask, size=(H_latent, W_latent), mode='nearest' or 'area')` to match `c_latent`'s 8× or `sl_loc`'s 32×/64× grid, because two different downsample factors are needed for `L_dist` vs `L_sem`, and only the model knows which `sl_loc` is active. Concretely: add a `mask` key to the dict returned from `get_input` (`model/diffeic.py:790-837`), alongside `bpp`, `q_bpp`, etc., and downsample it twice inside `p_losses` at the two injection points from §1.1.

`dataset/data_module.py` requires **no changes** — it is a generic `LightningDataModule` that instantiates whatever `dataset.target` the config points to (`instantiate_from_config`, `data_module.py:24`) and wraps it in a vanilla `DataLoader`. A new `WildlifeLICDataset` class just needs a new `target:` in a new dataset config; `DataModule` itself is dataset-agnostic by construction. **Do not modify `data_module.py`.**

`dataset/batch_transform.py` is currently a no-op passthrough (`IdentityBatchTransform`, 5 lines). If mask downsampling is instead done at the batch level (not inside the model), a `MaskDownsampleBatchTransform` could live here — but the recommendation above (downsample inside `p_losses`, since it needs `sl_loc`-dependent target resolution) is cleaner and keeps `batch_transform.py` untouched.

### 1.3 Configs layout — adding an experiment without forking

Confirmed structure: `configs/train_diffeic.yaml` (top-level: points at `data.train_config`/`data.val_config` YAML paths and `model.config` YAML path, plus PL `trainer`/`callbacks` block) → `configs/model/diffeic.yaml` (all `DiffEIC` constructor params) → `configs/dataset/lic_train.yaml` / `lic_valid.yaml` (dataset + dataloader params).

`train.py` already supports **two independent override mechanisms** without editing any YAML:
- Top-level dotlist overrides merged via `OmegaConf.merge(config, OmegaConf.from_dotlist(overrides))` (`train.py:43-46`) — covers anything under `data.*`, `model.*` (paths), `lightning.*`.
- **Model-param-specific overrides**, filtered by `overrides_model = [o.replace('model.params.', 'params.') for o in overrides if o.startswith('model.params.')]` (`train.py:53`) — covers any key inside `configs/model/diffeic.yaml`'s `params:` block, including nested ones like `model.params.preprocess_tag_config.params.enabled=true`.

**Recommendation — do not fork `configs/train_diffeic.yaml` per experiment.** Instead:
- One new dataset config per *data* variant that's structural (new `file_list`, new `out_size: 256`, new mask-aware `target:`): `configs/dataset/lic_train_wildlife.yaml`, `configs/dataset/lic_valid_wildlife.yaml`. This is unavoidable because `file_list` path and `crop_type`/`out_size` differ from the LSDIR-era defaults (currently `out_size: 512`, `crop_type: random`, batch 4 in `lic_train.yaml` vs. the plan's target of 256px crop, batch 2 + `accumulate_grad_batches: 8`).
- One new model config for genuinely new architecture surface (adding `roi_alpha`, `mask_key`, enabling `preprocess_tag_config`): `configs/model/diffeic_wildlife.yaml`, copied from `diffeic.yaml` with the new fields added. This is also somewhat unavoidable because new constructor params can't be introduced via CLI dotlist alone (the dotlist can *override* existing keys, not add brand-new ones the dataclass/constructor doesn't already accept as a name — though OmegaConf will happily add new dict keys, `DiffEIC.__init__`'s **kwargs will just silently swallow anything not explicitly named as a parameter, so new named params like `roi_alpha` must exist in the YAML from the start**, hence a config fork here is correct, not a workaround).
- For pure **scalar hyperparameter sweeps** (α ∈ {3,5,8}, λ_rate ∈ {2,8,32}, `l_semantic_weight`, `sl_loc`) — use CLI dotlist overrides against ONE base wildlife config, e.g. `python train.py --config configs/train_diffeic_wildlife.yaml model.params.roi_alpha=5 model.params.l_bpp_weight=8`. This keeps the sweep from generating a config file per run.
- One new top-level `configs/train_diffeic_wildlife.yaml` pointing at the two new dataset configs and the new model config, with `lightning.trainer.max_steps` set to the realistic Colab-scale figure (20-30K, not 300001) and `every_n_train_steps` reduced (see §4).

This gives exactly 3 new files for the whole project (`train_diffeic_wildlife.yaml`, `dataset/lic_{train,valid}_wildlife.yaml` — arguably 2 — and `model/diffeic_wildlife.yaml`), plus CLI overrides for everything else. **This is the concrete answer to "how should a new config be added without forking the existing ones"** — fork only the axes that need genuinely new keys or genuinely new file paths; override everything else.

### 1.4 Eval plumbing — `utils/metrics.py`, `inference.py`

**What already exists:**
- `utils/metrics.py`: `calculate_psnr_pt` (Y-channel aware, BasicSR-style), a hand-written `LPIPS` wrapper class around the `lpips` package, and standalone `compute_psnr`/`compute_ssim` functions. These are generic image-quality metrics, domain-agnostic.
- `model/diffeic.py`'s `calculate_metrics` config block (`configs/model/diffeic.yaml:137-149`) wires arbitrary `pyiqa` metrics (currently `psnr`, `ms_ssim`, `lpips`) into `self.metric_funcs` (`DiffEIC.__init__`, lines 653-659), consumed inside `validation_step`/`validation_epoch_end` (lines 1057-1092) — this is a **PyTorch-Lightning-internal validation loop**, running during training, one image at a time (`data_loader.batch_size: 1` in `lic_valid.yaml`), logging scalar averages to the PL logger. It writes per-checkpoint sample PNGs to `{logger.save_dir}/validation/{global_step}/{batch_idx}.png` (lines 1063-1071) but **does not write anything to a `results.jsonl`-style file, and does not run any downstream task model (detection/species/segmentation).**
- `inference.py`: a standalone script (`process()` function, lines 22-83) that takes a directory of images, compresses+decompresses each one via `apply_condition_compress`/`apply_condition_decompress` (the actual bitstream round-trip, not the training-time approximation), and writes decoded PNGs + per-image bpp. This is the correct tool to *produce* decoded images for external eval, but it computes no quality/task metrics itself — it only prints `avg_bpp`.

**What must be built (net-new, confirmed absent from the codebase):**
- Detection mAP (incl. `AP_s/AP_m/AP_l`), species top-1/top-5, segmentation mIoU — none of MegaDetector, SpeciesNet, or SAM-as-evaluator exist anywhere in this repo. These are external pretrained task models the plan calls for, wired into a new `eval/eval_harness.py` that: (1) calls `inference.py`-equivalent logic to produce decoded images at a given bitrate/config, (2) runs each task model on original vs. decoded images, (3) appends rows to `results.jsonl`.
- The `results.jsonl` writer/schema itself (§3).
- Day/night (`illumination`) stratification of any metric — not present in `validation_step`; must be added as a groupby dimension in the new harness, driven by metadata carried alongside each image (see §2).
- `inference.py` should be extended (or wrapped, not forked) to also accept/emit a `tag_ids` argument when `preprocess_tag_model.enabled` — it already threads `tag_ids` through `apply_condition_compress`/`decompress` at the `DiffEIC` level (`model/diffeic.py:732-788`), but `inference.py`'s own `process()` function (lines 22-83) never passes tags in; this must be extended when H3 lands, not before.

### 1.5 TGM / tag pathway — `src/recognize-anything/` and how tags reach `c`

`preprocess_tag_model` is `TagGCM` (`model/lfgcm.py:689-780`). Concretely:

1. **Model.** `self.model = ram_plus(pretrained=..., image_size=384, vit='swin_l')` (line 703) — loads RAM++ from `src/recognize-anything/ram/models/ram_plus.py`, weights frozen (`requires_grad_(False)`, lines 705-706).
2. **Tag extraction.** `extract_tag()` (lines 715-745) resizes input to 384×384, normalizes with ImageNet stats, then calls **`self.model.generate_index(x)`** (returns per-image lists of tag IDs from RAM++'s fixed 4,585-tag vocabulary — the vocab file is `src/recognize-anything/ram/data/ram_tag_list.txt`) followed by **`self.model.index2tag(indexs)`** which converts IDs back to a comma-joined tag string per image.
3. **Bit accounting.** `bits = 13 * n_all_indexs` (line 743) — hard-coded 13-bit fixed-length code, matching PROJECT.md's "4585 tags fit in 13 bits (8192 cap)" observation. This bit count feeds `tag_bpp` back in `DiffEIC.get_input` (line 833) and into the logged/optimized `total_bpp` (line 971).
4. **Reaching `c`.** Back in `DiffEIC.get_input` (`model/diffeic.py:814-826`): `c_tag, bits_tag = self.preprocess_tag_model(control)` produces the **tag string list**, not embeddings; `c_tag = self.cond_stage_model.encode(c_tag)` runs it through the **frozen OpenCLIP text encoder** (`FrozenOpenCLIPEmbedder`, `configs/model/diffeic.yaml:105-109`) to get the actual cross-attention conditioning tensor. If `preprocess_semantic_model` is also enabled, `c = torch.cat([c, c_tag], 1)` (concatenate along the token dimension); otherwise `c = c_tag` replaces `c` outright. This is a **fixed function of the model's own vocabulary and its own generic-image RAM++ weights** — there is no hook here for injecting metadata (`illumination`/`habitat`/`season`) that doesn't come from the image itself.

**Where H3's L1 (restricted vocab) and L2 (structured attributes) actually plug in — concretely, two options, both viable without touching RAM++ weights:**
- **Cheapest (recommended for L1+L2 MVP): bypass `TagGCM.extract_tag`'s tag-string output entirely.** Write a new preprocessing function (not a `nn.Module`, since it needs no gradient) that: (a) optionally still calls `self.model.generate_index(x)` and filters the returned IDs against a curated wildlife-relevant subset of `ram_tag_list.txt` (L1: ~256 of 4,585 IDs — cutting bit cost from 13 to 8 bits/tag as planned), then (b) appends deterministic, non-predicted structured attribute strings (illumination from grayscale-channel-variance check, season/hour from EXIF timestamp already present in the raw file before any cropping, habitat from a site_id → habitat lookup table built once in `dataset_card.md`) directly into the tag string list *before* it's passed to `self.cond_stage_model.encode(...)`. This requires **zero changes to `model/lfgcm.py`'s `TagGCM` class** — it's a wrapper function called from `DiffEIC.get_input` (or from `preprocess_tag_model.forward`, which is the more contained place — the L1/L2 filter naturally belongs as a post-processing step added right after line 743 in `extract_tag()`). **Because L1/L2 need no training** (per PROJECT.md, this is the plan's own quick-win insight, confirmed structurally correct here — swapping the text string changes the CLIP conditioning at *inference/decode* time only), this can be validated with the pretrained checkpoint before any GPU training run.
- **More invasive (L3, out of MVP scope): coarse spatial grid + counts.** Requires the control module to learn to read a new prompt grammar it wasn't trained on — this is a training-path change (touches `configure_optimizers`, needs gradient flow), correctly flagged by the plan as needing a real training run.

**Metadata must be threaded through the dataset, not invented at eval time.** For L2 to work at *training* time (not just at inference-time quick-win experiments), `illumination`/`habitat`/`season` need to reach `DiffEIC.get_input` per-sample, meaning the mask-aware dataset class from §1.2 should also carry a `meta` dict (or several new keys: `illumination`, `habitat_id`, `season`) alongside `mask`, sourced at dataset-construction time from `dataset_card.md`'s site→habitat table and each image's EXIF, NOT re-derived on the fly inside the training loop (EXIF reads and grayscale checks on every `__getitem__` call are cheap, but computing them once during T1.1's data-normalization pass and storing them in the COCO-style annotation JSON is more robust and reusable by the eval harness too).

---

## 2. Component Boundaries for the New Work

```
                    ┌────────────────────────────────────────────────────────┐
                    │                    data/ (raw + processed)              │
                    │  wildlife/{images,masks}/ · splits/ · annotations/      │
                    └───────────────┬──────────────────────────────────────┬─┘
                                    │ read                                 │ read
                                    ▼                                      ▼
          ┌─────────────────────────────────┐              ┌─────────────────────────────┐
          │  src/data/                       │              │  src/masks/                  │
          │  acquire.py, normalize.py,       │─produces────▶│  sam_mask_gen.py              │
          │  split.py, split_check.py ⭐     │  bbox coco   │  (SAM ViT-H, box-prompted)    │
          └───────────────┬───────────────────┘              └───────────────┬──────────────┘
                          │ splits/*.txt, location_split.json                │ masks_pseudo_gt/, roi_masks/
                          ▼                                                  ▼
          ┌─────────────────────────────────────────────────────────────────────────┐
          │  dataset/wildlife_licdataset.py  (NEW, subclasses/parallels LICDataset)  │
          │  returns dict(jpg, txt, hint, mask, illumination, habitat_id, season)    │
          └───────────────────────────────┬─────────────────────────────────────────┘
                                           │ DataLoader batches
                                           ▼
          ┌─────────────────────────────────────────────────────────────────────────┐
          │  model/diffeic.py  DiffEIC.get_input / p_losses   (MODIFIED, not new)    │
          │  + src/losses/roi_loss.py (weighted-mean helper, imported by p_losses)   │
          │  + model/lfgcm.py TagGCM.extract_tag (MODIFIED: L1 filter + L2 append)   │
          │  + src/tgm/vocab.py (curated wildlife tag ID subset + habitat lookup)    │
          └───────────────────────────────┬─────────────────────────────────────────┘
                                           │ checkpoints/{exp_id}/*.ckpt + config.yaml
                                           ▼
          ┌─────────────────────────────────────────────────────────────────────────┐
          │  inference.py  (MODIFIED: thread tag_ids, batch mode over a split)       │
          └───────────────────────────────┬─────────────────────────────────────────┘
                                           │ decoded/{exp_id}/{dataset}/lambda{λ}/*.png + bpp
                                           ▼
          ┌─────────────────────────────────────────────────────────────────────────┐
          │  src/eval/eval_harness.py (NEW)                                          │
          │  wraps MegaDetector / SpeciesNet / SAM-as-evaluator / MegaDescriptor,    │
          │  perceptual metrics from utils/metrics.py, stratifies by illumination    │
          └───────────────────────────────┬─────────────────────────────────────────┘
                                           │ append-only rows
                                           ▼
          ┌─────────────────────────────────────────────────────────────────────────┐
          │  results.jsonl  ⭐ single source of truth                                │
          └───────────────────────────────┬─────────────────────────────────────────┘
                                           │ read-only
                                           ▼
          ┌─────────────────────────────────────────────────────────────────────────┐
          │  make_all_figures.py + figures/  (NEW)                                  │
          └─────────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Talks to | Owned by (per PROJECT.md's 2-person split) |
|---|---|---|---|
| `src/data/acquire.py` + `normalize.py` | Download LILA BC Snapshot Serengeti + CCT, normalize species labels to 10-12 groups, crop metadata banners, compute per-site `habitat_id` and per-image `illumination`/`season` from EXIF | Writes `data/wildlife/`, `annotations/*_coco.json` | Data & Eval lead |
| `src/data/split.py` + `split_check.py` ⭐ | Site/burst-aware train/val split; **`split_check.py` is a hard assertion** (`assert set(train_sites) & set(val_sites) == set()`, plus a burst/`seq_id` variant), must run in CI or as a pre-training gate, not a manual step | Writes `splits/*.txt`, `splits/location_split.json` | Data & Eval lead |
| `src/masks/sam_mask_gen.py` | Run SAM ViT-H with bbox-prompt over val split (pseudo-GT) and over train split (ROI masks for H2) | Reads `annotations/*_coco.json`, writes `data/wildlife/masks/` | Data & Eval lead |
| `dataset/wildlife_licdataset.py` | Single point where image+mask+metadata become one aligned, augmented training sample. **Owns the crop-coordinate-sharing fix from §1.2.** | Reads `data/wildlife/{images,masks}/`, `splits/*.txt`; feeds `dataset/data_module.py` (unmodified) | Model & Training lead (it's on the training critical path even though data-shaped) |
| `src/losses/roi_loss.py` | Pure function(s): `weighted_mse(pred, target, weight_map)`, `weighted_cosine(x, y, weight_map)`. No PL/model dependency — testable in isolation. Imported into `model/diffeic.py::p_losses`. | Called by `model/diffeic.py` only | Model & Training lead |
| `src/tgm/vocab.py` + edits to `model/lfgcm.py::TagGCM` | Curated wildlife tag-ID allowlist (L1), habitat/illumination/season → text fragment templates (L2) | Reads `ram_tag_list.txt`, `annotations/*_coco.json` (habitat lookup); modifies `TagGCM.extract_tag` output | Data & Eval lead builds vocab/lookup; Model & Training lead wires it into the training path |
| `src/eval/eval_harness.py` | Batch-decode a split at a given checkpoint+λ, run task models, stratify by illumination, append rows to `results.jsonl` | Reads `checkpoints/{exp_id}/`, `data/wildlife/`; writes `decoded/`, `results.jsonl` | Data & Eval lead |
| `results.jsonl` | Append-only ledger, one JSON object per (exp_id, dataset, metric) row. **No other file may hold ground-truth numbers.** | Written by `eval_harness.py`; read by `make_all_figures.py` and the report | Both (contract, not owned) |
| `make_all_figures.py` | Deterministic figure/table generation from `results.jsonl` only — no hardcoded numbers anywhere | Reads `results.jsonl` only | Data & Eval lead |

**Boundary rule that matters most for parallel work:** `dataset/wildlife_licdataset.py` sits at the seam between the two people. It is filed under "Model & Training" ownership above (not Data & Eval) because it is imported directly by the training loop and any bug in it blocks GPU time — but its **inputs** (`data/wildlife/{images,masks}/`, `splits/*.txt`) are entirely owned and populated by Data & Eval. This is why the directory contract in §3 matters more than who-writes-which-.py-file.

---

## 3. The Two-Stream Contract

Three directory/schema contracts, verified against and reconciled with `docs/ke-hoach-difficmh-wildlife-8-tuan.md` §3.1/§3.7/Appendix B (the plan's own GD1-GD3):

### GD1 — Data directory layout (TV-A/Data-Eval writes → TV-B/Model-Training reads)

```
data/
└── wildlife/
    ├── images/{split}/{site_id}/{image_id}.jpg      # normalized, banner-cropped
    ├── masks/{split}/{site_id}/{image_id}.png        # SAM ROI mask, single channel, aligned filename to images/
    ├── splits/
    │   ├── train_wild.txt                            # one relative image path per line — same format load_file_list already expects
    │   ├── val_ss_1k.txt
    │   ├── val_cct_500.txt
    │   └── location_split.json                       # {"train_sites": [...], "val_sites": [...]} — split_check.py reads this
    ├── annotations/
    │   ├── train_wild_coco.json                       # COCO format: images/annotations/categories + custom fields per image:
    │   │                                                #   "illumination": "day"|"night"|"twilight", "habitat_id": int,
    │   │                                                #   "season": "wet"|"dry", "site_id": str, "seq_id": str
    │   ├── val_ss_1k_coco.json
    │   └── val_cct_500_coco.json
    ├── masks_pseudo_gt/{image_id}.png                 # val_cct_500 only, SAM box-prompt pseudo-GT (segmentation reference)
    └── dataset_card.md                                 # stats, species→group mapping, habitat lookup table
```
**Contract detail that must not drift:** the mask filename for `images/{split}/{site_id}/{image_id}.jpg` MUST be `masks/{split}/{site_id}/{image_id}.png` — identical relative path, different root and extension. This lets `WildlifeLICDataset` derive the mask path by string substitution instead of needing a second file-list file, which is simpler to keep in sync than two independently-ordered lists.

### GD3 — Checkpoint directory layout (TV-B/Model-Training writes → TV-A/Data-Eval reads)

```
checkpoints/
└── {exp_id}/                        # e.g. exp_h1_lambda8, exp_h2_alpha5, exp_h3_l2
    ├── config.yaml                  # full merged OmegaConf dump — train.py already does this at save_dir (train.py:61-71), just point default_root_dir here
    ├── config_model.yaml            # ditto for the model sub-config
    ├── step={N}.ckpt                # PL ModelCheckpoint output, filename pattern from configs/train_diffeic.yaml:52
    ├── last.ckpt                    # ALWAYS the most recent — see §4 for why this must be a real Lightning ckpt_path-restorable file
    └── STATUS.json                  # {"exp_id", "last_step", "last_updated_utc", "status": "running"|"done"|"crashed"} — NEW, cheap, lets TV-A poll without opening a 6GB .ckpt
```
`config.yaml`/`config_model.yaml` are **already produced automatically** by `train.py:61-71` at `config.lightning.trainer.default_root_dir` — the only change needed is making `default_root_dir` point at `checkpoints/{exp_id}/` per run (currently `./logs/debug`), which is a one-line override (`lightning.trainer.default_root_dir=checkpoints/exp_h1_lambda8`) requiring no code change. `STATUS.json` is new and cheap — write it from a small custom PL callback (`on_train_batch_end`, alongside the existing `ModelCheckpoint`/`ImageLogger` in `model/callbacks.py`) so the Data & Eval lead's eval harness can poll for new checkpoints without needing to load them.

### GD2 — `results.jsonl` schema (both write → `make_all_figures.py` + report read)

The plan has two slightly different versions of this schema in two places (PROJECT.md's Key Decisions table: `{exp_id, dataset, lambda_rate, metric, value, n_images, git_commit, date}`; the plan doc's §3.7 GD2: adds `illumination, ddim_steps`). **Reconcile by using the superset** — dropping fields is always safe for a JSON-lines ledger, omitting needed ones is not:

```json
{"exp_id": "exp_h2_alpha5", "dataset": "val_ss_1k", "split_subset": "day", "illumination": "day", "lambda_rate": 8, "roi_alpha": 5, "ddim_steps": 50, "metric": "mAP", "metric_subtype": "AP_medium", "value": 0.412, "n_images": 1000, "git_commit": "a1b2c3d", "date": "2026-09-21T14:03:00Z"}
```
Recommended additions beyond both plan versions, based on the concrete pitfalls surfaced above: `roi_alpha` (H2's α, null for non-H2 runs) and `tgm_tier` (`"none"|"L1"|"L2"|"L3"`, null for non-H3 runs) as explicit columns rather than folding them into `exp_id` string parsing — `make_all_figures.py` should never need to regex an experiment name to recover a hyperparameter. One row per `(exp_id, dataset, split_subset, metric, metric_subtype)` tuple; **append-only, never edited in place** — a corrected number is a new row with a later `date`, and figures always take the latest row per key.

---

## 4. Colab-Adapted Execution Architecture

This is the part with no equivalent in the source plan (written for a dedicated 24/7 GPU box) and the part most likely to silently waste paid compute units if skipped.

### 4.1 The resume-semantics gap (finding #1 from §0) — must be patched first

Concretely, `train.py` needs a **third** resume mode, distinct from the two that already exist:

| Existing mechanism | Code | What it restores | When to use |
|---|---|---|---|
| `config.model.resume` (full weights) | `train.py:73-82`, `load_state_dict(model, torch.load(...)['state_dict'], strict=False)` | Model weights only | Bootstrapping a new experiment from the author's pretrained checkpoint, or from a prior H1 run's final weights when starting H2 |
| `--resume_codec` flag | `train.py:75-79`, filters `state_dict` to keys containing `'preprocess_model'` | Codec (`E_c`/`D_c`) weights only | Swapping in a different control-module/TGM config while keeping a validated codec |
| **NEW — needed for Colab crash-resume** | Not present. Requires adding `--ckpt_path` arg, and `trainer.fit(model, datamodule=data_module, ckpt_path=args.ckpt_path)` at `train.py:88` | **Everything**: weights, both optimizers' state (main `AdamW` + `aux_opt` for the entropy-bottleneck quantiles), LR scheduler position, `global_step`, epoch count, all callback states (so `ModelCheckpoint`'s own step counter doesn't reset either) | Every session restart mid-run, after a crash or the ~12h Colab wall clock |

The fix is small (add one CLI arg, one conditional branch, pass it through to `trainer.fit`), but it is a **code change to the training entrypoint that must land before the first real (non-smoke-test) training run**, i.e. it belongs in the same work item as T1.3's "khảo cổ `train.py`" archaeology task, not deferred to whenever a crash first happens in Week 3. Detecting this gap only after losing several compute-unit-hours of an H1 run is exactly the kind of preventable loss the plan's own compute-scarcity framing (PROJECT.md Constraints: "~50h L4 for the whole project") should be defending against.

### 4.2 Checkpoint size / cadence — must patch `on_save_checkpoint` before it matters

Because `DiffEIC` saves its *entire* `state_dict()` by default (finding #2, §0), and the frozen SD 2.1 UNet + VAE + (if enabled) RAM++ dwarf the trainable control module + codec in parameter count, every checkpoint written today would be several GB. Two independent fixes, both cheap, both should land alongside the resume-semantics fix:

1. **Strip frozen submodules from saved checkpoints.** Override `on_save_checkpoint(self, checkpoint)` in `DiffEIC` to delete `state_dict` keys prefixed with `model.diffusion_model.` (except `output_blocks.`/`out.` if `sd_locked=False`), `first_stage_model.`, `cond_stage_model.`, and `preprocess_tag_model.model.` — these are reloadable from `sync_path`/RAM++'s own pretrained checkpoint at load time, not from this run's checkpoint. Correspondingly override `on_load_checkpoint` to re-merge them back in from the frozen source before `load_state_dict`. This is the single highest-leverage change for making checkpointing Colab-viable — it turns a multi-GB write into (roughly) a few-hundred-MB write, dominated by the control module (`control_model_ratio: 0.2` of SD's 320-channel base) and the codec.
2. **Checkpoint cadence.** `configs/train_diffeic.yaml:50` currently sets `every_n_train_steps: 10000` — appropriate for a run expected to complete uninterrupted on a dedicated box, wrong for a budget where a session can vanish at any point within its ~12h window and where the *total* project budget is ~50h L4-equivalent (PROJECT.md Constraints). Recommend **every 500-1000 steps** once checkpoints are small (fix #1 above) — at ~50h/project ÷ 20-30K iterations/run (plan's own per-run estimate), losing 500-1000 steps of forward/backward compute to a mid-session crash is a small, bounded, acceptable loss; losing 10,000 steps (a third to a half of an entire run) is not. Set `save_top_k` to a small rotating number (e.g. `3`) plus keep `last.ckpt` always current, rather than `-1` (keep-everything) — with small per-checkpoint size this is a minor concern, but Drive write-count itself (not just bytes) has overhead per the I/O findings in §4.3, so fewer, more meaningful checkpoints written more often is still better than "keep every one forever."

### 4.3 Dataset placement: Drive vs. local Colab disk — quantified

Web research (provider tier: `websearch`, confidence MEDIUM) confirms the concern raised in the question is real and severe, not theoretical: users report Google-Drive-mounted random reads of many small files dropping from **~300 samples/sec to ~1 sample/sec — a ~300× slowdown** — versus reading the same files packed sequentially [Working with huge datasets, 800K+ files in Google Colab and Google Drive](https://satyajitghana.medium.com/working-with-huge-datasets-800k-files-in-google-colab-and-google-drive-bcb175c79477), and the pattern repeats across multiple independent reports of the same failure mode [Insanely slow data reading from google drive #1691](https://github.com/googlecolab/colabtools/issues/1691), [Access of files on google drive too slow #4692](https://github.com/googlecolab/colabtools/issues/4692). WebDataset's own documentation and community benchmarks report **3-10× higher I/O throughput from sequential tar-shard reads vs. random access** even on local disk, before Drive's mount-layer overhead is added on top [webdataset (PyPI)](https://pypi.org/project/webdataset/).

Given `LICDataset.__getitem__` does one `Image.open(gt_path)` per sample from `self.paths` (a flat file list, `dataset/licdataset.py:33-42`) — i.e. exactly the "many small files, random-order `DataLoader` access" pattern that triggers the worst-case slowdown reported above — **do not train directly against a Drive-mounted `data/wildlife/images/` tree.** Concrete recommendation:

1. **Canonical dataset storage: Google Drive**, as flat per-split directories (the GD1 layout in §3) — Drive's 2TB is not the bottleneck (PROJECT.md Constraints already notes storage is not a constraint), *sequential* write/read of it is fine.
2. **At the start of every Colab session** (not once — Colab's local disk is ephemeral and wiped between sessions): copy the *training subset actually needed for this run* (PROJECT.md's own decision: "train on a sampled subset, not the full 60K corpus") from Drive to local `/content/data/` as **pre-packed shards**, not as loose files. Two options, in order of recommendation:
   - **Single `tar`/`zip` archive per split**, copied once (`cp` or `rsync`, sequential large-file transfer — fast even over the Drive mount) then extracted locally, OR read directly from the archive without extracting (`zipfile`/`tarfile` random access into an already-local archive is fast; it's the *many-small-files-over-the-network-mount* pattern that's slow, not archives per se). This requires the least new code: `WildlifeLICDataset` can keep using `Image.open(path)`-style access once `path` points at the local extracted copy.
   - **WebDataset shards** (`.tar` shards of ~1-2GB each, sequential-read `IterableDataset`) if random-access `Dataset.__getitem__` semantics turn out to be too slow even against local disk at the target throughput — this is a larger refactor (WebDataset's iteration model doesn't map 1:1 onto `LICDataset`'s index-based `__getitem__`/`__len__`) and should be treated as a fallback, not the default, given the added engineering cost against an already compute-constrained timeline.
   - **Recommendation for this project's scale** (a sampled training subset, not the full 60K corpus, per PROJECT.md's own decision): a single zip/tar-per-split copied to local disk at session start is very likely sufficient and is far cheaper to implement than a WebDataset refactor — reserve WebDataset shards as the escalation path only if measured local-disk throughput (T2.3's own "đo throughput thật" — measure real throughput — task) shows the simple copy isn't enough.
3. **Checkpoints stay on Drive**, written directly (not staged locally then copied) — per §4.2 these are now small and written infrequently, and unlike the dataset they are *written sequentially* (one file, whole-file write) rather than randomly read, which is the Drive-mount access pattern that performs acceptably.
4. **Session-start script** (a `!bash` cell or a `setup_session.sh`, one new small file, not a component boundary concern) should: mount Drive, `cp`/`rsync` the packed dataset shard(s) for the active `exp_id`'s data config to `/content/data/`, verify checksum/count against a manifest (`data/wildlife/splits/*.txt` line count) before starting `train.py`, then invoke `train.py` with `--ckpt_path` pointed at `checkpoints/{exp_id}/last.ckpt` on Drive if it exists (§4.1), else the fresh-start `config.model.resume` path.

---

## 5. Suggested Build Order

Numbering matches PROJECT.md's Active requirements where applicable; "critical path" = blocks GPU work; "parallel-safe" = can be built while GPU work runs on the other stream.

```
[0] Patch train.py resume semantics (§4.1) + DiffEIC.on_save/on_load_checkpoint (§4.2)
      ── CRITICAL PATH, must land before ANY real training run, including smoke test ──
      no dependencies · ~0.5-1 day · Model & Training lead
        │
        ▼
[1] Site/burst-aware split + split_check.py assertion (§2, §3 GD1)
      ── CRITICAL PATH for data ──
      no dependencies · Data & Eval lead
        │
        ├──▶ [2] SAM mask generation (val pseudo-GT + train ROI masks)
        │        depends on [1] (needs bbox annotations + split boundaries)
        │        parallel-safe once [1] lands · Data & Eval lead
        │
        ├──▶ [3] WildlifeLICDataset (mask-aligned crop, §1.2's shared-crop-coords fix)
        │        depends on [1] (needs splits/file lists) + [2] (needs mask files to exist,
        │        though can be stubbed with dummy masks for early dev)
        │        CRITICAL PATH — training cannot start without this · Model & Training lead
        │        (owns the code, but blocked on Data & Eval's outputs)
        │
        └──▶ [4] Eval harness skeleton + results.jsonl writer + task model envs (§1.4, §3 GD2)
                 depends on [1] only (needs a val split to point at)
                 parallel-safe — can run against ORIGINAL images before any training
                 exists, producing the "trần trên" (upper-bound) numbers PROJECT.md
                 calls for · Data & Eval lead
        │
        ▼
[5] Session-start / Drive-shard-copy script (§4.3) + smoke-test training run (2K iters)
      depends on [0] + [3]
      CRITICAL PATH — this is the go/no-go gate the plan calls T2.3
        │
        ▼
[6] H1 fine-tuning run
      depends on [5] passing
      CRITICAL PATH, GPU-bound
        │
        ├──▶ [7] src/losses/roi_loss.py + p_losses injection (§1.1)          ─┐
        │        depends on [3] (mask must be flowing through the dataset)   │  can be
        │        parallel-safe: pure functions, unit-testable without a GPU  │  BUILT while
        │        or a running training job                                  │  [6] runs on
        │                                                                     │  GPU, since
        ├──▶ [8] TGM L1/L2 vocab + habitat lookup + TagGCM.extract_tag edit  │  it's CPU-only
        │        depends on [1] (habitat lookup needs site metadata)         │  code, but the
        │        parallel-safe: L1/L2 need NO training (§1.5) — can be       │  actual H2/H3
        │        validated against the pretrained checkpoint immediately,   │  RUNS still
        │        even before [6] finishes                                    │  need GPU time
        │                                                                     │  after [6]
        ▼                                                                    ─┘
[9] H2 run (uses [7], built on top of [6]'s checkpoint)
[10] H3 run (uses [8], built on top of [6]'s or [9]'s checkpoint)
        │
        ▼
[11] eval_harness.py full task-model wiring (extends [4] with MegaDetector/
      SpeciesNet/SAM-as-evaluator/MegaDescriptor) — needed to score [6]/[9]/[10]'s
      outputs, but the SKELETON from [4] should exist much earlier so the
      "trần trên" baseline numbers aren't blocked on task-model integration effort
        │
        ▼
[12] make_all_figures.py — depends only on results.jsonl existing with real rows;
      build the script early (against fake/sparse rows) so it's never a last-week task
```

**What can be built while GPU work runs (parallel-safe, no GPU needed):** `split_check.py` and the split itself [1]; `roi_loss.py`'s pure functions [7]; the TGM vocab curation and habitat lookup table [8] (validated cheaply via CPU-only inference against the tiny pretrained checkpoint, not full training); `eval_harness.py`'s non-task-model plumbing (file I/O, `results.jsonl` writer, stratification logic) [4]; `make_all_figures.py` [12]. This matches the plan's own "nguyên tắc phân công" (assignment principle) in §3.1 of the source doc, which already identifies this as the fix for the original single-threaded Week-1→Week-2→Week-3 handoff — the addition here is pinning the *exact* files/functions each parallel-safe item touches, so "eval bất đồng bộ từng checkpoint" (async eval per checkpoint) has a concrete `STATUS.json`-polling mechanism (§3 GD3) instead of a manual message-passing habit.

**Everything under §4 (train.py resume patch, checkpoint-stripping override, session-start script) is [0] — it is the single highest-priority code change in the entire project, because every other GPU-bound item ([6], [9], [10]) inherits its correctness (or lack thereof) from it, and the failure mode (silently discarded optimizer state / oversized checkpoints choking a Colab session) is invisible until it has already cost real, unrecoverable compute-unit budget.**

---

## 6. Anti-Patterns to Avoid

### Anti-Pattern 1: Calling crop/augment functions independently on image and mask
**What people do:** subclass `LICDataset`, add a `mask` key, call `random_crop_arr(mask_pil, out_size)` right after calling it on the image.
**Why it's wrong:** `random_crop_arr` (`utils/image/common.py:33-54`) makes its own internal random choices for resize scale and crop offset on every call — two calls never agree. Confirmed by reading the function; no shared-seed or shared-state mechanism exists.
**Do this instead:** stack image+mask into one multi-channel array before cropping (§1.2, option (b)), or factor crop-coordinate selection into a function that returns coordinates for reuse.

### Anti-Pattern 2: Treating `config.model.resume` as crash-safe
**What people do:** assume `train.py --config X model.resume=checkpoints/exp/last.ckpt` after a Colab disconnect is equivalent to Lightning's official resume.
**Why it's wrong:** it's a manual `state_dict`-only load (`train.py:73-82`); optimizer momentum, LR schedule position, and `global_step` all reset. Over many short Colab sessions this silently degrades training dynamics and miscounts progress against `max_steps`.
**Do this instead:** add and use a real `--ckpt_path` flag wired to `trainer.fit(ckpt_path=...)` (§4.1).

### Anti-Pattern 3: Training directly against a Drive-mounted flat-file dataset
**What people do:** point `file_list` at paths under `/content/drive/MyDrive/.../images/`, run `train.py` unmodified.
**Why it's wrong:** `LICDataset.__getitem__` does one `Image.open()` per sample in `DataLoader`-shuffled (i.e. effectively random) order — exactly the access pattern reported to cause ~300× slowdowns on Drive mounts (§4.3).
**Do this instead:** copy a packed archive of the active training subset to local Colab disk at session start; train against the local copy.

### Anti-Pattern 4: ROI-weighting `L_sem` at the middle block
**What people do:** enable `l_semantic_weight` with the config default `sl_loc: mid` and apply the mask there because it's the default.
**Why it's wrong:** middle block is 64× downsampled from pixels — for a 256px training crop that's a 4×4 grid; most animals occupy well under one cell (confirmed by the plan's own Nyquist-style argument, and structurally consistent with `get_encode_features`'s `features_mid` shape).
**Do this instead:** set `sl_loc: enc_9` (32× downsample, one of `enc_1..enc_12` per the `sl_loc` assertion at `model/diffeic.py:648`) before applying ROI weighting to `L_sem`.

### Anti-Pattern 5: Letting `results.jsonl` rows be edited in place or letting figures read anything else
**What people do:** under deadline pressure, hand-edit a wrong number in `results.jsonl`, or hardcode a table value directly into the report because "it's just one number."
**Why it's wrong:** this is precisely the failure mode PROJECT.md's Key Decision "results.jsonl là nguồn chân lý duy nhất" exists to prevent — it silently reintroduces the hardcode-vs-figure drift risk the schema was designed to close.
**Do this instead:** append a new row with a later `date`; `make_all_figures.py` always takes latest-by-key.

---

## 7. Sources

**Codebase (HIGH confidence — read directly):**
- `D:/Diff_ICMH/model/diffeic.py` (full file, 1150 lines)
- `D:/Diff_ICMH/model/lfgcm.py` (SFGCM/TagGCM classes, lines 560-790)
- `D:/Diff_ICMH/model/callbacks.py`
- `D:/Diff_ICMH/dataset/licdataset.py`, `D:/Diff_ICMH/dataset/data_module.py`, `D:/Diff_ICMH/dataset/batch_transform.py`
- `D:/Diff_ICMH/utils/image/common.py` (`random_crop_arr`, `center_crop_arr`, `augment`, `pad`)
- `D:/Diff_ICMH/utils/metrics.py`, `D:/Diff_ICMH/utils/common.py`, `D:/Diff_ICMH/utils/file.py`
- `D:/Diff_ICMH/inference.py`, `D:/Diff_ICMH/train.py`
- `D:/Diff_ICMH/ldm/models/diffusion/ddpm.py` (`get_loss`), `D:/Diff_ICMH/ldm/modules/diffusionmodules/openaimodel.py` (`get_encode_features`)
- `D:/Diff_ICMH/configs/train_diffeic.yaml`, `D:/Diff_ICMH/configs/model/diffeic.yaml`, `D:/Diff_ICMH/configs/dataset/lic_train.yaml`, `D:/Diff_ICMH/configs/dataset/lic_valid.yaml`
- `D:/Diff_ICMH/src/recognize-anything/ram_utils.py` and directory listing of `src/recognize-anything/`

**Project plan (HIGH confidence — provided source doc, cross-checked against code):**
- `D:/Diff_ICMH/docs/ke-hoach-difficmh-wildlife-8-tuan.md` — §1 (intervention-point analysis), §3.1/3.7 (assignment principle, RACI, three-contract system), Appendix B (proposed repo structure), Appendix C (48h checklist)
- `D:/Diff_ICMH/.planning/PROJECT.md`

**Web (MEDIUM confidence, `websearch` provider tier — used only for the Colab I/O quantification, not for codebase claims):**
- [Working with huge datasets, 800K+ files in Google Colab and Google Drive](https://satyajitghana.medium.com/working-with-huge-datasets-800k-files-in-google-colab-and-google-drive-bcb175c79477) — ~300 samples/sec → ~1 sample/sec Drive random-read slowdown
- [Insanely slow data reading from google drive · Issue #1691 · googlecolab/colabtools](https://github.com/googlecolab/colabtools/issues/1691)
- [Access of files on google drive too slow · Issue #4692 · googlecolab/colabtools](https://github.com/googlecolab/colabtools/issues/4692)
- [webdataset (PyPI)](https://pypi.org/project/webdataset/) — 3-10× sequential vs. random I/O throughput claim
- [Saving and loading checkpoints (basic) — PyTorch Lightning docs](https://lightning.ai/docs/pytorch/stable/common/checkpointing_basic.html) — `ckpt_path=` resume semantics (weights + optimizer + scheduler + epoch/step state)

---
*Architecture research for: Wild-Diff-ICMH*
*Researched: 2026-09-07*
