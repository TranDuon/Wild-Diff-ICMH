---
last_mapped_commit: working-tree
---
# External Integrations

**Analysis Date:** 2026-09-24

## Model Hubs

- Hugging Face repository `RuoyuFeng/Diff-ICMH` hosts author checkpoints.
- `Manojb/stable-diffusion-2-1-base` hosts the SD 2.1 checkpoint.
- `xinyu1205/recognize-anything-plus-model` hosts RAM++ weights.
- Download wiring is documented in `COLAB_TRAINING.md` and the Colab notebook.

## Datasets

- LILA BC public object storage hosts Snapshot Kgalagadi and Serengeti images.
- `tools/data/download_images.py` is the resumable downloader and mirror boundary.
- Official annotation ZIP provenance is frozen in `data/manifests/build_info.json`.
- Kgalagadi is primary; Serengeti is supplemental evaluation only.

## Google Colab and Drive

- Code is cloned to `/content/Wild-Diff-ICMH` for fast local execution.
- Image bytes are copied from Drive to `/content/data` at session start.
- Run checkpoints are written directly to Drive for crash-safe resume.
- `Wild_Diff_ICMH_Kgalagadi_Train.ipynb` is the user entry point.

## External Evaluation Models

- MegaDetector produces a JSON/JSONL bbox sidecar consumed by H2 and foreground SSIM.
- RAM++ tag extraction is cached by `tools/precompute_ram_tags.py` before training.
- SpeciesNet and other task metrics remain separate future/evaluation environments.

## Security and Credentials

- Public model/data downloads need no embedded API keys.
- No credentials belong in YAML, JSONL, notebooks or manifests.
- Drive access is granted interactively by Colab mount.

*Integration analysis: 2026-09-24*
