---
last_mapped_commit: working-tree
---
# Technology Stack

**Analysis Date:** 2026-09-24

## Runtime

- Python 3.12 is the Colab target; local legacy instructions in `README.md` are not authoritative for training.
- PyTorch is supplied by Colab and must not be replaced by a random CUDA wheel.
- GPU target is one L4 24 GB; T4 uses `16-mixed`, L4 uses `bf16-mixed`.

## Core ML

- PyTorch drives all tensor/model code under `model/` and `ldm/`.
- Lightning 2.x (`lightning.pytorch`) owns the training loop in `train.py`.
- CompressAI 1.2.8 supplies entropy models used by `model/lfgcm.py`.
- Stable Diffusion 2.1 supplies the frozen VAE, text encoder and UNet prior.
- RAM++ is vendored at `src/recognize-anything/` and frozen.
- PyTorch scaled-dot-product attention is used in `ldm/modules/attention.py`.

## Data and Configuration

- OmegaConf/YAML composes model, data and Trainer settings under `configs/`.
- JSONL is the stable interchange format for manifests, tags and results.
- Pillow and NumPy handle image decoding and synchronized crop/mask transforms.
- `requirements-colab.txt` installs around Colab's existing Torch installation.

## Evaluation

- `pyiqa` provides LPIPS and other image-quality metrics.
- Project-native PSNR/SSIM helpers live in `utils/metrics.py`.
- Pytest covers deterministic data/split/orchestration behavior under `tests/data/`.

## Important Version Policy

- Keep NumPy below 2.0 for older research dependencies.
- Install CompressAI with `--no-deps` on Colab.
- xFormers is optional; do not install the repository's obsolete pinned wheel.
- Keep RAM++ dependencies isolated from detector/species environments.

*Stack analysis: 2026-09-24*
