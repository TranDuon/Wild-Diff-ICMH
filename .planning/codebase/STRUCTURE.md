---
last_mapped_commit: working-tree
---
# Repository Structure

**Analysis Date:** 2026-09-24

## Entry Points

- `train.py` — Lightning 2 training, preflight, warm start and resume.
- `inference.py` — simple full-checkpoint inference.
- `inference_partition.py` — SD plus codec/control checkpoint inference and H3 bitstream.
- `Wild_Diff_ICMH_Kgalagadi_Train.ipynb` — guarded Colab smoke-test entry.

## Model Code

- `model/diffeic.py` — DiffEIC integration, losses, optimizer and compact checkpoints.
- `model/lfgcm.py` — entropy codec, semantic preprocessor and RAM++ wrapper.
- `model/callbacks.py` — image logging and Lightning checkpoint callback export.
- `ldm/` — Stable Diffusion-derived modules and samplers.

## Data Code

- `dataset/camera_trap_dataset.py` — manifest dataset and synchronized image/ROI transforms.
- `dataset/data_module.py` — Lightning data module.
- `tools/data/` — manifest creation, validation and image download.
- `data/manifests/` — frozen JSONL split plus provenance.
- `data/vocab/` — H3 restricted RAM++ vocabulary.

## Experiment Configuration

- `configs/model/diffeic.yaml` — model architecture and loss defaults.
- `configs/dataset/kgalagadi_*.yaml` — split-specific data loaders.
- `configs/train_kgalagadi_colab.yaml` — H1 Colab defaults.
- `configs/train_kgalagadi_h1_control.yaml` — compute-matched H2 control.
- `configs/train_kgalagadi_h2.yaml` — ROI-weighted continuation.

## Utilities and Tests

- `tools/train_kgalagadi_sites.py` — sequential per-site launcher.
- `tools/precompute_ram_tags.py` — resumable RAM++ tag cache.
- `tools/evaluate_kgalagadi.py` — paper-aligned image/bitstream metrics.
- `tests/data/` — deterministic unit and pipeline tests.

*Structure analysis: 2026-09-24*
