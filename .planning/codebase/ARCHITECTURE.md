---
last_mapped_commit: working-tree
---
# Architecture

**Analysis Date:** 2026-09-24

## High-Level Flow

1. `tools/data/build_manifests.py` freezes the official annotation split.
2. `tools/data/split_check.py` blocks sequence leakage before train/eval.
3. `dataset/camera_trap_dataset.py` loads images, cached tags, ROI boxes and metadata.
4. `dataset/data_module.py` exposes train/validation loaders to Lightning.
5. `train.py` constructs DiffEIC, warm-starts or resumes, then calls `Trainer.fit`.
6. `inference_partition.py` writes real entropy bitstreams and reconstructed PNGs.
7. `tools/evaluate_kgalagadi.py` measures BPP and quality from those artifacts.

## Model Boundaries

- `model/diffeic.py::DiffEIC` coordinates codec, control network and frozen diffusion prior.
- `model/lfgcm.py::LFGCM` is the trainable learned condition codec.
- `model/diffeic.py::CDDM` is the trainable control module.
- SD 2.1 UNet/VAE/text encoder are frozen by configuration.
- `model/lfgcm.py::TagGCM` is frozen and used mainly in preprocessing/inference.

## Experiment Variants

- Baseline: unmodified author checkpoint on Kgalagadi test.
- H1: codec/control fine-tuning with cached original RAM++ conditioning.
- H1-control: same continuation budget as H2, but uniform loss.
- H2: ROI-weighted latent guide loss; optional semantic weighting is a later V2.
- H3: decode/encode-time restricted vocabulary plus transmitted domain metadata.

## Checkpoint Semantics

- `--init-checkpoint` is a weights-only warm start and resets optimizer/global step.
- `--resume` restores a project Lightning checkpoint with full training state.
- Compact project checkpoints omit frozen SD/VAE/RAM++ model state.
- Base SD weights are restored during construction before a compact delta is loaded.

## Data Contract

- Each site is present in train/val/test because training is per site.
- Sequence is the non-leakage unit; frames from one burst never cross splits.
- Metadata and transmitted tag overhead are counted in BPP.

*Architecture analysis: 2026-09-24*
