---
last_mapped_commit: working-tree
---
# Testing Strategy

**Analysis Date:** 2026-09-24

## Framework

- Pytest is used for local, CPU-cheap checks.
- Tests live in `tests/data/` because current coverage centers on data contracts.
- Torch-dependent tests use `pytest.importorskip` when local Torch is absent.

## Covered Behavior

- `test_build_manifests.py` checks deterministic Kgalagadi splits and exclusions.
- `test_split_check.py` checks allowed same-site splits and forbidden sequence leakage.
- `test_download_images.py` checks resume, checksum and download safety behavior.
- `test_pipeline_e2e.py` exercises the manifest-to-list data handoff.
- `test_camera_trap_dataset.py` checks image/mask alignment when Torch is installed.
- `test_train_sites.py` checks site discovery and launcher command construction.

## Mandatory Local Gates

- `python -m compileall -q train.py inference.py inference_partition.py model dataset tools tests`
- `python -m pytest tests/data -q`
- `python tools/data/split_check.py data/manifests/kgalagadi_site_split.jsonl --build-info data/manifests/build_info.json`
- `git diff --check`

## GPU-Only Gates

- Instantiate model with real SD, RAM++ and author Diff-ICMH checkpoints.
- Run 20 optimizer steps, save `last.ckpt`, restart, and confirm full-state resume.
- Confirm frozen SD/VAE/RAM++ parameters remain non-trainable.
- Round-trip at least one H3 bitstream and decode metadata only from the stream.
- Verify measured BPP uses original dimensions rather than padded dimensions.

## Current Limitation

- Local workspace has no Torch installation, so model-level runtime tests are pending Colab.

*Testing analysis: 2026-09-24*
