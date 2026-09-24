---
last_mapped_commit: working-tree
---
# Coding Conventions

**Analysis Date:** 2026-09-24

## Python

- New modules use `from __future__ import annotations` where practical.
- Paths use `pathlib.Path` except when preserving legacy inference APIs.
- CLI entry points expose `main(argv=None)` when unit testing is useful.
- JSONL writers use UTF-8, sorted keys and explicit newline behavior.

## Configuration

- YAML `target` plus `params` is instantiated through `utils/common.py`.
- Environment-specific paths use OmegaConf `${oc.env:NAME}` resolvers.
- Base training configs compose through the root-level `base` key in `train.py`.
- H1/H2 behavior is selected by config, not hard-coded branches in launch scripts.

## Data Integrity

- Never split camera-trap data per image; use sequence/burst identifiers.
- Keep pseudo-labels in sidecars instead of mutating the frozen manifest.
- Do not use source species labels as text conditioning because that is oracle leakage.
- Count bitstream headers, tag payload and metadata in reported BPP.

## Resource Safety

- Load multi-GB checkpoints sequentially and delete host copies immediately.
- Do not call Lightning `unfreeze()` merely to restore training mode.
- Keep frozen backbones out of recurring project checkpoints.
- Precompute frozen RAM++ tags instead of carrying RAM++ during training.

## Error Handling

- Fail early for missing manifest rows, files, checkpoints and empty filters.
- Training/evaluation calls `split_check` before expensive work.
- Colab smoke tests intentionally cap sites, steps and validation batches.

*Convention analysis: 2026-09-24*
