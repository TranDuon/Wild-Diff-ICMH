"""Small, dependency-free checkpoint compatibility helpers."""

from __future__ import annotations

from typing import Any, Mapping


def state_dict_shape_mismatches(
    model: Any,
    checkpoint: Mapping[str, Any],
) -> list[tuple[str, tuple[int, ...], tuple[int, ...]]]:
    """Return checkpoint/model tensor shape conflicts for shared keys.

    ``strict=False`` permits missing and unexpected keys, but PyTorch still
    raises for a shared key whose tensor shape differs. Computing the list
    before ``load_state_dict`` lets callers report the actual architecture
    contract that was violated instead of emitting hundreds of clipped lines.

    This module deliberately avoids importing PyTorch so the configuration
    contract can be regression-tested in lightweight environments.
    """
    checkpoint_state = checkpoint.get("state_dict", checkpoint)
    model_state = model.state_dict()
    mismatches = []
    for key, checkpoint_value in checkpoint_state.items():
        model_value = model_state.get(key)
        if model_value is None:
            continue
        checkpoint_shape = getattr(checkpoint_value, "shape", None)
        model_shape = getattr(model_value, "shape", None)
        if checkpoint_shape is not None and tuple(checkpoint_shape) != tuple(model_shape):
            mismatches.append((key, tuple(checkpoint_shape), tuple(model_shape)))
    return mismatches
