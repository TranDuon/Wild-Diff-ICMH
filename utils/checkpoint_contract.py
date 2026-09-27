"""Small, dependency-free checkpoint compatibility helpers."""

from __future__ import annotations

import re
from typing import Any, Mapping


_LEGACY_ENTROPY_PARAMETER = re.compile(
    r"^(?P<prefix>.+\.entropy_bottleneck)\._"
    r"(?P<kind>matrix|bias|factor)(?P<index>\d+)$"
)
_CURRENT_ENTROPY_PARAMETER = re.compile(
    r"^(?P<prefix>.+\.entropy_bottleneck)\."
    r"(?P<kind>matrices|biases|factors)\.(?P<index>\d+)$"
)
_ENTROPY_PARAMETER_LIST = {
    "matrix": "matrices",
    "bias": "biases",
    "factor": "factors",
}
PROJECT_CHECKPOINT_CONTRACT_VERSION = 2


def validate_project_resume_checkpoint(checkpoint: Mapping[str, Any]) -> None:
    """Reject project checkpoints created before the entropy migration fix."""
    version = checkpoint.get("wild_diff_checkpoint_contract_version")
    if version != PROJECT_CHECKPOINT_CONTRACT_VERSION:
        raise RuntimeError(
            "Unsafe project resume checkpoint: expected contract version "
            f"{PROJECT_CHECKPOINT_CONTRACT_VERSION}, found {version!r}. "
            "This checkpoint predates the author entropy-bottleneck migration fix; "
            "start in a fresh run directory instead of resuming it."
        )
    if int(checkpoint.get("global_step", -1)) < 0:
        raise RuntimeError("project resume checkpoint has no valid global_step")
    if not checkpoint.get("optimizer_states"):
        raise RuntimeError("project resume checkpoint has no optimizer_states")


def migrate_legacy_entropy_bottleneck_checkpoint(
    checkpoint: Mapping[str, Any],
) -> tuple[dict[str, Any], tuple[tuple[str, str], ...]]:
    """Translate old CompressAI entropy-parameter names without mutating input.

    CompressAI used to register the parameters as ``_matrix0``, ``_bias0``
    and ``_factor0``.  Current releases expose the same tensors through
    ``ParameterList`` objects named ``matrices.0``, ``biases.0`` and
    ``factors.0``.  Published Diff-ICMH checkpoints use the former layout.

    A checkpoint containing both spellings is rejected rather than choosing
    one silently: the two values could represent different learned entropy
    models, and accepting either would make the warm start ambiguous.
    """
    wrapped = "state_dict" in checkpoint
    state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state, Mapping):
        raise TypeError("checkpoint state_dict must be a mapping")

    migrated_state = dict(state)
    migrations: list[tuple[str, str]] = []
    for legacy_key in state:
        match = _LEGACY_ENTROPY_PARAMETER.fullmatch(str(legacy_key))
        if match is None:
            continue
        current_key = (
            f"{match.group('prefix')}."
            f"{_ENTROPY_PARAMETER_LIST[match.group('kind')]}."
            f"{match.group('index')}"
        )
        if current_key in state:
            raise ValueError(
                "Entropy-bottleneck checkpoint key collision: "
                f"both {legacy_key!r} and {current_key!r} are present"
            )
        migrated_state[current_key] = migrated_state.pop(legacy_key)
        migrations.append((str(legacy_key), current_key))

    if wrapped:
        migrated_checkpoint = dict(checkpoint)
        migrated_checkpoint["state_dict"] = migrated_state
    else:
        migrated_checkpoint = migrated_state
    return migrated_checkpoint, tuple(migrations)


def validate_entropy_bottleneck_checkpoint(
    model: Any,
    checkpoint: Mapping[str, Any],
) -> None:
    """Require a complete current-layout entropy parameter set.

    Validation is limited to entropy-bottleneck modules represented by the
    checkpoint, so partial checkpoints for unrelated model components remain
    valid.  Call this after migration and before shape validation/loading.
    """
    state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state, Mapping):
        raise TypeError("checkpoint state_dict must be a mapping")

    leftover_legacy = sorted(
        str(key) for key in state
        if _LEGACY_ENTROPY_PARAMETER.fullmatch(str(key)) is not None
    )
    if leftover_legacy:
        preview = ", ".join(leftover_legacy[:3])
        raise RuntimeError(
            "Legacy entropy-bottleneck keys remain; migrate the checkpoint "
            f"before validation/loading. First keys: {preview}"
        )

    checkpoint_keys = {str(key) for key in state}
    current_checkpoint_keys = {
        key for key in checkpoint_keys
        if _CURRENT_ENTROPY_PARAMETER.fullmatch(key) is not None
    }
    affected_prefixes = {
        _CURRENT_ENTROPY_PARAMETER.fullmatch(key).group("prefix")
        for key in current_checkpoint_keys
    }
    if not affected_prefixes:
        return

    model_keys = {str(key) for key in model.state_dict()}
    expected = {
        key for key in model_keys
        if (
            (match := _CURRENT_ENTROPY_PARAMETER.fullmatch(key)) is not None
            and match.group("prefix") in affected_prefixes
        )
    }
    missing = sorted(expected - checkpoint_keys)
    unexpected = sorted(current_checkpoint_keys - model_keys)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing current keys: {', '.join(missing[:5])}")
        if unexpected:
            details.append(f"keys absent from model: {', '.join(unexpected[:5])}")
        raise RuntimeError(
            "Incomplete entropy-bottleneck checkpoint migration ("
            + "; ".join(details)
            + ")"
        )


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
