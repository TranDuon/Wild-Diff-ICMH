"""Deterministic image geometry shared by decoding and evaluation."""
from __future__ import annotations

from typing import Iterable, Tuple

from PIL import Image


Box = Tuple[float, float, float, float]
# EVAL-11: camera-trap images are coded at this long side and evaluated at their
# original resolution.  Full 2592x2000 frames need ~26 GiB for the VAE attention.
DEFAULT_PROCESSING_LONG_SIDE = 1024
# Downscale before coding and upscale after decoding with the same filter so the
# protocol is reproducible; PIL's BICUBIC antialiases when it shrinks.
PROCESSING_RESAMPLING = "PIL.BICUBIC"


def resolve_geometry(
    crop_size: int | None,
    processing_long_side: int | None,
    *,
    manifest_supplied: bool,
) -> Tuple[int | None, int | None]:
    """Return ``(crop_size, processing_long_side)`` for one decode/evaluate run.

    A center crop is a smoke-test protocol and must be requested explicitly.
    Manifest-backed camera-trap runs otherwise use the original-resolution
    protocol at ``DEFAULT_PROCESSING_LONG_SIDE``; other inputs keep the legacy
    full-resolution behavior.
    """
    if crop_size is not None and processing_long_side is not None:
        raise ValueError("use either crop_size or processing_long_side, not both")
    if crop_size is not None:
        if crop_size <= 0:
            raise ValueError("crop_size must be positive")
        return crop_size, None
    if processing_long_side is not None:
        if processing_long_side <= 0:
            raise ValueError("processing_long_side must be positive")
        return None, processing_long_side
    return (None, DEFAULT_PROCESSING_LONG_SIDE) if manifest_supplied else (None, None)


def processing_size(width: int, height: int, long_side: int | None) -> Tuple[int, int]:
    """Size at which an image is coded: shrink so the long side fits, never enlarge."""
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if long_side is None or max(width, height) <= long_side:
        return width, height
    scale = long_side / max(width, height)
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def resize_for_processing(image: Image.Image, long_side: int | None) -> Image.Image:
    size = processing_size(image.width, image.height, long_side)
    if size == image.size:
        return image
    return image.resize(size, Image.Resampling.BICUBIC)


def restore_original_size(image: Image.Image, size: Tuple[int, int]) -> Image.Image:
    """Upscale a decoded image back to the original frame for evaluation."""
    if image.size == tuple(size):
        return image
    return image.resize(tuple(size), Image.Resampling.BICUBIC)


def scale_boxes(boxes: Iterable[Box], scale_x: float, scale_y: float) -> list[Box]:
    return [(x * scale_x, y * scale_y, w * scale_x, h * scale_y) for x, y, w, h in boxes]


def center_crop_geometry(
    width: int,
    height: int,
    crop_size: int,
) -> Tuple[int, int, int, int]:
    """Return resized width/height and the top-left corner of a square crop.

    Large images are cropped at native resolution, matching
    ``CameraTrapDataset(crop_type="center")``.  Only images smaller than the
    requested crop are upscaled first so that the crop always has the exact
    requested dimensions.
    """
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if crop_size <= 0:
        raise ValueError("crop_size must be positive")

    scale = max(crop_size / width, crop_size / height, 1.0)
    if scale > 1.0:
        resized_width = int(round(width * scale))
        resized_height = int(round(height * scale))
    else:
        resized_width, resized_height = width, height
    left = max(0, (resized_width - crop_size) // 2)
    top = max(0, (resized_height - crop_size) // 2)
    return resized_width, resized_height, left, top


def center_crop_image(image: Image.Image, crop_size: int) -> Image.Image:
    """Apply the deterministic center-crop protocol used by validation."""
    resized_width, resized_height, left, top = center_crop_geometry(
        image.width, image.height, crop_size
    )
    if (resized_width, resized_height) != image.size:
        image = image.resize(
            (resized_width, resized_height), Image.Resampling.BICUBIC
        )
    return image.crop((left, top, left + crop_size, top + crop_size))


def center_crop_boxes(
    boxes: Iterable[Box],
    original_width: int,
    original_height: int,
    crop_size: int,
) -> list[Box]:
    """Map pixel-space boxes into the same deterministic center crop."""
    resized_width, resized_height, left, top = center_crop_geometry(
        original_width, original_height, crop_size
    )
    scale_x = resized_width / original_width
    scale_y = resized_height / original_height
    transformed = []
    for x, y, width, height in boxes:
        x0 = max(0.0, x * scale_x - left)
        y0 = max(0.0, y * scale_y - top)
        x1 = min(float(crop_size), (x + width) * scale_x - left)
        y1 = min(float(crop_size), (y + height) * scale_y - top)
        if x1 > x0 and y1 > y0:
            transformed.append((x0, y0, x1 - x0, y1 - y0))
    return transformed

