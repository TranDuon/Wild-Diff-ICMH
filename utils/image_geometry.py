"""Deterministic image geometry shared by decoding and evaluation."""
from __future__ import annotations

from typing import Iterable, Tuple

from PIL import Image


Box = Tuple[float, float, float, float]


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

