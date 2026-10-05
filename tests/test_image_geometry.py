import pytest
from PIL import Image

from utils.image_geometry import (
    center_crop_boxes,
    center_crop_geometry,
    center_crop_image,
    processing_size,
    resize_for_processing,
    resolve_geometry,
    restore_original_size,
    scale_boxes,
)


def test_large_camera_trap_image_is_cropped_without_upscaling():
    assert center_crop_geometry(2592, 2000, 256) == (2592, 2000, 1168, 872)
    image = Image.new("RGB", (2592, 2000))
    assert center_crop_image(image, 256).size == (256, 256)


def test_small_image_is_upscaled_before_center_crop():
    assert center_crop_geometry(100, 200, 256) == (256, 512, 0, 128)
    image = Image.new("RGB", (100, 200))
    assert center_crop_image(image, 256).size == (256, 256)


def test_boxes_are_scaled_clipped_and_shifted_into_crop_coordinates():
    boxes = center_crop_boxes(
        [(1160.0, 870.0, 100.0, 100.0), (0.0, 0.0, 20.0, 20.0)],
        2592,
        2000,
        256,
    )
    assert boxes == [(0.0, 0.0, 92.0, 98.0)]


def test_manifest_commands_default_to_original_resolution_protocol():
    assert resolve_geometry(None, None, manifest_supplied=True) == (None, 1024)
    assert resolve_geometry(None, None, manifest_supplied=False) == (None, None)
    assert resolve_geometry(256, None, manifest_supplied=True) == (256, None)
    assert resolve_geometry(None, 512, manifest_supplied=True) == (None, 512)


def test_crop_and_processing_resolution_are_mutually_exclusive():
    with pytest.raises(ValueError, match="not both"):
        resolve_geometry(256, 1024, manifest_supplied=True)
    with pytest.raises(ValueError, match="positive"):
        resolve_geometry(None, 0, manifest_supplied=True)


def test_processing_size_shrinks_long_side_and_never_enlarges():
    assert processing_size(2592, 2000, 1024) == (1024, 790)
    assert processing_size(2000, 2592, 1024) == (790, 1024)
    assert processing_size(800, 600, 1024) == (800, 600)
    assert processing_size(2592, 2000, None) == (2592, 2000)


def test_processing_round_trip_restores_original_frame():
    image = Image.new("RGB", (2592, 2000), color=(10, 20, 30))
    small = resize_for_processing(image, 1024)
    assert small.size == (1024, 790)
    restored = restore_original_size(small, image.size)
    assert restored.size == (2592, 2000)
    assert restored.getpixel((1296, 1000)) == (10, 20, 30)


def test_boxes_follow_the_processing_resize():
    assert scale_boxes([(100.0, 200.0, 50.0, 40.0)], 0.5, 0.25) == [(50.0, 50.0, 25.0, 10.0)]

