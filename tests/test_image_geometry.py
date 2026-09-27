from PIL import Image

from utils.image_geometry import (
    center_crop_boxes,
    center_crop_geometry,
    center_crop_image,
    resolve_crop_size,
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


def test_manifest_commands_default_to_project_crop_for_stale_colab_cells():
    assert resolve_crop_size(None, manifest_supplied=True) == 256
    assert resolve_crop_size(None, manifest_supplied=False) is None
    assert resolve_crop_size(512, manifest_supplied=True) == 512

