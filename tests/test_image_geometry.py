from PIL import Image

from utils.image_geometry import (
    center_crop_boxes,
    center_crop_geometry,
    center_crop_image,
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

