from datetime import datetime

import pytest
from PIL import Image

from utils.illumination import (
    EXIF_FLASH,
    EXIF_IFD,
    apply_sidecar,
    classify,
    classify_file,
    read_flash,
    sun_elevation_deg,
)

LOCATION = {"latitude": -25.8, "longitude": 20.6, "utc_offset_hours": 2}


def _frame(colour=(120, 90, 60), sky=(200, 200, 220)):
    image = Image.new("RGB", (64, 48), colour)
    image.paste(sky, (0, 0, 64, 12))
    return image


def _save_with_flash(tmp_path, image, flash):
    exif = Image.Exif()
    if flash is not None:
        exif.get_ifd(EXIF_IFD)[EXIF_FLASH] = flash
    path = tmp_path / f"frame_{flash}.jpg"
    image.resize((640, 480)).save(path, exif=exif, quality=95)
    return path


def test_exif_flash_fired_is_night_even_for_a_bright_colour_frame(tmp_path):
    path = _save_with_flash(tmp_path, _frame(), 0x19)  # fired, auto mode
    result = classify_file(path)
    assert (result["illumination"], result["illumination_source"], result["confidence"]) == ("night", "exif_flash", "high")


def test_exif_flash_not_fired_is_day_even_at_a_night_hour(tmp_path):
    path = _save_with_flash(tmp_path, _frame(), 0x10)
    result = classify_file(path, capture_time=datetime(2018, 12, 16, 5, 57), location=LOCATION)
    assert (result["illumination"], result["illumination_source"]) == ("day", "exif_flash")


def test_no_flash_function_is_treated_as_missing(tmp_path):
    path = _save_with_flash(tmp_path, _frame(), 0x20)
    with Image.open(path) as image:
        assert read_flash(image) is None


def test_infrared_grayscale_is_night_without_exif():
    gray = _frame(colour=(90, 90, 90), sky=(150, 150, 150))
    result = classify(gray, None)
    assert (result["illumination"], result["illumination_source"]) == ("night", "ir_grayscale")
    assert result["is_grayscale"] is True


def test_grayscale_overrides_a_flash_tag_that_says_not_fired():
    gray = _frame(colour=(90, 90, 90), sky=(150, 150, 150))
    assert classify(gray, 0x10)["illumination"] == "night"


@pytest.mark.parametrize(
    "stamp, expected",
    [("2018-12-16 12:30", "day"), ("2019-01-06 01:55", "night"),
     ("2018-12-16 05:57", "day"), ("2018-12-20 19:34", "day"), ("2018-12-20 20:30", "night")],
)
def test_solar_rule_when_the_dataset_supplies_a_location(stamp, expected):
    result = classify(_frame(), None, capture_time=datetime.strptime(stamp, "%Y-%m-%d %H:%M"), location=LOCATION)
    assert (result["illumination"], result["illumination_source"], result["confidence"]) == (expected, "solar", "medium")


def test_pixel_brightness_is_the_low_confidence_last_resort():
    dark_sky = classify(_frame(sky=(20, 25, 30)), None)
    bright_sky = classify(_frame(), None)
    assert (dark_sky["illumination"], dark_sky["confidence"]) == ("night", "low")
    assert bright_sky["illumination"] == "day"


def test_summer_noon_sun_is_near_zenith_at_the_tropic():
    assert sun_elevation_deg(datetime(2018, 12, 21, 10, 40), -25.8, 20.6) > 85


def test_apply_sidecar_keeps_the_hour_proxy():
    rows = [{"image_id": "a", "illumination": "night"}, {"image_id": "b", "illumination": "day"}]
    changed = apply_sidecar(rows, {"a": {"illumination": "day", "illumination_source": "exif_flash"}})
    assert changed == 1
    assert rows[0] == {
        "image_id": "a", "illumination": "day", "illumination_hour_proxy": "night",
        "illumination_source": "exif_flash", "is_grayscale": None,
    }
    assert rows[1]["illumination"] == "day"
