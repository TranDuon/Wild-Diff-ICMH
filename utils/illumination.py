"""Day/night by light source, from the image file alone.

``day`` means the frame is lit by daylight; ``night`` means the camera lit it
itself, with a white flash (colour frame) or infrared LEDs (grayscale frame).
That is what changes how a frame looks, and every camera-trap image carries
enough evidence to decide it without dataset-specific knowledge (no park
coordinates, time zone or camera brand).  Signals, strongest first:

1. EXIF ``Flash`` (0x9209) bit 0 set                     -> night (exif_flash, high)
2. channels (almost) equal on a 64x64 thumbnail, i.e. IR -> night (ir_grayscale, high)
3. EXIF ``Flash`` present, flash function, not fired    -> day   (exif_flash, high)
4. capture time + a location the dataset supplies:
   sun elevation <= -6 deg (end of civil twilight)        -> night (solar, medium)
5. otherwise: mean luminance of the top quarter < 67      -> night (pixel_brightness, low)

A capture-hour rule (e.g. 19:00-06:00) is deliberately absent: it mislabels
dawn and dusk.  Thresholds are fixed constants recorded in every output.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Mapping, Optional

from PIL import Image

EXIF_IFD = 0x8769
EXIF_FLASH = 0x9209
FLASH_FIRED_BIT = 0x01
FLASH_NO_FUNCTION_BIT = 0x20
GRAYSCALE_MAX_SPREAD = 2.0          # same rule as download_images.is_grayscale
TOP_LUMA_NIGHT_BELOW = 67.0         # calibrated against EXIF flash; weakest signal
SOLAR_NIGHT_AT_OR_BELOW_DEG = -6.0  # civil twilight
THUMBNAIL_SIZE = (64, 48)

RULES = {
    "grayscale_max_spread": GRAYSCALE_MAX_SPREAD,
    "top_luma_night_below": TOP_LUMA_NIGHT_BELOW,
    "solar_night_at_or_below_deg": SOLAR_NIGHT_AT_OR_BELOW_DEG,
    "order": ["exif_flash_fired", "ir_grayscale", "exif_flash_not_fired", "solar", "pixel_brightness"],
}


def read_flash(image: Image.Image) -> Optional[int]:
    """Raw EXIF Flash value, or None when absent or the camera has no flash function."""
    exif = image.getexif()
    value = exif.get_ifd(EXIF_IFD).get(EXIF_FLASH, exif.get(EXIF_FLASH))
    if value is None:
        return None
    value = int(value)
    return None if value & FLASH_NO_FUNCTION_BIT else value


def thumbnail(image: Image.Image) -> Image.Image:
    if image.format == "JPEG":
        image.draft("RGB", (THUMBNAIL_SIZE[0] * 4, THUMBNAIL_SIZE[1] * 4))
    return image.convert("RGB").resize(THUMBNAIL_SIZE)


def channel_spread(small: Image.Image) -> float:
    pixels = list(small.getdata())
    return sum(max(abs(r - g), abs(g - b), abs(r - b)) for r, g, b in pixels) / len(pixels)


def top_luma(small: Image.Image) -> float:
    gray = list(small.convert("L").getdata())
    top = gray[: THUMBNAIL_SIZE[0] * (THUMBNAIL_SIZE[1] // 4)]
    return sum(top) / len(top)


def sun_elevation_deg(utc_time: datetime, latitude: float, longitude: float) -> float:
    """Solar elevation from a low-precision almanac (error well under 1 degree)."""
    days = (utc_time - datetime(2000, 1, 1, 12)).total_seconds() / 86400.0
    anomaly = math.radians((357.529 + 0.98560028 * days) % 360)
    mean_longitude = (280.459 + 0.98564736 * days) % 360
    ecliptic = math.radians((mean_longitude + 1.915 * math.sin(anomaly) + 0.020 * math.sin(2 * anomaly)) % 360)
    obliquity = math.radians(23.439 - 0.00000036 * days)
    right_ascension = math.atan2(math.cos(obliquity) * math.sin(ecliptic), math.cos(ecliptic))
    declination = math.asin(math.sin(obliquity) * math.sin(ecliptic))
    sidereal = (18.697374558 + 24.06570982441908 * days) % 24
    hour_angle = math.radians(((sidereal * 15 + longitude) - math.degrees(right_ascension) + 540) % 360 - 180)
    lat = math.radians(latitude)
    return math.degrees(math.asin(
        math.sin(lat) * math.sin(declination) + math.cos(lat) * math.cos(declination) * math.cos(hour_angle)
    ))


def classify(
    small: Image.Image,
    flash: Optional[int],
    *,
    capture_time: Optional[datetime] = None,
    location: Optional[Mapping[str, float]] = None,
) -> dict:
    """Label one frame; ``location`` needs ``latitude``, ``longitude``, ``utc_offset_hours``."""
    spread = channel_spread(small)
    luma = top_luma(small)
    record = {
        "flash": flash,
        "channel_spread": round(spread, 3),
        "top_luma": round(luma, 1),
        "is_grayscale": spread <= GRAYSCALE_MAX_SPREAD,
    }

    def label(value, source, confidence):
        return {**record, "illumination": value, "illumination_source": source, "confidence": confidence}

    if flash is not None and flash & FLASH_FIRED_BIT:
        return label("night", "exif_flash", "high")
    if record["is_grayscale"]:
        return label("night", "ir_grayscale", "high")
    if flash is not None:
        return label("day", "exif_flash", "high")
    if capture_time is not None and location is not None:
        utc = capture_time - timedelta(hours=float(location["utc_offset_hours"]))
        elevation = sun_elevation_deg(utc, float(location["latitude"]), float(location["longitude"]))
        record["sun_elevation_deg"] = round(elevation, 2)
        return label("night" if elevation <= SOLAR_NIGHT_AT_OR_BELOW_DEG else "day", "solar", "medium")
    return label("night" if luma < TOP_LUMA_NIGHT_BELOW else "day", "pixel_brightness", "low")


def classify_file(path, *, capture_time=None, location=None) -> dict:
    with Image.open(path) as image:
        flash = read_flash(image)
        small = thumbnail(image)
    return classify(small, flash, capture_time=capture_time, location=location)


def load_sidecar(path) -> dict[str, dict]:
    """``image_id -> record`` from a label_illumination.py sidecar (missing file -> {})."""
    import json
    from pathlib import Path

    if not path or not Path(path).is_file():
        return {}
    records = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
            records[record["image_id"]] = record
        except (ValueError, KeyError):
            continue  # a line cut by a dead runtime is relabelled
    return records


def apply_sidecar(rows, sidecar: Mapping[str, Mapping]) -> int:
    """Replace each row's ``illumination`` with the light-source label, in place.

    The manifest's capture-hour value is kept as ``illumination_hour_proxy``.
    Returns how many rows were relabelled.
    """
    changed = 0
    for row in rows:
        record = sidecar.get(row.get("image_id"))
        if record is None:
            continue
        row.setdefault("illumination_hour_proxy", row.get("illumination"))
        row["illumination"] = record["illumination"]
        row["illumination_source"] = record.get("illumination_source")
        row["is_grayscale"] = record.get("is_grayscale")
        changed += 1
    return changed
