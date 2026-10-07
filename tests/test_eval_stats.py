import pytest

from utils.eval_stats import (
    aggregate,
    bootstrap_ci,
    group_key,
    parse_group_key,
    stratified_groups,
)


def _row(site, empty, illumination, psnr, bytes_=1000):
    return {
        "site_id": site, "is_empty": empty, "illumination": illumination,
        "psnr": psnr, "width": 100, "height": 100, "bitstream_bytes": bytes_,
    }


ROWS = [
    _row("A", True, "day", 30.0), _row("A", False, "day", 32.0),
    _row("B", True, "night", 28.0, 2000), _row("B", False, "day", 34.0),
    _row("C", False, "night", 26.0, 500),
]


def test_aggregate_uses_total_bits_over_total_pixels():
    summary = aggregate(ROWS, ["bpp", "psnr"])
    assert summary["n"] == 5
    assert summary["bpp"] == pytest.approx((1000 * 3 + 2000 + 500) * 8 / 50000)
    assert summary["compression_ratio_rgb24"] == pytest.approx(24 / summary["bpp"])
    assert summary["psnr"] == pytest.approx(30.0)


def test_bootstrap_resamples_sites_and_brackets_the_estimate():
    result = bootstrap_ci(ROWS, ["psnr"], n_resamples=300, seed=1)
    assert result["method"] == "site_cluster"
    low, high = result["intervals"]["psnr"]
    assert low <= 30.0 <= high
    assert bootstrap_ci(ROWS, ["psnr"], n_resamples=300, seed=1) == result


def test_bootstrap_falls_back_to_images_for_one_site():
    result = bootstrap_ci([r for r in ROWS if r["site_id"] == "A"], ["psnr"], n_resamples=50)
    assert result["method"] == "image"


def test_groups_cover_day_night_and_empty_animal():
    groups = stratified_groups(ROWS, lambda row: row["illumination"])
    assert len(groups[group_key()]) == 5
    assert len(groups[group_key("night", "animal")]) == 1
    assert len(groups[group_key("all", "empty")]) == 2
    assert parse_group_key(group_key("day", "empty")) == {"illumination": "day", "content": "empty"}
    assert parse_group_key("site_id=KGA:A01") is None
