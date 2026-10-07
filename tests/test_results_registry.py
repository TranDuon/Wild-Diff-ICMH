import json

import pytest

from utils.results_registry import metric_rows, upsert_jsonl


def _summary(psnr=30.0):
    return {
        "all": {"n": 2, "bpp": 0.4, "compression_ratio_rgb24": 60.0, "psnr": psnr},
        "illumination=day": {"n": 1, "bpp": 0.3, "compression_ratio_rgb24": 80.0, "psnr": 31.0},
        "site_id=KGA:A01": {"n": 2, "bpp": 0.4, "compression_ratio_rgb24": 60.0, "psnr": psnr},
    }


def _rows(psnr=30.0):
    return metric_rows(
        _summary(psnr),
        exp_id="phase1-smoke",
        dataset="snapshot_kgalagadi",
        lambda_rate=2,
        ddim_steps=5,
        cu_estimate=0.2,
        git_commit="abc123",
        method="H1",
        split="test",
        site_id="KGA:A01",
    )


def test_metric_rows_follow_schema_and_skip_site_duplicate():
    rows = _rows()
    assert {row["illumination"] for row in rows} == {"all", "day"}
    assert all(row["n_images"] in {1, 2} for row in rows)
    assert all(row["git_commit"] == "abc123" for row in rows)


def test_upsert_replaces_same_experiment_metric(tmp_path):
    path = tmp_path / "results.jsonl"
    first = _rows(30.0)
    upsert_jsonl(path, first)
    second = _rows(33.0)
    upsert_jsonl(path, second)

    stored = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(stored) == len(second)
    all_psnr = next(row for row in stored if row["illumination"] == "all" and row["metric"] == "psnr")
    assert all_psnr["value"] == pytest.approx(33.0)


def test_stratified_groups_carry_subset_ci_and_protocol():
    summary = {
        "illumination=all|content=all": {
            "n": 3, "bpp": 0.1, "ms_ssim": 0.9,
            "ci": {"method": "site_cluster", "intervals": {"ms_ssim": [0.85, 0.95]}},
        },
        "illumination=night|content=empty": {"n": 1, "bpp": 0.2},
        "site_id=KGA:A01": {"n": 3, "bpp": 0.1},
    }
    rows = metric_rows(
        summary, exp_id="b0", dataset="snapshot_kgalagadi", lambda_rate=2, ddim_steps=50,
        cu_estimate=None, git_commit="abc", method="B0", split="val", site_id=None,
        protocol={"processing_long_side": 1024},
    )
    by_metric = {(row["illumination"], row["subset"], row["metric"]): row for row in rows}
    assert by_metric[("all", "all", "ms_ssim")]["ci95"] == [0.85, 0.95]
    assert by_metric[("night", "empty", "bpp")]["protocol"] == {"processing_long_side": 1024}
    assert all(row["site_id"] is None for row in rows)
    assert len(rows) == 3


def test_legacy_rows_without_subset_are_replaced_not_duplicated(tmp_path):
    path = tmp_path / "results.jsonl"
    legacy = [dict(row) for row in _rows()]
    for row in legacy:
        row.pop("subset", None)
    path.write_text("".join(json.dumps(row) + "\n" for row in legacy), encoding="utf-8")
    upsert_jsonl(path, _rows(31.0))
    stored = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(stored) == len(legacy)
