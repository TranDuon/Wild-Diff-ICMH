from __future__ import annotations

import pytest

from tools.phase1_closeout import build_closeout_report


def _manifest_rows():
    rows = []
    for index, split in enumerate(("train", "val", "test")):
        rows.append({
            "image_id": f"image-{index}", "source": "snapshot_kgalagadi",
            "source_version": "test", "license": "test",
            "relative_path": f"images/{index}.jpg", "source_file_name": f"{index}.jpg",
            "site_id": "KGA:A01", "sequence_id": f"sequence-{index}",
            "frame_num": 1, "datetime": "2026-01-01 12:00:00",
            "illumination": "day", "illumination_source": "datetime_hour_proxy",
            "species": [], "is_empty": True, "boxes": None, "width": 32,
            "height": 32, "split": split, "subset": "empty", "sample_rank": index,
            "sha256": None, "null_reasons": {"boxes": "none", "sha256": "external"},
        })
    return rows


def _result_rows():
    common = {
        "exp_id": "smoke", "dataset": "kgalagadi", "lambda_rate": 2,
        "illumination": "all", "ddim_steps": 5, "value": 1.0,
        "n_images": 2, "cu_estimate": 0.1, "git_commit": "abc", "date": "now",
    }
    return [{**common, "metric": metric} for metric in ("bpp", "psnr", "lpips")]


def _checkpoint(step=21):
    return {
        "wild_diff_checkpoint_contract_version": 2,
        "wild_diff_compact": True,
        "global_step": step,
        "optimizer_states": [{"state": {}}],
    }


def _build(**overrides):
    values = dict(
        manifest_rows=_manifest_rows(),
        metadata_report={"sample_size": 100, "datetime_usable_ratio": 1.0, "location_usable_ratio": 1.0},
        result_rows=_result_rows(), exp_id="smoke", checkpoint=_checkpoint(),
        checkpoint_path="last.ckpt", resume_before=20, resume_after=21,
        smoke_steps=20, smoke_elapsed_seconds=300.0, consumed_cu_estimate=0.2,
        cu_per_hour=1.54, phase_budget_cu=8.0, calibration_target_steps=2000,
        gpu_name="NVIDIA L4", git_commit="abc",
    )
    values.update(overrides)
    return build_closeout_report(**values)


def test_closeout_rejects_over_budget_2k_run():
    report = _build()
    assert report["calibration"]["projected_cu_estimate"] > 8.0
    assert report["calibration"]["recommendation"] == "do_not_run_2k"
    assert report["classification"] == "smoke/non-report"


def test_closeout_allows_bounded_run_when_forecast_fits():
    report = _build(smoke_elapsed_seconds=30.0)
    assert report["calibration"]["within_remaining_budget"] is True
    assert report["calibration"]["recommendation"] == "run_2k_allowed_not_started"


def test_closeout_requires_resume_to_advance():
    with pytest.raises(RuntimeError, match="resume did not advance"):
        _build(resume_after=20)


def test_closeout_requires_compact_full_state_checkpoint():
    checkpoint = _checkpoint()
    checkpoint["wild_diff_compact"] = False
    with pytest.raises(RuntimeError, match="not marked compact"):
        _build(checkpoint=checkpoint)


def test_closeout_requires_canonical_smoke_metrics():
    with pytest.raises(RuntimeError, match="missing metrics"):
        _build(result_rows=_result_rows()[:2])

