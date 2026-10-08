"""Phase 2 results summary built from registry rows."""
from __future__ import annotations

import csv
import json

from tools.summarize_phase2 import load_points, main, markdown_table


def _row(exp_id, metric, value, *, illumination="all", subset="all", date="2026-10-06T10:00:00", **extra):
    return {
        "exp_id": exp_id, "dataset": "kgalagadi", "lambda_rate": "x", "illumination": illumination,
        "subset": subset, "ddim_steps": 0, "metric": metric, "value": value, "n_images": 202,
        "cu_estimate": None, "git_commit": "abc", "date": date, **extra,
    }


def _write(tmp_path, rows):
    path = tmp_path / "results.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_keeps_only_dev_whole_content_rows_and_newest_value(tmp_path):
    registry = _write(tmp_path, [
        _row("p2dev_B0_ls512_ddim50__lambda_2", "bpp", 0.01),
        _row("p2dev_B0_ls512_ddim50__lambda_2", "map", 0.5, date="2026-10-05T00:00:00"),
        _row("p2dev_B0_ls512_ddim50__lambda_2", "map", 0.7, ci95=[0.6, 0.8]),
        _row("p2dev_B0_ls512_ddim50__lambda_2", "map", 0.1, subset="empty"),
        _row("p2dev_B0_ls512_ddim50__lambda_2", "map", 0.66, illumination="night"),
        _row("phase1_smoke", "bpp", 9.9),
        _row("p2dev_jpeg_ls512__q5", "bpp", 0.02, protocol={"rate": "coded"}),
    ])
    points = load_points(registry)
    b0 = points[("B0_ls512_ddim50", "lambda_2", "all")]
    assert b0["map"] == 0.7 and b0["map_ci"] == [0.6, 0.8] and b0["bpp"] == 0.01
    assert points[("B0_ls512_ddim50", "lambda_2", "night")]["map"] == 0.66
    assert points[("jpeg_ls512", "q5", "all")]["rate"] == "coded"
    assert all(not key[0].startswith("phase1") for key in points)


def test_table_orders_points_numerically(tmp_path):
    registry = _write(tmp_path, [
        _row("p2dev_jpeg_ls512__q40", "bpp", 0.3),
        _row("p2dev_jpeg_ls512__q5", "bpp", 0.02),
        _row("p2dev_jpeg_ls512__q15", "bpp", 0.1),
    ])
    lines = markdown_table(load_points(registry)).splitlines()[2:]
    assert [line.split("|")[2].strip() for line in lines] == ["q5", "q15", "q40"]


def test_main_writes_csv_and_table(tmp_path):
    registry = _write(tmp_path, [
        _row("p2dev_B0_ls512_ddim50__lambda_8", "bpp", 0.005),
        _row("p2dev_B0_ls512_ddim50__lambda_8", "missed_animal_rate", 0.2, ci95=[0.1, 0.3]),
    ])
    out = tmp_path / "out"
    assert main(["--registry", str(registry), "--out", str(out), "--no-plot"]) == 0
    rows = list(csv.DictReader((out / "phase2_points.csv").open(encoding="utf-8")))
    assert rows[0]["missed_animal_rate_ci_high"] == "0.3"
    assert "B0_ls512_ddim50" in (out / "phase2_table.md").read_text(encoding="utf-8")
