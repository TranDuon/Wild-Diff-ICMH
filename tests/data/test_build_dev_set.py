from __future__ import annotations

import json
from pathlib import Path

from tools.data.build_dev_set import select_dev_rows

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data" / "manifests" / "kgalagadi_site_split.jsonl"
DEV_LIST = ROOT / "data" / "manifests" / "kgalagadi_dev.txt"


def _rows():
    return [json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_dev_set_is_validation_only_one_frame_per_sequence():
    selected = select_dev_rows(_rows(), size=300, seed=20261005)
    assert len(selected) == 300
    assert {row["split"] for row in selected} == {"val"}
    assert len({row["sequence_id"] for row in selected}) == 300


def test_dev_set_keeps_every_scarce_sequence():
    rows = _rows()
    selected = select_dev_rows(rows, size=300, seed=20261005)
    night = {row["sequence_id"] for row in rows if row["split"] == "val" and row["illumination"] == "night"}
    animal = {row["sequence_id"] for row in rows if row["split"] == "val" and not row["is_empty"]}
    chosen = {row["sequence_id"] for row in selected}
    assert night <= chosen
    assert animal <= chosen


def test_dev_set_is_deterministic_and_matches_committed_list():
    first = [row["image_id"] for row in select_dev_rows(_rows(), size=300, seed=20261005)]
    second = [row["image_id"] for row in select_dev_rows(_rows(), size=300, seed=20261005)]
    assert first == second
    committed = [
        line for line in DEV_LIST.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    assert committed == first
