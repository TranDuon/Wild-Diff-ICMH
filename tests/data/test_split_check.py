"""Deliberate leak-injection cases proving tools/data/split_check.py blocks."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import split_check  # noqa: E402


def _row(**overrides):
    row = {
        "image_id": "SER:img1",
        "source": "snapshot_serengeti",
        "source_version": "v1",
        "license": "CDLA-Permissive-1.0",
        "relative_path": "snapshot_serengeti/a.jpg",
        "source_file_name": "a.jpg",
        "site_id": "SER:D01",
        "sequence_id": "SER:SEQ1",
        "frame_num": 1,
        "datetime": "2010-10-08 12:00:00",
        "illumination": "day",
        "illumination_source": "datetime_hour_proxy",
        "species": None,
        "is_empty": None,
        "boxes": [],
        "width": 100,
        "height": 100,
        "split": "train",
        "subset": "bbox_subset",
        "sample_rank": 0,
        "sha256": None,
        "null_reasons": {"species": "x", "is_empty": "x", "sha256": "x"},
    }
    row.update(overrides)
    return row


def _write_manifest(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True))
            fh.write("\n")
    return path


def test_clean_manifest_passes(tmp_path):
    rows = [_row(image_id="SER:img1"), _row(image_id="SER:img2", site_id="SER:D02", sequence_id="SER:SEQ2", split="val")]
    manifest = _write_manifest(tmp_path / "m.jsonl", rows)
    assert split_check.find_violations(split_check.load_manifest_rows([manifest])) == []
    split_check.assert_no_leakage([manifest])  # does not raise


def test_shared_site_between_train_and_val_is_blocked(tmp_path):
    rows = [
        _row(image_id="SER:img1", site_id="SER:D01", sequence_id="SER:SEQ1", split="train"),
        _row(image_id="SER:img2", site_id="SER:D01", sequence_id="SER:SEQ2", split="val"),
    ]
    manifest = _write_manifest(tmp_path / "m.jsonl", rows)
    violations = split_check.find_violations(split_check.load_manifest_rows([manifest]))
    assert any("site" in v and "SER:D01" in v for v in violations)

    with pytest.raises(split_check.SplitLeakageError):
        split_check.assert_no_leakage([manifest])

    result = subprocess.run(
        [sys.executable, str(_REPO_ROOT / "tools" / "data" / "split_check.py"), str(manifest)],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "FAILED" in result.stdout


def test_kgalagadi_site_overlap_allowed_but_sequence_overlap_rejected(tmp_path):
    train = _row(
        image_id="KGA:img1",
        source="snapshot_kgalagadi",
        relative_path="snapshot_kgalagadi/train.jpg",
        site_id="KGA:A01",
        sequence_id="KGA:SEQ1",
        split="train",
    )
    val = _row(
        image_id="KGA:img2",
        source="snapshot_kgalagadi",
        relative_path="snapshot_kgalagadi/val.jpg",
        site_id="KGA:A01",
        sequence_id="KGA:SEQ2",
        split="val",
    )
    manifest = _write_manifest(tmp_path / "kga.jsonl", [train, val])
    rows = split_check.load_manifest_rows([manifest])
    assert split_check.find_violations(rows) == []

    val["sequence_id"] = train["sequence_id"]
    manifest = _write_manifest(tmp_path / "kga_leak.jsonl", [train, val])
    violations = split_check.find_violations(split_check.load_manifest_rows([manifest]))
    assert any("sequence" in violation for violation in violations)


def test_shared_sequence_across_splits_even_with_different_sites(tmp_path):
    rows = [
        _row(image_id="SER:img1", site_id="SER:D01", sequence_id="SER:SEQ1", split="train"),
        _row(image_id="SER:img2", site_id="SER:D02", sequence_id="SER:SEQ1", split="val"),
    ]
    manifest = _write_manifest(tmp_path / "m.jsonl", rows)
    violations = split_check.find_violations(split_check.load_manifest_rows([manifest]))
    assert any("sequence" in v and "SER:SEQ1" in v for v in violations)


def test_duplicate_image_id(tmp_path):
    rows = [
        _row(image_id="SER:img1", site_id="SER:D01", sequence_id="SER:SEQ1"),
        _row(image_id="SER:img1", site_id="SER:D02", sequence_id="SER:SEQ2"),
    ]
    manifest = _write_manifest(tmp_path / "m.jsonl", rows)
    violations = split_check.find_violations(split_check.load_manifest_rows([manifest]))
    assert any("duplicate image_id" in v for v in violations)


def test_invalid_split(tmp_path):
    rows = [_row(split="bogus")]
    manifest = _write_manifest(tmp_path / "m.jsonl", rows)
    violations = split_check.find_violations(split_check.load_manifest_rows([manifest]))
    assert any("invalid split" in v for v in violations)


def test_missing_field(tmp_path):
    row = _row()
    del row["width"]
    manifest = _write_manifest(tmp_path / "m.jsonl", [row])
    violations = split_check.find_violations(split_check.load_manifest_rows([manifest]))
    assert any("missing required field" in v for v in violations)


def test_none_value_with_no_reason(tmp_path):
    row = _row(width=None)
    row["null_reasons"] = {"species": "x", "is_empty": "x", "sha256": "x"}  # no reason for width
    manifest = _write_manifest(tmp_path / "m.jsonl", [row])
    violations = split_check.find_violations(split_check.load_manifest_rows([manifest]))
    assert any("width" in v and "null_reasons" in v for v in violations)


@pytest.mark.parametrize(
    "bad_path",
    ["../x.jpg", "snapshot_serengeti/../x.jpg", "/abs.jpg", "C:/x.jpg"],
)
def test_unsafe_relative_path(tmp_path, bad_path):
    rows = [_row(relative_path=bad_path)]
    manifest = _write_manifest(tmp_path / "m.jsonl", rows)
    violations = split_check.find_violations(split_check.load_manifest_rows([manifest]))
    assert any("unsafe relative_path" in v for v in violations)


def test_list_file_declares_val_path_as_train(tmp_path):
    rows = [
        _row(image_id="SER:img1", site_id="SER:D01", sequence_id="SER:SEQ1", split="train", relative_path="snapshot_serengeti/train1.jpg"),
        _row(image_id="SER:img2", site_id="SER:D02", sequence_id="SER:SEQ2", split="val", relative_path="snapshot_serengeti/val1.jpg"),
    ]
    manifest = _write_manifest(tmp_path / "m.jsonl", rows)

    list_dir = tmp_path / "lists"
    list_dir.mkdir()
    train_list = list_dir / "train.list"
    # wrongly declare the val row's path as train
    train_list.write_text(str(tmp_path / "images" / "snapshot_serengeti" / "val1.jpg") + "\n", encoding="utf-8")

    with pytest.raises(split_check.SplitLeakageError):
        split_check.assert_no_leakage([manifest], list_files={"train": str(train_list)})


def test_build_info_sha_mismatch_after_tampering(tmp_path):
    rows = [_row(image_id="SER:img1")]
    manifest = _write_manifest(tmp_path / "m.jsonl", rows)

    import hashlib

    build_info = {"output_sha256": {"m.jsonl": hashlib.sha256(manifest.read_bytes()).hexdigest()}}
    build_info_path = tmp_path / "build_info.json"
    build_info_path.write_text(json.dumps(build_info), encoding="utf-8")

    # sanity: passes before tampering
    split_check.assert_no_leakage([manifest], build_info=str(build_info_path))

    # tamper with the manifest after build_info was written
    with manifest.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(_row(image_id="SER:img_extra", site_id="SER:D09", sequence_id="SER:SEQ9")))
        fh.write("\n")

    with pytest.raises(split_check.SplitLeakageError):
        split_check.assert_no_leakage([manifest], build_info=str(build_info_path))
