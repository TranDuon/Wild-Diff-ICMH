"""Contract tests for tools/data/build_manifests.py (Task 2 behaviors)."""
from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import build_manifests  # noqa: E402
import split_check  # noqa: E402


# ---------------------------------------------------------------------------
# illumination_from_datetime
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        ("2010-10-08 05:59:00", "night"),
        ("2010-10-08 06:00:00", "day"),
        ("2010-10-08 18:59:59", "day"),
        ("2010-10-08 19:00:00", "night"),
    ],
)
def test_illumination_from_datetime_boundaries(value, expected):
    label, reason = build_manifests.illumination_from_datetime(value)
    assert label == expected
    assert reason is None


@pytest.mark.parametrize("value", [None, "", "garbage"])
def test_illumination_from_datetime_invalid(value):
    label, reason = build_manifests.illumination_from_datetime(value)
    assert label is None
    assert reason and isinstance(reason, str)


# ---------------------------------------------------------------------------
# Full manifest contract on the synthetic fixture
# ---------------------------------------------------------------------------


def _build(tmp_path, make_coco_ct, **overrides):
    bbox_path, data = make_coco_ct(
        n_sites=overrides.pop("n_sites", 12),
        seqs_per_site=overrides.pop("seqs_per_site", 4),
        frames_per_seq=overrides.pop("frames_per_seq", 3),
        night_fraction=overrides.pop("night_fraction", 0.3),
        add_null_datetime_image=overrides.pop("add_null_datetime_image", True),
        add_person_box_image=overrides.pop("add_person_box_image", True),
        add_vehicle_box_image=overrides.pop("add_vehicle_box_image", True),
        add_zero_box_image=overrides.pop("add_zero_box_image", True),
    )
    out_dir = tmp_path / overrides.pop("out_dir_name", "manifests")
    args = [
        "--serengeti-bbox-json",
        str(bbox_path),
        "--out-dir",
        str(out_dir),
        "--skip-kgalagadi",
        "--seed",
        str(overrides.pop("seed", 20260916)),
        "--train-size",
        str(overrides.pop("train_size", 20)),
        "--val-size",
        str(overrides.pop("val_size", 4)),
        "--val-site-fraction",
        str(overrides.pop("val_site_fraction", 0.25)),
        "--max-per-sequence",
        str(overrides.pop("max_per_sequence", 1)),
    ]
    rc = build_manifests.main(args)
    manifest_path = out_dir / "serengeti_trainval.jsonl"
    return rc, manifest_path, data


def test_every_row_has_required_fields_and_null_reasons(tmp_path, make_coco_ct):
    rc, manifest_path, _ = _build(tmp_path, make_coco_ct)
    assert rc == 0
    rows = split_check.load_manifest_rows([manifest_path])
    assert rows
    for row in rows:
        for field in split_check.REQUIRED_FIELDS:
            assert field in row, f"missing {field} in {row.get('image_id')}"
        null_reasons = row["null_reasons"]
        for field in split_check.REQUIRED_FIELDS:
            if field == "null_reasons":
                continue
            if row[field] is None:
                assert field in null_reasons, f"{field} is None with no reason on {row['image_id']}"


def test_train_val_site_and_sequence_disjoint(tmp_path, make_coco_ct):
    rc, manifest_path, _ = _build(tmp_path, make_coco_ct)
    assert rc == 0
    rows = split_check.load_manifest_rows([manifest_path])
    train_sites = {r["site_id"] for r in rows if r["split"] == "train"}
    val_sites = {r["site_id"] for r in rows if r["split"] == "val"}
    train_seqs = {r["sequence_id"] for r in rows if r["split"] == "train"}
    val_seqs = {r["sequence_id"] for r in rows if r["split"] == "val"}
    assert not (train_sites & val_sites)
    assert not (train_seqs & val_seqs)

    by_seq = {}
    for row in rows:
        by_seq.setdefault(row["sequence_id"], []).append(row)
    assert all(len(v) <= 1 for v in by_seq.values())  # max_per_sequence=1

    val_rows = [r for r in rows if r["split"] == "val"]
    assert any(r["illumination"] == "day" for r in val_rows)
    assert any(r["illumination"] == "night" for r in val_rows)


def test_person_vehicle_and_null_datetime_excluded(tmp_path, make_coco_ct):
    rc, manifest_path, _ = _build(tmp_path, make_coco_ct)
    assert rc == 0
    rows = split_check.load_manifest_rows([manifest_path])
    for row in rows:
        categories = {b["category"] for b in row["boxes"]}
        assert not (categories & {"person", "vehicle"})
        assert row["datetime"] is not None


def test_build_info_has_exclusion_counts(tmp_path, make_coco_ct):
    rc, manifest_path, _ = _build(tmp_path, make_coco_ct)
    assert rc == 0
    build_info = json.loads((manifest_path.parent / "build_info.json").read_text(encoding="utf-8"))
    excluded = build_info["excluded"]["snapshot_serengeti"]
    assert excluded["person_or_vehicle_box"] >= 2
    assert excluded["illumination_none"] >= 1


def test_determinism_same_seed_identical_bytes(tmp_path, make_coco_ct):
    bbox_path, _ = make_coco_ct(n_sites=12, seqs_per_site=4, frames_per_seq=3, night_fraction=0.3)

    out_dir_a = tmp_path / "a"
    out_dir_b = tmp_path / "b"
    common = [
        "--serengeti-bbox-json",
        str(bbox_path),
        "--skip-kgalagadi",
        "--seed",
        "20260916",
        "--train-size",
        "20",
        "--val-size",
        "4",
        "--val-site-fraction",
        "0.25",
        "--max-per-sequence",
        "1",
    ]
    rc_a = build_manifests.main(["--out-dir", str(out_dir_a)] + common)
    rc_b = build_manifests.main(["--out-dir", str(out_dir_b)] + common)
    assert rc_a == 0 and rc_b == 0

    bytes_a = (out_dir_a / "serengeti_trainval.jsonl").read_bytes()
    bytes_b = (out_dir_b / "serengeti_trainval.jsonl").read_bytes()
    assert bytes_a == bytes_b

    out_dir_c = tmp_path / "c"
    common_diff_seed = list(common)
    seed_idx = common_diff_seed.index("--seed") + 1
    common_diff_seed[seed_idx] = "999"
    rc_c = build_manifests.main(["--out-dir", str(out_dir_c)] + common_diff_seed)
    assert rc_c == 0
    bytes_c = (out_dir_c / "serengeti_trainval.jsonl").read_bytes()
    assert bytes_c != bytes_a


def test_sample_rank_prefix_mixes_sites(tmp_path, make_coco_ct):
    rc, manifest_path, _ = _build(
        tmp_path,
        make_coco_ct,
        n_sites=12,
        seqs_per_site=4,
        frames_per_seq=3,
        night_fraction=0.3,
        train_size=20,
        val_size=4,
        val_site_fraction=0.25,
        add_null_datetime_image=False,
        add_person_box_image=False,
        add_vehicle_box_image=False,
        add_zero_box_image=False,
    )
    assert rc == 0
    rows = split_check.load_manifest_rows([manifest_path])
    train_rows = sorted((r for r in rows if r["split"] == "train"), key=lambda r: r["sample_rank"])
    n_train_sites = len({r["site_id"] for r in train_rows})
    prefix = train_rows[: min(20, len(train_rows))]
    prefix_sites = {r["site_id"] for r in prefix}
    assert len(prefix_sites) >= min(20, n_train_sites)


def test_train_size_larger_than_pool_raises(tmp_path, make_coco_ct):
    bbox_path, _ = make_coco_ct(n_sites=3, seqs_per_site=2, frames_per_seq=2, night_fraction=0.3)
    out_dir = tmp_path / "manifests"
    rc = build_manifests.main(
        [
            "--serengeti-bbox-json",
            str(bbox_path),
            "--out-dir",
            str(out_dir),
            "--skip-kgalagadi",
            "--seed",
            "20260916",
            "--train-size",
            "9999",
            "--val-size",
            "1",
            "--val-site-fraction",
            "0.25",
            "--max-per-sequence",
            "1",
        ]
    )
    assert rc != 0


# ---------------------------------------------------------------------------
# Kgalagadi
# ---------------------------------------------------------------------------


def _make_kgalagadi_zip(tmp_path, n_sites=5, seqs_per_site=6, empty_fraction=0.5, add_human=True):
    images = []
    annotations = []
    categories = [
        {"id": 0, "name": "empty"},
        {"id": 1, "name": "human"},
        {"id": 2, "name": "gemsbokoryx"},
        {"id": 3, "name": "ostrich"},
    ]
    img_counter = 1
    total_seqs = n_sites * seqs_per_site
    empty_seqs = int(round(total_seqs * empty_fraction))
    seq_index = 0
    for site_index in range(n_sites):
        site = f"A{site_index:02d}"
        for seq_num in range(seqs_per_site):
            seq_id = f"KGA_S1#{site}#1#{seq_num}"
            is_empty = seq_index < empty_seqs
            stem = f"KGA_S1_{site}_R1_IMAG{img_counter:04d}"
            file_name = f"KGA_S1/{site}/{site}_R1/{stem}.JPG"
            image_id = f"KGA_S1/{site}/{site}_R1/{stem}"
            images.append(
                {
                    "id": image_id,
                    "file_name": file_name,
                    "frame_num": 1,
                    "seq_id": seq_id,
                    "width": 2592,
                    "height": 2000,
                    "corrupt": False,
                    "location": site,
                    "seq_num_frames": 1,
                    "datetime": "2018-10-27 12:00:00",
                }
            )
            if add_human and seq_index == 0:
                category_id = 1
            elif is_empty:
                category_id = 0
            else:
                category_id = 2 if seq_index % 2 == 0 else 3
            annotations.append(
                {
                    "id": f"ann{img_counter}",
                    "category_id": category_id,
                    "seq_id": seq_id,
                    "image_id": image_id,
                    "location": site,
                    "datetime": "2018-10-27 12:00:00",
                }
            )
            img_counter += 1
            seq_index += 1

    data = {
        "info": {"version": "1.0"},
        "categories": categories,
        "images": images,
        "annotations": annotations,
    }
    out_path = tmp_path / "kgalagadi.zip"
    with zipfile.ZipFile(out_path, "w") as zf:
        zf.writestr("SnapshotKgalagai_S1_v1.0.json", json.dumps(data))
    return out_path, data


def test_kgalagadi_site_coverage_and_exclusions(tmp_path):
    kga_path, data = _make_kgalagadi_zip(tmp_path, n_sites=5, seqs_per_site=6, empty_fraction=0.5)
    coco, meta = build_manifests.load_coco_ct(kga_path)
    report = {}
    rows = build_manifests.build_kgalagadi_rows(
        coco, meta, seed=20260916, nonempty_size=100, empty_size=50, report=report
    )
    nonempty_rows = [r for r in rows if r["subset"] == "nonempty"]
    empty_rows = [r for r in rows if r["subset"] == "empty_check"]

    nonempty_sites = {r["site_id"] for r in nonempty_rows}
    # every site with a non-empty candidate must be covered given a generous budget
    assert len(nonempty_sites) == report["sites_covered"]
    assert report["sites_covered"] >= 1

    for row in rows:
        assert row["site_id"].startswith("KGA:")
        assert row["split"] == "test"
        assert row["boxes"] is None
        assert "boxes" in row["null_reasons"]

    for row in empty_rows:
        assert row["is_empty"] is True
    for row in nonempty_rows:
        assert row["is_empty"] is False

    # the human-labeled image must never appear
    assert all("human" not in (r["species"] or []) for r in rows)
    assert report["excluded"]["human_label"] >= 1


def test_kgalagadi_no_annotation_excluded(tmp_path):
    kga_path, data = _make_kgalagadi_zip(tmp_path, n_sites=3, seqs_per_site=3, empty_fraction=0.3)
    coco, meta = build_manifests.load_coco_ct(kga_path)
    # drop annotations for the first image to simulate an unlabeled image
    coco["annotations"] = [a for a in coco["annotations"] if a["image_id"] != coco["images"][0]["id"]]
    report = {}
    rows = build_manifests.build_kgalagadi_rows(
        coco, meta, seed=20260916, nonempty_size=50, empty_size=50, report=report
    )
    assert all(r["image_id"] != f"KGA:{coco['images'][0]['id']}" for r in rows)
    assert report["excluded"]["no_annotation"] >= 1
