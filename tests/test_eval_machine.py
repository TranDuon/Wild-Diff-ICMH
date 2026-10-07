import json

import pytest

pytest.importorskip("pycocotools")

from tools.detect.run_megadetector import to_detections
from tools.eval_machine import animal_boxes, image_flags, main, summarize


def _record(image_id, boxes):
    return {"image_id": image_id, "width": 1000, "height": 500,
            "detections": [{"category": c, "conf": s, "bbox": b} for c, s, b in boxes]}


ANIMAL_BOX = ("1", 0.9, [0.1, 0.2, 0.2, 0.2])  # 200 x 100 px: a "large" animal


def test_megadetector_boxes_become_normalised_xywh_categories():
    records = to_detections([[100, 50, 300, 150]], [0.8], [0], 1000, 500)
    assert records == [{"category": "1", "conf": 0.8, "bbox": [0.1, 0.1, 0.2, 0.2]}]


def test_person_and_low_confidence_boxes_are_not_animals():
    record = _record("a", [ANIMAL_BOX, ("2", 0.99, [0, 0, 0.1, 0.1]), ("1", 0.05, [0, 0, 0.1, 0.1])])
    assert animal_boxes(record, 0.2) == [(100.0, 100.0, 200.0, 100.0, 0.9)]


def test_flags_separate_hallucination_from_missed_animals():
    empty = {"row": {"is_empty": True}, "gt": _record("e", []), "pred": _record("e", [ANIMAL_BOX])}
    lost = {"row": {"is_empty": False}, "gt": _record("l", [ANIMAL_BOX]), "pred": _record("l", [])}
    assert image_flags(empty, 0.2) == {"empty_fp": True, "hallucination": True, "missed_animal": None}
    assert image_flags(lost, 0.2) == {"empty_fp": None, "hallucination": None, "missed_animal": True}


def test_perfect_reconstruction_scores_full_ap():
    items = [{"row": {"is_empty": False}, "gt": _record(str(i), [ANIMAL_BOX]),
              "pred": _record(str(i), [ANIMAL_BOX])} for i in range(4)]
    summary = summarize(items, threshold=0.2, gt_threshold=0.2)
    assert summary["map"] == pytest.approx(1.0)
    assert summary["ap_large"] == pytest.approx(1.0)
    assert summary["ap_small"] is None  # no small ground truth
    assert summary["missed_animal_rate"] == 0.0


def test_cli_reports_groups_with_intervals(tmp_path):
    rows = []
    for i in range(6):
        rows.append({"image_id": f"img{i}", "relative_path": f"x/{i}.jpg", "split": "val",
                     "site_id": f"S{i % 3}", "is_empty": i % 2 == 0, "illumination": "day"})
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    gt = [_record(row["image_id"], [] if row["is_empty"] else [ANIMAL_BOX]) for row in rows]
    pred = [_record(row["image_id"], [ANIMAL_BOX] if row["image_id"] == "img0" else
                    ([] if row["is_empty"] else [ANIMAL_BOX])) for row in rows]
    for name, records in (("gt", gt), ("pred", pred)):
        (tmp_path / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    out = tmp_path / "summary.json"
    main(["--manifest", str(manifest), "--dev-list", "", "--illumination-sidecar", "",
          "--gt", str(tmp_path / "gt.jsonl"), "--pred", str(tmp_path / "pred.jsonl"),
          "--bootstrap-resamples", "20", "--output", str(out), "--method", "B0",
          "--results-registry", str(tmp_path / "results.jsonl")])
    groups = json.loads(out.read_text())["groups"]
    overall = groups["illumination=all|content=all"]
    assert overall["hallucination_rate"] == pytest.approx(1 / 3)
    assert overall["ci"]["method"] == "site_cluster"
    assert "hallucination_rate" in overall["ci"]["intervals"]


def test_domain_stats_buckets_shrink_at_the_coding_resolution(tmp_path):
    from tools.data.domain_stats import coco_bucket, mask_coverage, split_stats

    # A 60x60 px animal in a 2592x2000 frame is "medium" originally, "small" at 1024.
    record = {"image_id": "a", "width": 2592, "height": 2000,
              "detections": [{"category": "1", "conf": 0.9, "bbox": [0.5, 0.5, 60 / 2592, 60 / 2000]}]}
    stats = split_stats([{"image_id": "a", "is_empty": False, "illumination": "day"}],
                        {"a": record}, 0.2, 1024)
    assert stats["coco_size_original"] == {"medium": 1}
    assert stats["coco_size_at_long_side_1024"] == {"small": 1}
    assert coco_bucket(100 ** 2) == "large"
    assert 0 < mask_coverage([(0, 0, 1296, 1000, 0.9)], 2592, 2000) <= 0.27
