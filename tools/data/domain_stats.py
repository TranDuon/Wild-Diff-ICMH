"""Domain statistics of the corpus (DATA-04/06): what the codec actually faces.

Per split: human-labelled empty ratio, day/night by light source (with the
signal each label came from), animal-box sizes as a fraction of the frame and
in COCO size buckets -- both at the original resolution and at the coding
resolution, where small animals get smaller -- and the ROI-mask coverage that
H2's loss weighting will see.  Boxes are MegaDetector pseudo labels.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.eval_machine import animal_boxes, load_detections  # noqa: E402
from utils.illumination import apply_sidecar, load_sidecar  # noqa: E402
from utils.image_geometry import DEFAULT_PROCESSING_LONG_SIDE, processing_size  # noqa: E402

AREA_FRACTION_BINS = (0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.01)
COVERAGE_BINS = (0.0, 1e-9, 0.01, 0.05, 0.1, 0.25, 0.5, 1.01)
COVERAGE_GRID = (128, 99)  # union of boxes rasterised on a coarse grid


def coco_bucket(area: float) -> str:
    return "small" if area < 32 ** 2 else ("medium" if area < 96 ** 2 else "large")


def histogram(values, bins) -> dict[str, int]:
    counts = collections.Counter()
    for value in values:
        for low, high in zip(bins, bins[1:]):
            if low <= value < high:
                counts[f"[{low:g},{min(high, 1):g})"] += 1
                break
    return {f"[{low:g},{min(high, 1):g})": counts.get(f"[{low:g},{min(high, 1):g})", 0) for low, high in zip(bins, bins[1:])}


def mask_coverage(boxes, width: int, height: int) -> float:
    cols, rows = COVERAGE_GRID
    covered = set()
    for x, y, w, h, _ in boxes:
        for gx in range(int(x / width * cols), min(cols, int((x + w) / width * cols) + 1)):
            for gy in range(int(y / height * rows), min(rows, int((y + h) / height * rows) + 1)):
                covered.add((gx, gy))
    return len(covered) / (cols * rows)


def split_stats(rows, detections, threshold, long_side) -> dict:
    boxes_by_image = {
        row["image_id"]: animal_boxes(detections[row["image_id"]], threshold)
        for row in rows if row["image_id"] in detections
    }
    fractions, buckets_original, buckets_coded, coverages = [], collections.Counter(), collections.Counter(), []
    for row in rows:
        record = detections.get(row["image_id"])
        if record is None:
            continue
        width, height = record["width"], record["height"]
        coded_w, coded_h = processing_size(width, height, long_side)
        scale = (coded_w / width) * (coded_h / height)
        boxes = boxes_by_image[row["image_id"]]
        for x, y, w, h, _ in boxes:
            fractions.append(w * h / (width * height))
            buckets_original[coco_bucket(w * h)] += 1
            buckets_coded[coco_bucket(w * h * scale)] += 1
        coverages.append(mask_coverage(boxes, width, height))
    detected = sum(1 for boxes in boxes_by_image.values() if boxes)
    empty_human = sum(1 for row in rows if row.get("is_empty"))
    return {
        "images": len(rows),
        "with_detections_file": len(boxes_by_image),
        "empty_ratio_human_labels": empty_human / len(rows) if rows else None,
        "images_with_animal_detection": detected,
        "illumination": dict(collections.Counter(str(row.get("illumination")) for row in rows)),
        "illumination_source": dict(collections.Counter(str(row.get("illumination_source")) for row in rows)),
        "animal_boxes": len(fractions),
        "box_area_fraction_histogram": histogram(fractions, AREA_FRACTION_BINS),
        "coco_size_original": dict(buckets_original),
        f"coco_size_at_long_side_{long_side}": dict(buckets_coded),
        "roi_mask_coverage_histogram": histogram(coverages, COVERAGE_BINS),
        "roi_mask_coverage_mean": sum(coverages) / len(coverages) if coverages else None,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/manifests/kgalagadi_site_split.jsonl")
    parser.add_argument("--illumination-sidecar", default="data/manifests/kgalagadi_illumination.jsonl")
    parser.add_argument("--detections", required=True, help="MegaDetector sidecar on the original frames")
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--processing-long-side", type=int, default=DEFAULT_PROCESSING_LONG_SIDE)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    rows = [json.loads(line) for line in Path(args.manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
    apply_sidecar(rows, load_sidecar(args.illumination_sidecar))
    detections = load_detections(args.detections)
    report = {
        "detector_threshold": args.threshold,
        "processing_long_side": args.processing_long_side,
        "pseudo_labels": "MegaDetector boxes, not human annotation",
        "splits": {},
    }
    for split in ("all", "train", "val", "test"):
        members = rows if split == "all" else [row for row in rows if row["split"] == split]
        if members:
            report["splits"][split] = split_stats(members, detections, args.threshold, args.processing_long_side)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["splits"]["all"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
