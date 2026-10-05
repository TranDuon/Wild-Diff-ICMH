"""Machine-consumer metrics of a decode archive (EVAL-03/05/06/07/09).

Inputs are two MegaDetector sidecars written by ``tools/detect/run_megadetector.py``:
``--gt`` on the original frames (pseudo ground truth, DATA-06) and ``--pred`` on
the reconstructions.  Reported per day/night group, with bootstrap intervals
over sites:

* ``map``, ``ap50``, ``ap_small``/``ap_medium``/``ap_large`` -- COCO bbox AP of
  the animal class against the pseudo ground truth (areas in original pixels).
* ``empty_fp_rate`` -- frames a human labelled empty where the reconstruction
  yields an animal detection (EVAL-06).
* ``hallucination_rate`` -- empty frames where the original yields no animal
  but the reconstruction does: the decoder invented one (EVAL-07).
* ``missed_animal_rate`` -- frames with an animal on the original and none on
  the reconstruction.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.baselines.run_classical import load_rows  # noqa: E402
from utils.eval_stats import bootstrap_ci, group_key  # noqa: E402
from utils.illumination import apply_sidecar, load_sidecar  # noqa: E402
from utils.results_registry import metric_rows, upsert_jsonl  # noqa: E402

MACHINE_METRICS = (
    "map", "ap50", "ap_small", "ap_medium", "ap_large",
    "empty_fp_rate", "hallucination_rate", "missed_animal_rate",
)
ANIMAL = "1"


def load_detections(path) -> dict[str, dict]:
    records = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            records[record["image_id"]] = record
    return records


def animal_boxes(record: dict, threshold: float) -> list[tuple[float, float, float, float, float]]:
    """Pixel xywh + score of animal detections at or above ``threshold``."""
    width, height = record["width"], record["height"]
    boxes = []
    for detection in record.get("detections", []):
        if str(detection.get("category")) != ANIMAL or float(detection["conf"]) < threshold:
            continue
        x, y, w, h = detection["bbox"]
        boxes.append((x * width, y * height, w * width, h * height, float(detection["conf"])))
    return boxes


def coco_metrics(items: list[dict], gt_threshold: float) -> dict:
    """COCO AP of the animal class; ``items`` may repeat images (bootstrap)."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    images, annotations, predictions = [], [], []
    for image_id, item in enumerate(items, 1):
        images.append({"id": image_id, "width": item["gt"]["width"], "height": item["gt"]["height"]})
        for x, y, w, h, _ in animal_boxes(item["gt"], gt_threshold):
            annotations.append({
                "id": len(annotations) + 1, "image_id": image_id, "category_id": 1,
                "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0,
            })
        for x, y, w, h, score in animal_boxes(item["pred"], 0.0):
            predictions.append({"image_id": image_id, "category_id": 1, "bbox": [x, y, w, h], "score": score})
    empty = {name: None for name in ("map", "ap50", "ap_small", "ap_medium", "ap_large")}
    if not annotations or not predictions:
        return empty
    with contextlib.redirect_stdout(io.StringIO()):
        truth = COCO()
        truth.dataset = {"images": images, "annotations": annotations, "categories": [{"id": 1, "name": "animal"}]}
        truth.createIndex()
        evaluation = COCOeval(truth, truth.loadRes(predictions), "bbox")
        evaluation.evaluate()
        evaluation.accumulate()
        evaluation.summarize()
    stats = evaluation.stats

    def value(index):
        return None if stats[index] < 0 else float(stats[index])  # -1: no GT of that size

    return {"map": value(0), "ap50": value(1), "ap_small": value(3), "ap_medium": value(4), "ap_large": value(5)}


def image_flags(item: dict, threshold: float) -> dict:
    original = bool(animal_boxes(item["gt"], threshold))
    decoded = bool(animal_boxes(item["pred"], threshold))
    empty = bool(item["row"].get("is_empty"))
    return {
        "empty_fp": decoded if empty else None,
        "hallucination": decoded if (empty and not original) else None,
        "missed_animal": (not decoded) if original else None,
    }


def summarize(items: list[dict], *, threshold: float, gt_threshold: float, with_ap: bool = True) -> dict:
    def rate(name):
        values = [flags[name] for flags in (image_flags(item, threshold) for item in items) if flags[name] is not None]
        return sum(values) / len(values) if values else None

    summary = {
        "n": len(items),
        "empty_fp_rate": rate("empty_fp"),
        "hallucination_rate": rate("hallucination"),
        "missed_animal_rate": rate("missed_animal"),
    }
    if with_ap:
        summary.update(coco_metrics(items, gt_threshold))
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/manifests/kgalagadi_site_split.jsonl")
    parser.add_argument("--split", default="val")
    parser.add_argument("--dev-list", default="data/manifests/kgalagadi_dev.txt")
    parser.add_argument("--illumination-sidecar", default="data/manifests/kgalagadi_illumination.jsonl")
    parser.add_argument("--gt", required=True, help="MegaDetector sidecar on the original frames")
    parser.add_argument("--pred", required=True, help="MegaDetector sidecar on the reconstructions")
    parser.add_argument("--gt-threshold", type=float, default=0.2, help="pseudo ground-truth confidence (DATA-06)")
    parser.add_argument("--threshold", type=float, default=0.2, help="detection threshold for the rates")
    parser.add_argument("--bootstrap-resamples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True, help="summary JSON")
    parser.add_argument("--results-registry", default=None)
    parser.add_argument("--exp-id", default=None)
    parser.add_argument("--method", required=True)
    parser.add_argument("--dataset", default="snapshot_kgalagadi")
    parser.add_argument("--lambda-rate", default="unknown")
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--git-commit", default="unknown")
    args = parser.parse_args(argv)

    rows = load_rows(args.manifest, args.split, args.dev_list or None, None)
    apply_sidecar(rows, load_sidecar(args.illumination_sidecar))
    gt, pred = load_detections(args.gt), load_detections(args.pred)
    missing = [row["image_id"] for row in rows if row["image_id"] not in gt or row["image_id"] not in pred]
    if missing:
        raise ValueError(f"{len(missing)} images lack detections, e.g. {missing[:3]}")
    items = [{"row": row, "gt": gt[row["image_id"]], "pred": pred[row["image_id"]],
              "site_id": row["site_id"]} for row in rows]

    groups = {"all": items}
    for illumination in sorted({str(item["row"].get("illumination")) for item in items}):
        groups[illumination] = [item for item in items if str(item["row"].get("illumination")) == illumination]
    summary = {}
    for illumination, members in groups.items():
        values = summarize(members, threshold=args.threshold, gt_threshold=args.gt_threshold)
        values["ci"] = bootstrap_ci(
            members, MACHINE_METRICS, n_resamples=args.bootstrap_resamples, seed=args.seed,
            summarize=lambda sample: summarize(sample, threshold=args.threshold, gt_threshold=args.gt_threshold),
        )
        summary[group_key(illumination, "all")] = values

    protocol = {
        "pseudo_ground_truth": f"MegaDetector on original frames, animal conf >= {args.gt_threshold}",
        "rate_threshold": args.threshold,
        "coco_areas": "small < 32^2, medium < 96^2, large >= 96^2 original-frame pixels",
        "gt_file": args.gt, "pred_file": args.pred, "dev_list": args.dev_list,
        "illumination_labels": "light_source_sidecar" if load_sidecar(args.illumination_sidecar) else "manifest_capture_hour_proxy",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"protocol": protocol, "groups": summary}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.results_registry:
        upsert_jsonl(args.results_registry, metric_rows(
            summary, exp_id=args.exp_id or f"{args.method}_machine", dataset=args.dataset,
            lambda_rate=args.lambda_rate, ddim_steps=args.ddim_steps, cu_estimate=None,
            git_commit=args.git_commit, method=args.method, split=args.split, site_id=None,
            protocol=protocol, metric_names=MACHINE_METRICS,
        ))
    overall = summary[group_key("all", "all")]
    print(json.dumps({name: overall.get(name) for name in MACHINE_METRICS}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
