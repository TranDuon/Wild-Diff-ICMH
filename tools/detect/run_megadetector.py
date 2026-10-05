"""Run MegaDetector V6 over original frames or a decode archive (DATA-06, EVAL-02).

Run it in its own environment (``pip install PytorchWildlife``), never in the
training environment: its ultralytics pins conflict with the codec stack.  The
output is the JSONL sidecar every consumer already reads -- one line per image,
including images with no detection -- so the H2 preflight, ``CameraTrapDataset``
and ``tools/evaluate_kgalagadi.py`` accept it unchanged:

    {"image_id": ..., "file": <relative path>, "width": W, "height": H,
     "detections": [{"category": "1", "conf": 0.93, "bbox": [x, y, w, h]}]}

``category`` follows the MegaDetector convention (1 animal, 2 person,
3 vehicle) and ``bbox`` is normalised xywh.  Detections are kept down to
``--min-conf`` (default 0.01) so mAP can be computed; consumers apply their
own threshold (0.2 for pseudo ground truth).  Resumable.

Original frames:   --image-root /content/data/.../images
Reconstructions:   --image-root <decode archive> --suffix .png
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.baselines.run_classical import load_rows  # noqa: E402

DEFAULT_VERSION = "MDV6-yolov9-e"


def to_detections(xyxy, confidence, class_id, width: int, height: int) -> list[dict]:
    """Convert PytorchWildlife pixel xyxy boxes to MegaDetector-style records."""
    records = []
    for (x1, y1, x2, y2), conf, cls in zip(xyxy, confidence, class_id):
        records.append({
            "category": str(int(cls) + 1),
            "conf": round(float(conf), 4),
            "bbox": [
                round(float(x1) / width, 6), round(float(y1) / height, 6),
                round(float(x2 - x1) / width, 6), round(float(y2 - y1) / height, 6),
            ],
        })
    return sorted(records, key=lambda record: -record["conf"])


def image_path(root: Path, relative: str, suffix: str | None) -> Path:
    path = root / relative
    return path.with_suffix(suffix) if suffix else path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/manifests/kgalagadi_site_split.jsonl")
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--suffix", default=None, help="e.g. .png for a decode archive")
    parser.add_argument("--split", default=None, help="restrict to one split (default: every image)")
    parser.add_argument("--dev-list", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument("--min-conf", type=float, default=0.01)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    import numpy as np
    from PIL import Image
    from PytorchWildlife.models import detection as pw_detection

    if args.split:
        rows = load_rows(args.manifest, args.split, args.dev_list, args.limit)
    else:
        rows = [json.loads(line) for line in Path(args.manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
        rows.sort(key=lambda row: row["image_id"])
        rows = rows[:args.limit] if args.limit else rows
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if output.is_file():
        for line in output.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["image_id"])
            except (json.JSONDecodeError, KeyError):
                continue  # a line cut by a dead runtime is detected again
    pending = [row for row in rows if row["image_id"] not in done]
    print(f"{len(done)} already detected, {len(pending)} to go", flush=True)

    model = pw_detection.MegaDetectorV6(device=args.device, pretrained=True, version=args.version)
    output.with_suffix(".info.json").write_text(json.dumps({
        "created_at": datetime.now(timezone.utc).isoformat(),
        "detector": f"MegaDetectorV6 {args.version} (PytorchWildlife)",
        "min_conf": args.min_conf,
        "image_root": args.image_root,
        "suffix": args.suffix,
        "categories": {"1": "animal", "2": "person", "3": "vehicle"},
        "bbox": "normalised xywh",
    }, indent=2) + "\n", encoding="utf-8")

    root = Path(args.image_root)
    started = time.perf_counter()
    with output.open("a", encoding="utf-8", newline="\n") as stream:
        for index, row in enumerate(pending, 1):
            path = image_path(root, row["relative_path"], args.suffix)
            with Image.open(path) as opened:
                array = np.asarray(opened.convert("RGB"))
            height, width = array.shape[:2]
            result = model.single_image_detection(array, img_path=str(path), det_conf_thres=args.min_conf)
            found = result["detections"]
            stream.write(json.dumps({
                "image_id": row["image_id"],
                "file": row["relative_path"],
                "width": width,
                "height": height,
                "detections": to_detections(found.xyxy, found.confidence, found.class_id, width, height),
            }, sort_keys=True) + "\n")
            if index % 200 == 0 or index == len(pending):
                stream.flush()
                rate = index / (time.perf_counter() - started)
                print(f"detected {index}/{len(pending)} ({rate:.1f} img/s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
