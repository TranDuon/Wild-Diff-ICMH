"""Evaluate reconstructed Kgalagadi images with the comparison-paper metrics."""
from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from dataset.camera_trap_dataset import _box_xywh, _load_detection_map
from tools.data.split_check import assert_no_leakage
from utils.metrics import LPIPS, compute_psnr, compute_ssim, compute_ssim_masked


def _load_rows(path, split, site_id):
    with open(path, "r", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    return [
        row for row in rows
        if row.get("split") == split and (site_id is None or row.get("site_id") == site_id)
    ]


def _boxes_for(row, detections, width, height, threshold):
    raw = row.get("boxes")
    if not isinstance(raw, list):
        keys = (str(row.get("image_id")), str(row.get("source_file_name")), str(row.get("relative_path")))
        raw = next((detections[key] for key in keys if key in detections), [])
    boxes = []
    for value in raw:
        if isinstance(value, dict):
            box = _box_xywh(value, width, height, threshold)
            if box:
                boxes.append(box)
    return boxes


def _mask_from_boxes(width, height, boxes):
    mask = np.zeros((height, width), dtype=np.float32)
    for x, y, w, h in boxes:
        x0, y0 = int(math.floor(x)), int(math.floor(y))
        x1, y1 = int(math.ceil(x + w)), int(math.ceil(y + h))
        mask[max(0, y0):min(height, y1), max(0, x0):min(width, x1)] = 1.0
    return torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)


def _tensor(path):
    array = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def _mean(values):
    finite = [value for value in values if value is not None and math.isfinite(value)]
    return sum(finite) / len(finite) if finite else None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/manifests/kgalagadi_site_split.jsonl")
    parser.add_argument("--build-info", default="data/manifests/build_info.json")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--reconstruction-root", required=True)
    parser.add_argument("--detections", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--site-id", default=None)
    parser.add_argument("--method", required=True, help="baseline, H1, H2 or H3")
    parser.add_argument("--min-detection-confidence", type=float, default=0.2)
    parser.add_argument("--lpips", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    assert_no_leakage([Path(args.manifest)], build_info=args.build_info)
    rows = _load_rows(args.manifest, args.split, args.site_id)
    if not rows:
        parser.error("manifest filter selected no rows")
    detections = _load_detection_map(args.detections)
    if args.detections:
        uncovered = [
            row["image_id"] for row in rows
            if not any(str(row.get(key)) in detections for key in ("image_id", "source_file_name", "relative_path"))
        ]
        if uncovered:
            preview = ", ".join(uncovered[:5])
            raise ValueError(f"detection sidecar has no record for {len(uncovered)} images: {preview}")
    lpips_metric = LPIPS("alex") if args.lpips else None
    data_root, reconstruction_root = Path(args.data_root), Path(args.reconstruction_root)
    results = []
    for row in rows:
        relative = Path(row["relative_path"])
        source_path = data_root / relative
        reconstructed_path = (reconstruction_root / relative).with_suffix(".png")
        stream_path = reconstructed_path.parent / "data" / reconstructed_path.stem
        if not source_path.is_file() or not reconstructed_path.is_file() or not stream_path.is_file():
            raise FileNotFoundError(
                f"missing source/reconstruction/bitstream for {row['image_id']}: "
                f"{source_path}, {reconstructed_path}, {stream_path}"
            )
        source, reconstruction = _tensor(source_path), _tensor(reconstructed_path)
        if source.shape != reconstruction.shape:
            raise ValueError(f"shape mismatch for {row['image_id']}: {source.shape} vs {reconstruction.shape}")
        _, _, height, width = source.shape
        boxes = _boxes_for(row, detections, width, height, args.min_detection_confidence)
        mask = _mask_from_boxes(width, height, boxes)
        bitstream_bits = stream_path.stat().st_size * 8
        bpp = bitstream_bits / (height * width)
        foreground_ssim = compute_ssim_masked(source, reconstruction, mask).item() if boxes else None
        result = {
            "schema_version": 1,
            "method": args.method,
            "dataset": "snapshot_kgalagadi",
            "split": args.split,
            "image_id": row["image_id"],
            "site_id": row["site_id"],
            "sequence_id": row["sequence_id"],
            "illumination": row.get("illumination"),
            "is_empty": row.get("is_empty"),
            "bpp": bpp,
            "compression_ratio_rgb24": 24.0 / bpp,
            "psnr": compute_psnr(source, reconstruction).item(),
            "ssim": compute_ssim(source, reconstruction).item(),
            "foreground_ssim": foreground_ssim,
            "lpips": lpips_metric(source, reconstruction, normalize=True).mean().item() if lpips_metric else None,
            "bitstream_bytes": stream_path.stat().st_size,
            "width": width,
            "height": height,
        }
        results.append(result)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for result in results:
            stream.write(json.dumps(result, sort_keys=True) + "\n")

    groups = {"all": results}
    for key in ("illumination", "site_id"):
        for value in sorted({str(row[key]) for row in results}):
            groups[f"{key}={value}"] = [row for row in results if str(row[key]) == value]
    summary = {}
    metric_names = ("bpp", "compression_ratio_rgb24", "psnr", "ssim", "foreground_ssim", "lpips")
    for group, group_rows in groups.items():
        total_pixels = sum(row["width"] * row["height"] for row in group_rows)
        aggregate_bpp = sum(row["bitstream_bytes"] * 8 for row in group_rows) / total_pixels
        summary[group] = {
            "n": len(group_rows),
            "pixels": total_pixels,
            "bpp": aggregate_bpp,
            "compression_ratio_rgb24": 24.0 / aggregate_bpp,
        }
        summary[group].update({
            name: _mean([row[name] for row in group_rows])
            for name in metric_names
            if name not in {"bpp", "compression_ratio_rgb24"}
        })
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {len(results)} rows to {output} and {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
