"""Evaluate reconstructed Kgalagadi images (EVAL-11/12/13/15).

Reads a decode archive -- ``<root>/<relative>.png`` reconstructions at the
evaluation resolution, ``<root>/<dir>/data/<stem>`` bitstreams and, when the
decoder wrote them, ``decode_log.jsonl`` timings and ``run_info.json`` protocol --
and compares every reconstruction with the original frame.  Baselines (JPEG,
WebP, CompressAI) write the same layout, so one evaluator scores every method.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import subprocess
import sys
from pathlib import Path

# Direct execution (``python tools/evaluate_kgalagadi.py``) makes Python put
# ``tools/`` rather than the repository root first on ``sys.path``.  The Colab
# notebook invokes this file directly, so bootstrap the project packages before
# importing ``dataset`` or ``utils`` below.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from PIL import Image

from dataset.camera_trap_dataset import _box_xywh, _load_detection_map
from tools.data.split_check import assert_no_leakage
from utils.eval_stats import aggregate, bootstrap_ci, stratified_groups
from utils.illumination import RULES as ILLUMINATION_RULES, apply_sidecar, load_sidecar
from utils.image_geometry import center_crop_boxes, center_crop_image, resolve_geometry
from utils.metrics import LPIPS, compute_psnr, compute_ssim, compute_ssim_masked
from utils.results_registry import REGISTRY_METRICS, metric_rows, upsert_jsonl
from tools.baselines.run_classical import archived_paths

PER_IMAGE_METRICS = (
    "psnr", "ssim", "ssim_fullres", "ms_ssim", "foreground_ssim", "lpips", "dists",
    "encode_seconds", "decode_seconds",
)
SSIM_IMPLEMENTATIONS = {
    "ssim": "utils.metrics.compute_ssim (gaussian 11/1.5, auto-downsample by round(min(H,W)/256))",
    "ssim_fullres": "pytorch_msssim.ssim (gaussian 11/1.5, no downsampling, data_range=1)",
    "ms_ssim": "pytorch_msssim.ms_ssim (5 scales, data_range=1)",
}


def load_dev_ids(path) -> set[str]:
    return {
        line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }


def _load_rows(path, split, site_id, dev_list=None):
    with open(path, "r", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    rows = [
        row for row in rows
        if row.get("split") == split and (site_id is None or row.get("site_id") == site_id)
    ]
    if dev_list is not None:
        wanted = load_dev_ids(dev_list)
        rows = [row for row in rows if row["image_id"] in wanted]
        missing = wanted - {row["image_id"] for row in rows}
        if missing and site_id is None:
            raise ValueError(f"dev list has {len(missing)} ids outside split={split!r}")
    return rows


def _illumination(row):
    return str(row.get("illumination") or "unknown")


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


def _tensor(path, crop_size=None):
    image = Image.open(path).convert("RGB")
    if crop_size is not None:
        image = center_crop_image(image, crop_size)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def _read_jsonl_by(path: Path, key: str) -> dict:
    records = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
                records[record[key]] = record
            except (json.JSONDecodeError, KeyError):
                continue
    return records


def _version(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_commit():
    process = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    return process.stdout.strip() if process.returncode == 0 else "unknown"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/manifests/kgalagadi_site_split.jsonl")
    parser.add_argument("--build-info", default="data/manifests/build_info.json")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--reconstruction-root", required=True)
    parser.add_argument("--detections", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--site-id", default=None)
    parser.add_argument("--dev-list", default=None, help="frozen image_id list (EVAL-16), e.g. kgalagadi_dev.txt")
    parser.add_argument(
        "--illumination-sidecar", default="data/manifests/kgalagadi_illumination.jsonl",
        help="light-source day/night labels from tools/data/label_illumination.py; "
             "the manifest capture-hour proxy is used only if the file is absent",
    )
    parser.add_argument("--method", required=True, help="B0, H1, H2, H3, jpeg, webp, compressai:<model>")
    parser.add_argument("--dataset", default="snapshot_kgalagadi")
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N filtered rows")
    parser.add_argument(
        "--crop-size", type=int, default=None,
        help="smoke protocol only: apply the same center crop used during decoding. "
             "Without it, reconstructions must be at the original resolution (EVAL-11)",
    )
    parser.add_argument(
        "--archived-only", action="store_true",
        help="score only the images listed in <reconstruction-root>/decode_log.jsonl",
    )
    parser.add_argument("--min-detection-confidence", type=float, default=0.2)
    parser.add_argument("--lpips", action="store_true")
    parser.add_argument("--dists", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--results-registry", default=None, help="optional canonical results.jsonl")
    parser.add_argument("--exp-id", default=None)
    parser.add_argument("--lambda-rate", default="unknown")
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--cu-estimate", type=float, default=None)
    parser.add_argument("--git-commit", default=None)
    args = parser.parse_args(argv)

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    try:
        # Evaluation never resizes: the reference is the original frame unless a
        # smoke crop is requested explicitly.
        args.crop_size, _ = resolve_geometry(args.crop_size, None, manifest_supplied=False)
    except ValueError as exc:
        parser.error(str(exc))

    from pytorch_msssim import ms_ssim, ssim as ssim_fullres

    assert_no_leakage([Path(args.manifest)], build_info=args.build_info)
    rows = _load_rows(args.manifest, args.split, args.site_id, args.dev_list)
    if args.archived_only:
        archived = archived_paths(args.reconstruction_root)
        rows = [row for row in rows if row["relative_path"] in archived]
    if args.limit is not None:
        rows = rows[:args.limit]
    if not rows:
        parser.error("manifest filter selected no rows")
    illumination_sidecar = load_sidecar(args.illumination_sidecar)
    relabelled = apply_sidecar(rows, illumination_sidecar)
    if illumination_sidecar and relabelled != len(rows):
        raise ValueError(
            f"illumination sidecar covers {relabelled}/{len(rows)} selected rows; "
            "re-run tools/data/label_illumination.py"
        )
    illumination_labels = "light_source_sidecar" if relabelled else "manifest_capture_hour_proxy"
    print(f"Day/night labels: {illumination_labels}")
    detections = _load_detection_map(args.detections)
    if args.detections:
        uncovered = [
            row["image_id"] for row in rows
            if not any(str(row.get(key)) in detections for key in ("image_id", "source_file_name", "relative_path"))
        ]
        if uncovered:
            preview = ", ".join(uncovered[:5])
            raise ValueError(f"detection sidecar has no record for {len(uncovered)} images: {preview}")
    device = torch.device(args.device)
    lpips_metric = LPIPS("alex").to(args.device) if args.lpips else None
    dists_metric = None
    if args.dists:
        import pyiqa

        dists_metric = pyiqa.create_metric("dists", device=device)
    data_root, reconstruction_root = Path(args.data_root), Path(args.reconstruction_root)
    decode_log = _read_jsonl_by(reconstruction_root / "decode_log.jsonl", "relative_path")
    run_info_path = reconstruction_root / "run_info.json"
    run_info = json.loads(run_info_path.read_text(encoding="utf-8")) if run_info_path.is_file() else {}

    results = []
    for index, row in enumerate(rows, 1):
        relative = Path(row["relative_path"])
        source_path = data_root / relative
        reconstructed_path = (reconstruction_root / relative).with_suffix(".png")
        stream_path = reconstructed_path.parent / "data" / reconstructed_path.stem
        timing = decode_log.get(relative.as_posix(), {})
        # Autoregressive CompressAI models log an estimated rate instead of a file.
        estimated = timing.get("estimated_bits") if not stream_path.is_file() else None
        if not source_path.is_file() or not reconstructed_path.is_file() or (
            not stream_path.is_file() and estimated is None
        ):
            raise FileNotFoundError(
                f"missing source/reconstruction/bitstream for {row['image_id']}: "
                f"{source_path}, {reconstructed_path}, {stream_path}"
            )
        with Image.open(source_path) as source_image:
            original_width, original_height = source_image.size
        source = _tensor(source_path, crop_size=args.crop_size).to(device)
        reconstruction = _tensor(reconstructed_path).to(device)
        if source.shape != reconstruction.shape:
            raise ValueError(
                f"shape mismatch for {row['image_id']}: {tuple(source.shape)} vs "
                f"{tuple(reconstruction.shape)}; reconstructions must be restored to the "
                "original frame (or decoded with the same --crop-size)"
            )
        _, _, height, width = source.shape
        boxes = _boxes_for(
            row, detections, original_width, original_height,
            args.min_detection_confidence,
        )
        if args.crop_size is not None:
            boxes = center_crop_boxes(
                boxes, original_width, original_height, args.crop_size
            )
        mask = _mask_from_boxes(width, height, boxes)
        bitstream_bytes = stream_path.stat().st_size if estimated is None else estimated / 8
        bpp = bitstream_bytes * 8 / (height * width)
        with torch.no_grad():
            result = {
                "schema_version": 2,
                "method": args.method,
                "dataset": args.dataset,
                "split": args.split,
                "image_id": row["image_id"],
                "relative_path": relative.as_posix(),
                "site_id": row["site_id"],
                "sequence_id": row["sequence_id"],
                "illumination": _illumination(row),
                "is_empty": row.get("is_empty"),
                "bpp": bpp,
                "compression_ratio_rgb24": 24.0 / bpp,
                "psnr": compute_psnr(source, reconstruction).item(),
                "ssim": compute_ssim(source, reconstruction).item(),
                "ssim_fullres": ssim_fullres(source, reconstruction, data_range=1.0).item(),
                "ms_ssim": ms_ssim(source, reconstruction, data_range=1.0).item(),
                "foreground_ssim": compute_ssim_masked(source, reconstruction, mask).item() if boxes else None,
                "lpips": lpips_metric(source, reconstruction, normalize=True).mean().item() if lpips_metric else None,
                "dists": dists_metric(reconstruction, source).mean().item() if dists_metric else None,
                "encode_seconds": timing.get("encode_seconds"),
                "decode_seconds": timing.get("decode_seconds"),
                "coded_size": timing.get("coded_size"),
                "bitstream_bytes": bitstream_bytes,
                "rate_source": "coded" if estimated is None else "estimated",
                "width": width,
                "height": height,
            }
        results.append(result)
        if index % 25 == 0 or index == len(rows):
            print(f"evaluated {index}/{len(rows)}", flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for result in results:
            stream.write(json.dumps(result, sort_keys=True) + "\n")

    metric_names = ("bpp", "compression_ratio_rgb24", *PER_IMAGE_METRICS)
    summary = {}
    for group, group_rows in stratified_groups(results, lambda item: item["illumination"]).items():
        summary[group] = aggregate(group_rows, metric_names)
        summary[group]["ci"] = bootstrap_ci(
            group_rows, metric_names,
            n_resamples=args.bootstrap_resamples, seed=args.seed,
        )
    for site in sorted({row["site_id"] for row in results}):
        summary[f"site_id={site}"] = aggregate(
            [row for row in results if row["site_id"] == site], metric_names
        )

    protocol = {
        "evaluation_resolution": f"center_crop_{args.crop_size}" if args.crop_size else "original",
        "processing_long_side": run_info.get("processing_long_side"),
        "decode_protocol": run_info.get("protocol"),
        "resampling": run_info.get("resampling"),
        "sampler": run_info.get("sampler"),
        "seed": run_info.get("seed"),
        "c_cfg_scale": run_info.get("c_cfg_scale"),
        "checkpoint": run_info.get("checkpoint"),
        "ssim_implementations": SSIM_IMPLEMENTATIONS,
        "metric_versions": {
            "pyiqa": _version("pyiqa"), "pytorch_msssim": _version("pytorch-msssim"),
        },
        "compression_ratio": "24*H*W / bitstream bits, H and W of the evaluation reference",
        "rate": run_info.get("rate", "coded"),
        "bootstrap": f"{args.bootstrap_resamples} resamples, site clusters when >1 site",
        "dev_list": args.dev_list,
        "illumination_labels": illumination_labels,
        "illumination_rules": ILLUMINATION_RULES if relabelled else None,
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps({"protocol": protocol, "groups": summary}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.results_registry:
        exp_id = args.exp_id or f"{args.method}_{args.site_id or 'all'}_{args.split}"
        registry_rows = metric_rows(
            summary,
            exp_id=exp_id,
            dataset=args.dataset,
            lambda_rate=args.lambda_rate,
            ddim_steps=args.ddim_steps,
            cu_estimate=args.cu_estimate,
            git_commit=args.git_commit or _git_commit(),
            method=args.method,
            split=args.split,
            site_id=args.site_id,
            protocol=protocol,
            metric_names=REGISTRY_METRICS,
        )
        upsert_jsonl(args.results_registry, registry_rows)
        print(f"Upserted {len(registry_rows)} metric rows into {args.results_registry}")
    print(f"Wrote {len(results)} rows to {output} and {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
