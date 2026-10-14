"""Small, real-transform ROI evidence and cache-preserving Colab staging.

This module never starts training, downloads a model or installs a package.
An empty detection list is valid evidence; a missing record is not.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import random
import shutil
import tempfile


AUTHOR_RELATIVE = "difficmh_models/CNscale1.0_1_1_2_2_WTagGCM_bs16x1_lr0.00005_cfg7.0/model.ckpt"
CHECKPOINT_RELATIVES = ("sd2p1/v2-1_512-ema-pruned.ckpt", AUTHOR_RELATIVE)
TRANSFORMS = {"processing_long_side": 512, "out_size": 256, "crop_type": "random",
              "bbox_crop_probability": 0.0, "use_hflip": True,
              "min_detection_confidence": 0.2, "latent_downscale": 8}


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def contained_path(root, relative):
    """Accept only portable, relative manifest paths, including on Windows."""
    root = Path(root).resolve()
    value = str(relative).replace("\\", "/")
    part = PurePosixPath(value)
    if not value or part.is_absolute() or any(p in {"..", "."} for p in value.split("/")) or ":" in value:
        raise ValueError(f"Unsafe relative asset path: {relative!r}")
    target = (root / Path(*part.parts)).resolve()
    if target == root or not target.is_relative_to(root):
        raise ValueError(f"Asset path escapes its root: {relative!r}")
    return target


def _copy_if_needed(source, target):
    source, target = Path(source), Path(target)
    if not source.is_file() or source.stat().st_size <= 0:
        raise FileNotFoundError(f"Missing/empty source asset: {source}")
    size = source.stat().st_size
    if target.is_file() and target.stat().st_size == size:
        return False, size
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=target.name + ".partial-", dir=target.parent)
    os.close(fd)
    try:
        shutil.copyfile(source, temporary)
        if Path(temporary).stat().st_size != size:
            raise OSError(f"Incomplete copy: {source}")
        os.replace(temporary, target)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()
    return True, size


def stage_assets(drive_root, local_images, relative_paths, *, include_checkpoints=False, checkpoint_root=None):
    """Copy this action's exact selection, skipping already complete local files.

    Source images live under DRIVE_ROOT/images; their manifest paths include
    snapshot_kgalagadi/. Checkpoints are the fixed SD and author lambda2 only.
    Size checks are for cache reuse, not a replacement for experiment digests.
    """
    root = Path(drive_root)
    selected = list(dict.fromkeys(map(str, relative_paths)))
    jobs = [(contained_path(root / "images", relative), contained_path(local_images, relative))
            for relative in selected]
    if include_checkpoints:
        if checkpoint_root is None:
            raise ValueError("checkpoint_root is required for checkpoint staging")
        jobs.extend((contained_path(root / "checkpoints", relative), contained_path(checkpoint_root, relative))
                    for relative in CHECKPOINT_RELATIVES)
    stats = {"requested_images": len(selected), "requested_files": len(jobs), "copied_files": 0,
             "reused_files": 0, "copied_bytes": 0, "reused_bytes": 0}
    # Validate all sources/paths first; a malformed later record cannot cause a
    # partial corpus copy before its rejection.
    for source, _ in jobs:
        if not source.is_file() or source.stat().st_size <= 0:
            raise FileNotFoundError(f"Missing/empty source asset: {source}")
    print(f"H2 staging: {len(selected)} selected originals; existing files are preserved.", flush=True)
    for index, (source, target) in enumerate(jobs, 1):
        copied, size = _copy_if_needed(source, target)
        prefix = "copied" if copied else "reused"
        stats[prefix + "_files"] += 1
        stats[prefix + "_bytes"] += size
        if copied or index == len(jobs) or index % 500 == 0:
            print(f"  {index}/{len(jobs)}: {prefix}; copied={stats['copied_files']}, reused={stats['reused_files']}", flush=True)
    return stats


def _jsonl(path):
    rows = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"Expected an object at {path}:{number}")
        rows.append(row)
    return rows


def _detections(path):
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        rows = _jsonl(path)
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("images", payload) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("Detection evidence must be a list of records")
    result = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Malformed detection evidence record")
        key = row.get("image_id") or row.get("id") or row.get("file")
        boxes = row.get("boxes", row.get("detections"))
        if not key or not isinstance(boxes, list) or str(key) in result:
            raise ValueError("Missing/duplicate ID or missing detection list in evidence")
        for box in boxes:
            coordinates = box.get("bbox_xywh", box.get("bbox")) if isinstance(box, dict) else None
            if not isinstance(coordinates, (list, tuple)) or len(coordinates) != 4:
                raise ValueError(f"Malformed box for {key}")
            values = [float(value) for value in coordinates]
            confidence = float(box.get("conf", box.get("confidence", box.get("score", 1.0))))
            if not all(math.isfinite(value) for value in [*values, confidence]) or values[2] <= 0 or values[3] <= 0:
                raise ValueError(f"Non-finite/degenerate box for {key}")
        result[str(key)] = boxes
    return result


def _boxes(row, detections):
    for key in (row.get("image_id"), row.get("source_file_name"), row.get("relative_path")):
        if str(key) in detections:
            return detections[str(key)]
    raise ValueError(f"Missing detection record, not a valid empty frame: {row.get('image_id')}")


def _animal_area(box, row):
    if str(box.get("category", "")).lower() in {"2", "3", "person", "vehicle"}:
        return None
    if float(box.get("conf", box.get("confidence", box.get("score", 1.0)))) < 0.2:
        return None
    x, y, w, h = map(float, box.get("bbox_xywh", box.get("bbox")))
    if max(abs(x), abs(y), abs(w), abs(h)) <= 1.5:
        return max(0.0, w) * max(0.0, h)
    width, height = row.get("width"), row.get("height")
    if not width or not height:
        return None  # Pixel box size cannot be classified without source dimensions.
    return max(0.0, w) * max(0.0, h) / (float(width) * float(height))


def select_overlay_cases(manifest, detections, illumination, *, limit=10):
    """Deterministic coverage selection; never changes the training sampler."""
    if not 1 <= int(limit) <= 20:
        raise ValueError("Overlay count must be between 1 and 20")
    detection_map = _detections(detections)
    lights = {str(row["image_id"]): row.get("illumination", "unknown") for row in _jsonl(illumination)}
    candidates = []
    seen = set()
    for row in _jsonl(manifest):
        if row.get("split") != "train":
            continue
        image_id = str(row.get("image_id", ""))
        if not image_id or image_id in seen or not isinstance(row.get("relative_path"), str):
            raise ValueError("Training manifest IDs/paths are missing or duplicated")
        seen.add(image_id)
        # Validate every training row's coverage; no missing-as-empty fallback.
        boxes = _boxes(row, detection_map)
        animal_boxes = [box for box in boxes if str(box.get("category", "")).lower()
                        not in {"2", "3", "person", "vehicle"}
                        and float(box.get("conf", box.get("confidence", box.get("score", 1.0)))) >= 0.2]
        areas = [area for box in animal_boxes if (area := _animal_area(box, row)) is not None and area > 0]
        kind = ("empty" if not animal_boxes else "size_unknown" if not areas
                else "small" if min(areas) < 0.01 else "ordinary")
        candidates.append({**row, "overlay_illumination": lights.get(image_id, "unknown"),
                           "overlay_kind": kind})
    candidates.sort(key=lambda row: row["image_id"])
    if not candidates:
        raise ValueError("No training records for real overlay evidence")
    chosen, used = [], set()
    buckets = [(light, kind) for light in ("day", "night") for kind in ("small", "ordinary", "empty")]
    for light, kind in buckets:
        match = next((row for row in candidates if row["overlay_illumination"] == light
                      and row["overlay_kind"] == kind and row["image_id"] not in used), None)
        if match and len(chosen) < limit:
            chosen.append(match)
            used.add(match["image_id"])
    for row in candidates:
        if len(chosen) >= limit:
            break
        if row["image_id"] not in used:
            chosen.append(row)
            used.add(row["image_id"])
    return chosen


@contextmanager
def preserved_rng(seed):
    import numpy as np
    import torch
    state = random.getstate(), np.random.get_state(), torch.random.get_rng_state()
    try:
        random.seed(seed)
        np.random.seed(seed % (2**32))
        # Seed the CPU generator only. A live Colab CUDA generator must not be
        # changed by a cheap preview in the current notebook kernel.
        torch.random.default_generator.manual_seed(seed)
        yield
    finally:
        random.setstate(state[0])
        np.random.set_state(state[1])
        torch.random.set_rng_state(state[2])


def _freeze_bytes(path, payload):
    path = Path(path)
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"Frozen overlay evidence changed; preserve and review: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".partial-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
        os.replace(temporary, path)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()


def generate_overlays(*, manifest, detections, tags, illumination, image_root, output_dir,
                      source_identity, seed=20260916, limit=10):
    """Run actual dataset __getitem__; show masks even when the random crop misses.

    ROI weights operate on nearest-neighbour latent masks, not SAM pixel labels.
    A source animal outside a uniform crop is truthful evidence, not repaired by
    retrying seeds until it appears. All RNG state is restored after the preview.
    """
    if len(str(source_identity)) != 64 or any(c not in "0123456789abcdef" for c in str(source_identity)):
        raise ValueError("Overlay source identity must be a SHA256 digest")
    import io
    import numpy as np
    from PIL import Image, ImageDraw
    import torch
    from torch.nn import functional as F
    from dataset.camera_trap_dataset import CameraTrapDataset

    selected = select_overlay_cases(manifest, detections, illumination, limit=limit)
    output_dir = Path(output_dir)
    for row in selected:
        path = contained_path(image_root, row["relative_path"])
        if not path.is_file():
            raise FileNotFoundError(f"Stage this selected original before overlays: {path}")

    class PreviewDataset(CameraTrapDataset):
        def _joint_transform(self, image, boxes):
            self.processed_image = image.copy()
            self.processed_boxes = list(boxes)
            return super()._joint_transform(image, self.processed_boxes)

    dataset = PreviewDataset(str(manifest), str(image_root), "train", detections_path=str(detections),
                             tags_path=str(tags), illumination_sidecar=str(illumination),
                             **{key: value for key, value in TRANSFORMS.items() if key != "latent_downscale"})
    indices = {str(row["image_id"]): index for index, row in enumerate(dataset.rows)}
    records, panels = [], []
    source_digests = {name: sha256(path) for name, path in
                      (("manifest", manifest), ("detections", detections), ("tags", tags), ("illumination", illumination))}
    for number, row in enumerate(selected):
        case_seed = int(seed) + number
        with preserved_rng(case_seed):
            sample = dataset[indices[row["image_id"]]]
        mask = sample["roi_mask"][..., 0]
        if sample["roi_known"] != 1 or not np.isfinite(mask).all() or mask.shape != (256, 256):
            raise RuntimeError(f"ROI evidence is missing/malformed for {row['image_id']}")
        source = (np.clip(sample["hint"], 0, 1) * 255).round().astype(np.uint8)
        colored = source.astype(np.float32)
        colored[mask > 0] = colored[mask > 0] * 0.55 + np.array([255, 40, 40]) * 0.45
        original = dataset.processed_image.copy()
        draw = ImageDraw.Draw(original)
        for x, y, w, h in dataset.processed_boxes:
            draw.rectangle((x, y, x + w, y + h), outline="red", width=2)
        original.thumbnail((256, 256))
        latent = F.interpolate(torch.from_numpy(mask.copy())[None, None], size=(32, 32), mode="nearest")[0, 0].numpy()
        panel = Image.new("RGB", (768, 298), "white")
        panel.paste(original, (0, 28))
        panel.paste(Image.fromarray(colored.clip(0, 255).astype(np.uint8)), (256, 28))
        latent_image = Image.fromarray((latent * 255).astype(np.uint8)).resize((256, 256), Image.Resampling.NEAREST)
        panel.paste(latent_image.convert("RGB"), (512, 28))
        labels = ImageDraw.Draw(panel)
        labels.text((4, 6), "Processed frame + animal boxes", fill="black")
        labels.text((260, 6), "Actual random crop + ROI mask", fill="black")
        labels.text((516, 6), "Actual nearest latent mask (32x32)", fill="black")
        labels.text((4, 285), f"{number + 1}: {row['overlay_illumination']}/{row['overlay_kind']} | seed={case_seed} | ROI={mask.mean():.4f}", fill="black")
        buffer = io.BytesIO()
        panel.save(buffer, format="PNG")
        filename = f"case_{number + 1:02d}.png"
        _freeze_bytes(output_dir / filename, buffer.getvalue())
        panels.append(panel)
        records.append({"image_id": row["image_id"], "relative_path": row["relative_path"],
                        "source_image_sha256": sha256(contained_path(image_root, row["relative_path"])),
                        "illumination": row["overlay_illumination"], "kind": row["overlay_kind"],
                        "seed": case_seed, "roi_known": True, "roi_fraction": float(mask.mean()),
                        "latent_roi_fraction": float(latent.mean()), "crop_contains_animal": bool(mask.any()),
                        "processed_size": list(dataset.processed_image.size), "crop_size": [256, 256],
                        "panel": filename, "panel_sha256": sha256(output_dir / filename)})
    sheet = Image.new("RGB", (768, 298 * len(panels)), "white")
    for number, panel in enumerate(panels):
        sheet.paste(panel, (0, number * 298))
    buffer = io.BytesIO()
    sheet.save(buffer, format="PNG")
    _freeze_bytes(output_dir / "contact_sheet.png", buffer.getvalue())
    present = {(row["illumination"], row["kind"]) for row in records}
    report = {"schema_version": 1, "source_identity": source_identity, "source_sha256": source_digests,
              "seed": int(seed), "transforms": TRANSFORMS, "records": records,
              "contact_sheet": "contact_sheet.png", "contact_sheet_sha256": sha256(output_dir / "contact_sheet.png"),
              "unavailable_cases": [f"{light}/{kind}" for light in ("day", "night")
                                    for kind in ("small", "ordinary", "empty") if (light, kind) not in present],
              "training_sampler_changed": False,
              "notes": ["MegaDetector boxes are pseudo-reference, not human segmentation.",
                        "Uniform random crops can miss source animals; preview never retries a seed.",
                        "Small boxes can disappear at latent resolution; inspect the right panel."]}
    _freeze_bytes(output_dir / "report.json", (json.dumps(report, sort_keys=True, indent=2) + "\n").encode("utf-8"))
    print("Real ROI overlays:", output_dir / "contact_sheet.png", flush=True)
    print("Overlay report SHA256:", sha256(output_dir / "report.json"), flush=True)
    return report


def validate_overlay_report(report_path, *, source_identity, image_root):
    """Validate acknowledged, frozen evidence without regenerating a preview."""
    report_path = Path(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (report.get("schema_version") != 1 or report.get("source_identity") != source_identity
            or report.get("transforms") != TRANSFORMS or report.get("training_sampler_changed") is not False):
        raise RuntimeError("Overlay provenance/transform contract mismatch; no GPU job allowed")
    records = report.get("records")
    if not isinstance(records, list) or not 1 <= len(records) <= 20:
        raise RuntimeError("Real transformed overlay records are missing")
    ids = [row.get("image_id") for row in records if isinstance(row, dict)]
    if len(ids) != len(records) or not all(isinstance(item, str) and item for item in ids) or len(set(ids)) != len(ids):
        raise RuntimeError("Overlay image IDs are missing/duplicated")
    source_digests = report.get("source_sha256", {})
    if set(source_digests) != {"manifest", "detections", "tags", "illumination"}:
        raise RuntimeError("Overlay source digests are incomplete")
    if not all(isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
               for value in source_digests.values()):
        raise RuntimeError("Malformed overlay source digest")
    for record in records:
        for field in ("roi_fraction", "latent_roi_fraction"):
            value = record.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise RuntimeError("Malformed ROI coverage evidence")
        if record.get("roi_known") is not True or record.get("crop_size") != [256, 256]:
            raise RuntimeError("Unknown or mis-sized real ROI crop")
        image = contained_path(image_root, record.get("relative_path", ""))
        panel = contained_path(report_path.parent, record.get("panel", ""))
        if not image.is_file() or sha256(image) != record.get("source_image_sha256"):
            raise RuntimeError(f"Overlay source image changed/missing: {image}")
        if not panel.is_file() or sha256(panel) != record.get("panel_sha256"):
            raise RuntimeError(f"Frozen overlay panel changed/missing: {panel}")
    sheet = contained_path(report_path.parent, report.get("contact_sheet", ""))
    if not sheet.is_file() or sha256(sheet) != report.get("contact_sheet_sha256"):
        raise RuntimeError("Frozen overlay contact sheet changed/missing")
    return report
