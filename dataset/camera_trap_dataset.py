"""Manifest-backed camera-trap dataset for H1/H2/H3 experiments.

The image, ROI and metadata transformations live in one class so a crop or
flip can never silently desynchronise the animal mask from the image.
"""
from __future__ import annotations

import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import torch.utils.data as data


def _load_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
    return rows


def _austral_season(datetime_text: Optional[str]) -> str:
    """Return the Southern-Hemisphere season for a COCO-CT timestamp."""
    if not datetime_text:
        return "unknown season"
    try:
        month = datetime.strptime(datetime_text.replace("T", " "), "%Y-%m-%d %H:%M:%S").month
    except ValueError:
        return "unknown season"
    if month in (12, 1, 2):
        return "austral summer"
    if month in (3, 4, 5):
        return "austral autumn"
    if month in (6, 7, 8):
        return "austral winter"
    return "austral spring"


_SEASONS = ("austral summer", "austral autumn", "austral winter", "austral spring")


def encode_domain_metadata(row: Mapping) -> int:
    """Pack illumination (1 bit) and austral season (2 bits) into one byte."""
    illumination_bit = 1 if row.get("illumination") == "night" else 0
    season = _austral_season(row.get("datetime"))
    season_index = _SEASONS.index(season) if season in _SEASONS else 0
    return illumination_bit | (season_index << 1)


def prompt_from_domain_metadata(
    code: int,
    *,
    site_id: Optional[str] = None,
    habitat: Optional[str] = None,
) -> str:
    illumination = "infrared night image" if code & 1 else "daylight RGB image"
    season = _SEASONS[(int(code) >> 1) & 0b11]
    parts = ["camera trap wildlife photograph", illumination, season]
    if site_id:
        parts.append(f"camera site {str(site_id).split(':')[-1]}")
    if habitat:
        parts.append(f"habitat {habitat}")
    return ", ".join(parts)


def build_domain_prompt(
    row: Mapping,
    *,
    include_site: bool = False,
    include_illumination: bool = True,
    include_season: bool = True,
    habitat_by_site: Optional[Mapping[str, str]] = None,
) -> str:
    """Build H3 text without using ground-truth species as an oracle."""
    parts = ["camera trap wildlife photograph"]
    illumination = row.get("illumination")
    if include_illumination and illumination:
        parts.append("infrared night image" if illumination == "night" else "daylight RGB image")
    if include_season:
        parts.append(_austral_season(row.get("datetime")))
    site_id = str(row.get("site_id") or "")
    if include_site and site_id:
        parts.append(f"camera site {site_id.split(':')[-1]}")
    habitat = (habitat_by_site or {}).get(site_id)
    if habitat:
        parts.append(f"habitat {habitat}")
    return ", ".join(parts)


def _load_habitat_map(path: Optional[str]) -> Dict[str, str]:
    if not path:
        return {}
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in value.items()):
        raise ValueError("habitat_map must be a JSON object of site_id -> habitat text")
    return value


def _load_detection_map(path: Optional[str]) -> Dict[str, List[dict]]:
    """Load a compact JSONL sidecar or MegaDetector-style JSON file."""
    if not path:
        return {}
    path_obj = Path(path)
    if path_obj.suffix.lower() == ".jsonl":
        records = _load_jsonl(path_obj)
    else:
        payload = json.loads(path_obj.read_text(encoding="utf-8"))
        records = payload.get("images", payload) if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError(f"detection file must contain a list, got {type(records).__name__}")

    result: Dict[str, List[dict]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        image_id = record.get("image_id") or record.get("id") or record.get("file")
        if image_id is None:
            continue
        boxes = record.get("boxes")
        if boxes is None:
            boxes = record.get("detections", [])
        result[str(image_id)] = boxes if isinstance(boxes, list) else []
    return result


def _load_tag_map(path: Optional[str]) -> Dict[str, dict]:
    """Load RAM++ tag strings cached before GPU-heavy codec training."""
    if not path:
        return {}
    records = _load_jsonl(Path(path))
    result = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        image_id = record.get("image_id")
        tags = record.get("tags")
        if image_id is None or not isinstance(tags, str):
            continue
        result[str(image_id)] = record
    return result


def _box_xywh(box: Mapping, width: int, height: int, min_confidence: float) -> Optional[Tuple[float, float, float, float]]:
    category = str(box.get("category", "")).strip().lower()
    # MegaDetector: 1=animal, 2=person, 3=vehicle.
    if category in {"2", "3", "person", "vehicle"}:
        return None
    confidence = float(box.get("conf", box.get("confidence", box.get("score", 1.0))))
    if confidence < min_confidence:
        return None
    value = box.get("bbox_xywh", box.get("bbox"))
    if not isinstance(value, Sequence) or len(value) != 4:
        return None
    x, y, w, h = (float(v) for v in value)
    # MegaDetector emits normalised xywh; the LILA bbox manifests use pixels.
    if max(abs(x), abs(y), abs(w), abs(h)) <= 1.5:
        x, w = x * width, w * width
        y, h = y * height, h * height
    x0, y0 = max(0.0, x), max(0.0, y)
    x1, y1 = min(float(width), x + max(0.0, w)), min(float(height), y + max(0.0, h))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1 - x0, y1 - y0


class CameraTrapDataset(data.Dataset):
    """Read one split/site from the frozen camera-trap manifest."""

    def __init__(
        self,
        manifest_path: str,
        data_root: str,
        split: str,
        out_size: int = 256,
        crop_type: str = "random",
        site_id: Optional[str] = None,
        detections_path: Optional[str] = None,
        tags_path: Optional[str] = None,
        min_detection_confidence: float = 0.2,
        bbox_crop_probability: float = 0.5,
        use_hflip: bool = True,
        domain_conditioning: bool = False,
        include_site_in_prompt: bool = False,
        include_illumination_in_prompt: bool = True,
        include_season_in_prompt: bool = True,
        habitat_map: Optional[str] = None,
        domain_metadata_bits: int = 8,
    ) -> None:
        super().__init__()
        if split not in {"train", "val", "test"}:
            raise ValueError(f"invalid split {split!r}")
        if crop_type not in {"none", "center", "random"}:
            raise ValueError(f"invalid crop_type {crop_type!r}")
        if not 0.0 <= bbox_crop_probability <= 1.0:
            raise ValueError("bbox_crop_probability must be in [0, 1]")

        self.manifest_path = Path(manifest_path)
        self.data_root = Path(data_root)
        self.split = split
        self.out_size = int(out_size)
        self.crop_type = crop_type
        self.site_id = site_id
        self.min_detection_confidence = float(min_detection_confidence)
        self.bbox_crop_probability = float(bbox_crop_probability)
        self.use_hflip = bool(use_hflip)
        self.domain_conditioning = bool(domain_conditioning)
        self.include_site_in_prompt = bool(include_site_in_prompt)
        self.include_illumination_in_prompt = bool(include_illumination_in_prompt)
        self.include_season_in_prompt = bool(include_season_in_prompt)
        self.domain_metadata_bits = int(domain_metadata_bits) if domain_conditioning else 0
        self.habitat_by_site = _load_habitat_map(habitat_map)
        self.detections = _load_detection_map(detections_path)
        self.tags = _load_tag_map(tags_path)

        rows = _load_jsonl(self.manifest_path)
        self.rows = [
            row
            for row in rows
            if row.get("split") == split and (site_id is None or row.get("site_id") == site_id)
        ]
        if not self.rows:
            suffix = f" and site_id={site_id!r}" if site_id else ""
            raise ValueError(f"no rows for split={split!r}{suffix} in {manifest_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def _boxes_for(self, row: Mapping, width: int, height: int) -> List[Tuple[float, float, float, float]]:
        raw_boxes = row.get("boxes")
        if not isinstance(raw_boxes, list):
            keys = [str(row.get("image_id")), str(row.get("source_file_name")), str(row.get("relative_path"))]
            raw_boxes = next((self.detections[key] for key in keys if key in self.detections), [])
        boxes = []
        for raw_box in raw_boxes:
            if isinstance(raw_box, Mapping):
                box = _box_xywh(raw_box, width, height, self.min_detection_confidence)
                if box is not None:
                    boxes.append(box)
        return boxes

    def _open_image(self, path: Path) -> Image.Image:
        last_error = None
        for attempt in range(3):
            try:
                with Image.open(path) as image:
                    return image.convert("RGB")
            except (OSError, ValueError) as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(0.25 * (attempt + 1))
        raise OSError(f"failed to load image {path}: {last_error}")

    def _joint_transform(
        self,
        image: Image.Image,
        boxes: Iterable[Tuple[float, float, float, float]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        width, height = image.size
        mask = Image.new("L", (width, height), color=0)
        mask_array = np.zeros((height, width), dtype=np.uint8)
        for x, y, w, h in boxes:
            x0, y0 = int(np.floor(x)), int(np.floor(y))
            x1, y1 = int(np.ceil(x + w)), int(np.ceil(y + h))
            mask_array[max(0, y0):min(height, y1), max(0, x0):min(width, x1)] = 255
        mask = Image.fromarray(mask_array, mode="L")

        if self.crop_type != "none":
            scale = max(self.out_size / width, self.out_size / height, 1.0)
            if scale > 1.0:
                new_size = (int(round(width * scale)), int(round(height * scale)))
                image = image.resize(new_size, Image.Resampling.BICUBIC)
                mask = mask.resize(new_size, Image.Resampling.NEAREST)
                width, height = new_size

            if self.crop_type == "center":
                left = max(0, (width - self.out_size) // 2)
                top = max(0, (height - self.out_size) // 2)
            else:
                mask_np = np.asarray(mask)
                ys, xs = np.where(mask_np > 0)
                use_bbox_crop = len(xs) > 0 and random.random() < self.bbox_crop_probability
                if use_bbox_crop:
                    chosen = random.randrange(len(xs))
                    center_x, center_y = int(xs[chosen]), int(ys[chosen])
                    min_left = max(0, center_x - self.out_size + 1)
                    max_left = min(center_x, width - self.out_size)
                    min_top = max(0, center_y - self.out_size + 1)
                    max_top = min(center_y, height - self.out_size)
                    left = random.randint(min_left, max_left) if max_left >= min_left else max(0, max_left)
                    top = random.randint(min_top, max_top) if max_top >= min_top else max(0, max_top)
                else:
                    left = random.randint(0, width - self.out_size)
                    top = random.randint(0, height - self.out_size)
            box = (left, top, left + self.out_size, top + self.out_size)
            image, mask = image.crop(box), mask.crop(box)

        if self.use_hflip and random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        image_array = np.asarray(image, dtype=np.float32) / 255.0
        mask_array = (np.asarray(mask, dtype=np.float32) / 255.0)[..., None]
        return image_array, mask_array

    def __getitem__(self, index: int) -> Dict[str, object]:
        row = self.rows[index]
        image_path = self.data_root / Path(row["relative_path"])
        image = self._open_image(image_path)
        boxes = self._boxes_for(row, *image.size)
        source, roi_mask = self._joint_transform(image, boxes)
        target = (source * 2.0 - 1.0).astype(np.float32)
        tag_record = self.tags.get(str(row["image_id"]), {})
        prompt = str(tag_record.get("tags") or "")
        tag_ids = tag_record.get("tag_ids")
        tag_payload_bits = 0
        if isinstance(tag_ids, list):
            # Three uint32 headers plus byte-padded 13-bit RAM++ ids, matching
            # apply_condition_compress for the authors' full vocabulary.
            tag_payload_bits = 96 + ((13 * len(tag_ids) + 7) // 8) * 8
        if self.domain_conditioning:
            domain_prompt = build_domain_prompt(
                row,
                include_site=self.include_site_in_prompt,
                include_illumination=self.include_illumination_in_prompt,
                include_season=self.include_season_in_prompt,
                habitat_by_site=self.habitat_by_site,
            )
            prompt = ", ".join(part for part in (prompt, domain_prompt) if part)
        return {
            "jpg": target,
            "hint": source.astype(np.float32),
            "txt": prompt,
            "roi_mask": roi_mask.astype(np.float32),
            "domain_metadata_bits": np.float32(self.domain_metadata_bits),
            "tag_payload_bits": np.float32(tag_payload_bits),
            "image_id": str(row["image_id"]),
            "site_id": str(row["site_id"]),
            "sequence_id": str(row["sequence_id"]),
            "illumination": str(row.get("illumination") or "unknown"),
        }
