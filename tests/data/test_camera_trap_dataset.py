from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

pytest.importorskip("torch")
from dataset.camera_trap_dataset import (
    CameraTrapDataset,
    build_domain_prompt,
    encode_domain_metadata,
    prompt_from_domain_metadata,
)


def _row(**overrides):
    row = {
        "image_id": "KGA:img1",
        "relative_path": "snapshot_kgalagadi/a.jpg",
        "source_file_name": "a.jpg",
        "site_id": "KGA:A01",
        "sequence_id": "KGA:SEQ1",
        "split": "train",
        "datetime": "2018-12-24 22:10:22",
        "illumination": "night",
        "species": ["gemsbokoryx"],
        "boxes": None,
    }
    row.update(overrides)
    return row


def _write_fixture(tmp_path: Path, rows):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    image_path = tmp_path / "images" / "snapshot_kgalagadi" / "a.jpg"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (100, 80), color=(100, 120, 140)).save(image_path)
    return manifest, tmp_path / "images"


def test_domain_prompt_uses_metadata_but_not_ground_truth_species():
    prompt = build_domain_prompt(_row(), include_site=True)
    assert "infrared night image" in prompt
    assert "austral summer" in prompt
    assert "camera site A01" in prompt
    assert "gemsbok" not in prompt


def test_detection_mask_and_image_transform_stay_aligned(tmp_path):
    manifest, root = _write_fixture(tmp_path, [_row()])
    detections = tmp_path / "detections.jsonl"
    detections.write_text(
        json.dumps(
            {
                "image_id": "KGA:img1",
                "detections": [{"category": "1", "conf": 0.9, "bbox": [0.4, 0.25, 0.2, 0.25]}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    dataset = CameraTrapDataset(
        str(manifest),
        str(root),
        split="train",
        out_size=64,
        crop_type="random",
        detections_path=str(detections),
        bbox_crop_probability=1.0,
        use_hflip=False,
    )
    random.seed(7)
    sample = dataset[0]
    assert sample["jpg"].shape == (64, 64, 3)
    assert sample["hint"].shape == (64, 64, 3)
    assert sample["roi_mask"].shape == (64, 64, 1)
    assert float(sample["roi_mask"].sum()) > 0
    assert np.allclose((sample["jpg"] + 1.0) / 2.0, sample["hint"], atol=1e-6)


def test_processing_resize_keeps_image_and_mask_aligned(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(_row(illumination="day")) + "\n", encoding="utf-8")
    pixels = np.zeros((2000, 2592, 3), dtype=np.uint8)
    pixels[800:1000, 1200:1400] = 255  # the "animal"
    image_path = tmp_path / "images" / "snapshot_kgalagadi" / "a.jpg"
    image_path.parent.mkdir(parents=True)
    Image.fromarray(pixels).save(image_path, quality=100)
    detections = tmp_path / "detections.jsonl"
    detections.write_text(json.dumps({
        "image_id": "KGA:img1",
        "detections": [{"category": "1", "conf": 0.9,
                        "bbox": [1200 / 2592, 800 / 2000, 200 / 2592, 200 / 2000]}],
    }) + "\n", encoding="utf-8")
    dataset = CameraTrapDataset(
        str(manifest), str(tmp_path / "images"), split="train", out_size=256,
        crop_type="random", detections_path=str(detections),
        bbox_crop_probability=1.0, use_hflip=False, processing_long_side=1024,
    )
    random.seed(3)
    sample = dataset[0]
    mask = sample["roi_mask"][..., 0] > 0.5
    # 200 px at 2592 wide becomes ~79 px at 1024 wide.
    assert 0 < mask.sum() <= 80 * 80
    assert sample["hint"][mask].mean() > 0.95
    assert sample["hint"][~mask].mean() < 0.05


def test_site_and_split_filtering(tmp_path):
    rows = [
        _row(),
        _row(image_id="KGA:img2", site_id="KGA:A02", sequence_id="KGA:SEQ2"),
        _row(image_id="KGA:img3", split="val", sequence_id="KGA:SEQ3"),
    ]
    manifest, root = _write_fixture(tmp_path, rows)
    dataset = CameraTrapDataset(
        str(manifest),
        str(root),
        split="train",
        site_id="KGA:A01",
        crop_type="none",
        use_hflip=False,
        domain_conditioning=True,
    )
    assert len(dataset) == 1
    sample = dataset[0]
    assert sample["site_id"] == "KGA:A01"
    assert sample["txt"]
    assert sample["domain_metadata_bits"] == 8


def test_domain_metadata_round_trip_prompt():
    row = _row(illumination="night", datetime="2020-07-02 19:30:00")
    code = encode_domain_metadata(row)
    prompt = prompt_from_domain_metadata(
        code, site_id=row["site_id"], habitat="arid savanna"
    )
    assert code == 5
    assert "infrared night image" in prompt
    assert "austral winter" in prompt
    assert "camera site A01" in prompt
    assert "habitat arid savanna" in prompt
