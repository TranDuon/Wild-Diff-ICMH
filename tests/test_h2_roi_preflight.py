"""Cheap local evidence tests; production transforms, no model/GPU/Drive job."""
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image
import pytest
import torch

from tools import h2_roi_preflight as surface


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


@pytest.fixture
def inputs(tmp_path):
    images = tmp_path / "local_images"
    images.mkdir()
    rows, boxes, tags, illumination = [], [], [], []
    for index in range(6):
        identifier = f"KGA:{index}"
        relative = f"snapshot_kgalagadi/animal_{index}.png"
        path = images / relative
        path.parent.mkdir(exist_ok=True)
        array = np.zeros((400, 512, 3), dtype=np.uint8)
        array[..., 1] = 70
        array[100:160, 180:240] = (255, 20, 20)
        Image.fromarray(array).save(path)
        kind = index % 3
        detections = [] if kind == 2 else [{"category": "1", "conf": 0.9,
                    "bbox": [0.35, 0.25, 0.025 if kind == 0 else 0.15, 0.025 if kind == 0 else 0.15]}]
        rows.append({"image_id": identifier, "relative_path": relative, "split": "train",
                     "width": 512, "height": 400, "site_id": "KGA:A01", "sequence_id": str(index)})
        boxes.append({"image_id": identifier, "detections": detections})
        tags.append({"image_id": identifier, "tags": "wildlife", "tag_ids": [1]})
        illumination.append({"image_id": identifier, "illumination": "day" if index < 3 else "night"})
    return {"manifest": write_jsonl(tmp_path / "manifest.jsonl", rows),
            "detections": write_jsonl(tmp_path / "detections.jsonl", boxes),
            "tags": write_jsonl(tmp_path / "tags.jsonl", tags),
            "illumination": write_jsonl(tmp_path / "illumination.jsonl", illumination),
            "image_root": images, "output_dir": tmp_path / "overlays", "source_identity": "a" * 64}


@pytest.mark.parametrize("relative", ["../outside", "/etc/image", "C:/private/image", "a/../../outside", "a\\..\\image", "", "./image"])
def test_staging_rejects_escaping_paths_before_copy(tmp_path, relative):
    with pytest.raises(ValueError):
        surface.stage_assets(tmp_path / "drive", tmp_path / "local", [relative])
    assert not (tmp_path / "local").exists()


def test_scoped_staging_reuses_valid_and_atomically_replaces_partial(tmp_path):
    root, local = tmp_path / "drive", tmp_path / "local"
    for name, content in (("selected/a.png", b"first"), ("selected/b.png", b"second"), ("not_selected.png", b"unused")):
        path = root / "images" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    target = local / "selected/a.png"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"first")
    before = target.stat().st_mtime_ns
    (local / "selected/b.png").write_bytes(b"bad")
    stats = surface.stage_assets(root, local, ["selected/a.png", "selected/b.png", "selected/a.png"])
    assert stats["requested_images"] == 2 and stats["reused_files"] == 1 and stats["copied_files"] == 1
    assert target.stat().st_mtime_ns == before
    assert (local / "selected/b.png").read_bytes() == b"second"
    assert not (local / "not_selected.png").exists()
    assert not list(local.rglob("*.partial-*"))


def test_selection_spans_real_cases_and_missing_detection_is_not_empty(inputs):
    rows = surface.select_overlay_cases(inputs["manifest"], inputs["detections"], inputs["illumination"], limit=6)
    assert {(row["overlay_illumination"], row["overlay_kind"]) for row in rows} == {
        (light, kind) for light in ("day", "night") for kind in ("small", "ordinary", "empty")}
    detections = [json.loads(line) for line in inputs["detections"].read_text().splitlines()]
    write_jsonl(inputs["detections"], detections[:-1])
    with pytest.raises(ValueError, match="Missing detection record"):
        surface.select_overlay_cases(inputs["manifest"], inputs["detections"], inputs["illumination"])


def test_real_dataset_overlay_preserves_rng_and_is_frozen(inputs, monkeypatch):
    from dataset.camera_trap_dataset import CameraTrapDataset
    actual_getitem = CameraTrapDataset.__getitem__
    calls = []
    def watched(self, index):
        calls.append(self.rows[index]["image_id"])
        return actual_getitem(self, index)
    monkeypatch.setattr(CameraTrapDataset, "__getitem__", watched)
    random.seed(55)
    np.random.seed(56)
    torch.random.default_generator.manual_seed(57)
    py_state, np_state, torch_state = random.getstate(), np.random.get_state(), torch.random.get_rng_state()
    report = surface.generate_overlays(**inputs, limit=6)
    assert len(calls) == 6 and set(calls) == {f"KGA:{index}" for index in range(6)}
    assert random.getstate() == py_state
    assert np.array_equal(np.random.get_state()[1], np_state[1])
    assert torch.equal(torch.random.get_rng_state(), torch_state)
    assert report["training_sampler_changed"] is False
    assert report["unavailable_cases"] == []
    assert all(case["crop_size"] == [256, 256] and case["roi_known"] for case in report["records"])
    for case in report["records"]:
        if case["kind"] == "empty":
            assert case["roi_fraction"] == 0 and case["latent_roi_fraction"] == 0
    checked = surface.validate_overlay_report(inputs["output_dir"] / "report.json",
                                              source_identity="a" * 64, image_root=inputs["image_root"])
    assert checked == report
    assert surface.generate_overlays(**inputs, limit=6) == report
    (inputs["output_dir"] / "case_01.png").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="panel changed"):
        surface.validate_overlay_report(inputs["output_dir"] / "report.json",
                                        source_identity="a" * 64, image_root=inputs["image_root"])


def test_overlay_validator_rejects_foreign_identity_and_source_change(inputs):
    report = surface.generate_overlays(**inputs, limit=2)
    path = inputs["output_dir"] / "report.json"
    with pytest.raises(RuntimeError, match="provenance"):
        surface.validate_overlay_report(path, source_identity="b" * 64, image_root=inputs["image_root"])
    source = inputs["image_root"] / report["records"][0]["relative_path"]
    source.write_bytes(b"new source")
    with pytest.raises(RuntimeError, match="source image changed"):
        surface.validate_overlay_report(path, source_identity="a" * 64, image_root=inputs["image_root"])
