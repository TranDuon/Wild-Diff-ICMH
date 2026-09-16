"""Shared fixtures for tests/data. No network by default; CPU only."""
from __future__ import annotations

import json
import os
import sys
import zipfile
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TOOLS_DATA = _REPO_ROOT / "tools" / "data"
for _p in (str(_REPO_ROOT), str(_TOOLS_DATA)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "network: requires network access; opt-in via WILD_DATA_NET_TESTS=1"
    )


def pytest_collection_modifyitems(config, items):
    if os.environ.get("WILD_DATA_NET_TESTS") == "1":
        return
    skip_network = pytest.mark.skip(reason="network tests are opt-in; set WILD_DATA_NET_TESTS=1")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip_network)


@pytest.fixture
def make_coco_ct(tmp_path):
    """Factory building an in-memory Serengeti-style COCO-CT dict, written to disk."""

    def _make(
        n_sites: int = 12,
        seqs_per_site: int = 4,
        frames_per_seq: int = 3,
        night_fraction: float = 0.3,
        add_null_datetime_image: bool = False,
        add_zero_box_image: bool = False,
        add_person_box_image: bool = False,
        add_vehicle_box_image: bool = False,
        out_name: str = "serengeti_bboxes",
        as_zip: bool = True,
        season: str = "S1",
    ):
        images = []
        annotations = []
        ann_id = 1
        img_counter = 1
        seq_index = 0
        total_seqs = n_sites * seqs_per_site
        night_seqs = int(round(total_seqs * night_fraction))

        for site_index in range(n_sites):
            site = f"D{site_index:02d}"
            for seq_num in range(seqs_per_site):
                seq_id = f"SEQ{site_index:03d}{seq_num:02d}"
                is_night = seq_index < night_seqs
                hour = 22 if is_night else 12
                for frame in range(1, frames_per_seq + 1):
                    stem = f"{season}_{site}_R1_PICT{img_counter:04d}"
                    file_name = f"{season}/{site}/{site}_R1/{stem}.JPG"
                    image_id = f"{season}/{site}/{site}_R1/{stem}"
                    datetime_value = f"2010-10-08 {hour:02d}:00:00"
                    if add_null_datetime_image and img_counter == 1:
                        datetime_value = None
                    images.append(
                        {
                            "id": image_id,
                            "file_name": file_name,
                            "seq_id": seq_id,
                            "location": site,
                            "height": 1536,
                            "width": 2048,
                            "seq_num_frames": frames_per_seq,
                            "frame_num": frame,
                            "season": season,
                            "datetime": datetime_value,
                        }
                    )
                    skip_box = add_zero_box_image and img_counter == 2
                    if not skip_box:
                        category_id = 1
                        if add_person_box_image and img_counter == 3:
                            category_id = 2
                        elif add_vehicle_box_image and img_counter == 4:
                            category_id = 4
                        annotations.append(
                            {
                                "id": f"ann{ann_id}",
                                "category_id": category_id,
                                "image_id": image_id,
                                "bbox": [10.0, 10.0, 50.0, 50.0],
                            }
                        )
                        ann_id += 1
                    img_counter += 1
                seq_index += 1

        data = {
            "info": {"version": "test-1", "description": "synthetic test fixture"},
            "categories": [
                {"id": 1, "name": "animal"},
                {"id": 2, "name": "person"},
                {"id": 3, "name": "group"},
                {"id": 4, "name": "vehicle"},
            ],
            "images": images,
            "annotations": annotations,
        }

        member_name = out_name if out_name.endswith(".json") else out_name + ".json"
        if as_zip:
            out_path = tmp_path / (out_name.replace(".json", "") + ".zip")
            with zipfile.ZipFile(out_path, "w") as zf:
                zf.writestr(member_name, json.dumps(data))
        else:
            out_path = tmp_path / member_name
            out_path.write_text(json.dumps(data), encoding="utf-8")

        return out_path, data

    return _make


@pytest.fixture
def jpeg_mirror(tmp_path):
    """Factory writing small real Pillow JPEGs at mirror_dir/source_file_name."""
    from PIL import Image

    def _make(rows, mirror_dir=None, exif_datetime=None, grayscale=False):
        mirror_dir = Path(mirror_dir) if mirror_dir is not None else tmp_path / "mirror"
        mirror_dir.mkdir(parents=True, exist_ok=True)
        for row in rows:
            dest = mirror_dir / row["source_file_name"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            if grayscale:
                img = Image.new("RGB", (16, 16), color=(128, 128, 128))
            else:
                img = Image.new("RGB", (16, 16), color=(200, 50, 50))
            save_kwargs = {}
            if exif_datetime:
                exif = img.getexif()
                exif[36867] = exif_datetime
                save_kwargs["exif"] = exif.tobytes()
            img.save(dest, format="JPEG", **save_kwargs)
        return mirror_dir

    return _make
