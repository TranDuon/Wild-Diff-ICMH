from __future__ import annotations

import json

import pytest

from tools.data.audit_metadata import build_metadata_audit, main


def _row(index: int) -> dict:
    return {
        "image_id": f"image-{index}",
        "source": "snapshot_kgalagadi",
        "source_version": "test",
        "license": "test",
        "relative_path": f"site/image-{index}.jpg",
        "source_file_name": f"site/image-{index}.jpg",
        "site_id": "KGA:A01",
        "sequence_id": f"sequence-{index}",
        "frame_num": 1,
        "datetime": "2026-01-01 12:00:00",
        "illumination": "day",
        "illumination_source": "datetime_hour_proxy",
        "species": [],
        "is_empty": True,
        "boxes": None,
        "width": 32,
        "height": 32,
        "split": "train",
        "subset": "empty",
        "sample_rank": index,
        "sha256": None,
        "null_reasons": {"boxes": "none", "sha256": "external"},
    }


def _write_fixture(tmp_path, count=100):
    manifest = tmp_path / "manifest.jsonl"
    checksums = tmp_path / "checksums.jsonl"
    with manifest.open("w", encoding="utf-8") as manifest_stream, checksums.open(
        "w", encoding="utf-8"
    ) as checksum_stream:
        for index in range(count):
            manifest_stream.write(json.dumps(_row(index)) + "\n")
            checksum_stream.write(
                json.dumps(
                    {
                        "image_id": f"image-{index}",
                        "exif_datetime": "2026:01:01 12:00:00" if index < 25 else None,
                        "is_grayscale": False,
                    }
                )
                + "\n"
            )
    return manifest, checksums


def test_audit_uses_exif_then_manifest_fallback(tmp_path):
    manifest, checksums = _write_fixture(tmp_path)
    report = build_metadata_audit(
        manifest, site_id="KGA:A01", sample_size=100, checksums_path=checksums
    )
    assert report["datetime_source_counts"] == {"exif": 25, "manifest": 75}
    assert report["datetime_usable_ratio"] == 1.0
    assert report["location_usable_ratio"] == 1.0
    assert len(report["image_ids"]) == 100


def test_audit_fails_closed_when_less_than_requested_are_available(tmp_path):
    manifest, checksums = _write_fixture(tmp_path, count=2)
    with pytest.raises(RuntimeError, match="needs 100 available images"):
        build_metadata_audit(
            manifest, site_id="KGA:A01", sample_size=100, checksums_path=checksums
        )


def test_cli_writes_json_artifact(tmp_path):
    manifest, checksums = _write_fixture(tmp_path)
    output = tmp_path / "audit.json"
    assert main([
        "--manifest", str(manifest),
        "--site-id", "KGA:A01",
        "--checksums", str(checksums),
        "--output", str(output),
    ]) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["sample_size"] == 100

