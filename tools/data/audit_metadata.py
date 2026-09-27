"""Audit EXIF-first camera-trap metadata coverage and write a JSON artifact."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
import sys

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import download_images  # noqa: E402
import split_check  # noqa: E402


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _record_for_row(row: dict, checksums: dict[str, dict], data_root: Path | None) -> tuple[dict, str]:
    record = checksums.get(row["image_id"])
    if record is not None:
        return record, "checksums"
    if data_root is None:
        raise FileNotFoundError(f"no checksum metadata for {row['image_id']}")
    image_path = download_images.safe_join(data_root, row["relative_path"])
    if not image_path.is_file():
        raise FileNotFoundError(f"image not found for metadata audit: {image_path}")
    return download_images._extract_image_metadata(image_path.read_bytes()), "image"


def build_metadata_audit(
    manifest: str | Path,
    *,
    site_id: str,
    sample_size: int = 100,
    checksums_path: str | Path | None = None,
    data_root: str | Path | None = None,
) -> dict:
    """Return deterministic coverage counts for the first available site rows."""
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    rows = split_check.assert_no_leakage([Path(manifest)])
    site_rows = sorted(
        (row for row in rows if row.get("site_id") == site_id),
        key=lambda row: (int(row.get("sample_rank", 0)), str(row["image_id"])),
    )
    checksums = (
        download_images.load_checksums(checksums_path)
        if checksums_path is not None and Path(checksums_path).is_file()
        else {}
    )
    root = Path(data_root) if data_root is not None else None

    audited: list[tuple[dict, dict, str]] = []
    missing_assets: list[str] = []
    for row in site_rows:
        try:
            record, evidence_source = _record_for_row(row, checksums, root)
        except FileNotFoundError:
            missing_assets.append(str(row["image_id"]))
            continue
        audited.append((row, record, evidence_source))
        if len(audited) == sample_size:
            break

    if len(audited) < sample_size:
        raise RuntimeError(
            f"metadata audit needs {sample_size} available images for {site_id}, "
            f"found {len(audited)} (missing assets: {len(missing_assets)})"
        )

    datetime_sources: Counter[str] = Counter()
    evidence_sources: Counter[str] = Counter()
    location_usable = 0
    grayscale_illumination_disagreements = 0
    for row, record, evidence_source in audited:
        evidence_sources[evidence_source] += 1
        if record.get("exif_datetime"):
            datetime_sources["exif"] += 1
        elif row.get("datetime"):
            datetime_sources["manifest"] += 1
        else:
            datetime_sources["missing"] += 1
        if row.get("site_id"):
            location_usable += 1
        grayscale = record.get("is_grayscale")
        illumination = row.get("illumination")
        if grayscale is not None and illumination in {"day", "night"}:
            if bool(grayscale) != (illumination == "night"):
                grayscale_illumination_disagreements += 1

    usable_datetime = sample_size - datetime_sources["missing"]
    return {
        "schema_version": 1,
        "manifest": str(Path(manifest)),
        "site_id": site_id,
        "requested_sample_size": sample_size,
        "sample_size": sample_size,
        "selection": "first_available_by_sample_rank",
        "datetime_source_counts": dict(sorted(datetime_sources.items())),
        "datetime_usable_count": usable_datetime,
        "datetime_usable_ratio": usable_datetime / sample_size,
        "location_source": "manifest.site_id",
        "location_usable_count": location_usable,
        "location_usable_ratio": location_usable / sample_size,
        "evidence_source_counts": dict(sorted(evidence_sources.items())),
        "grayscale_illumination_disagreements": grayscale_illumination_disagreements,
        "fallback_policy": "EXIF datetime, then manifest datetime; location from manifest site_id",
        "image_ids": [row["image_id"] for row, _, _ in audited],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--site-id", required=True)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--checksums", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output", required=True)
    return parser


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    report = build_metadata_audit(
        args.manifest,
        site_id=args.site_id,
        sample_size=args.sample_size,
        checksums_path=args.checksums,
        data_root=args.data_root,
    )
    output = Path(args.output)
    _atomic_write_json(output, report)
    print(
        f"metadata audit: OK ({report['sample_size']} images, "
        f"datetime={report['datetime_usable_ratio']:.1%}, "
        f"location={report['location_usable_ratio']:.1%})"
    )
    print(f"metadata audit artifact: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

