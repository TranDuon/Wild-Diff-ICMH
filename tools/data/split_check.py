"""Leakage and contract gate for camera-trap manifests (data/manifests/*.jsonl).

This module owns the data contract for the manifest rows produced by
``build_manifests.py`` and consumed by ``download_images.py`` and any later
train/eval job: the required fields, the valid split values, and the
site/sequence leakage rules. ``assert_no_leakage`` is the single function all
other tools call before trusting a manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

# The full manifest row contract (CAMERA_TRAP_FINE_TUNING_PLAN.md item 2, plus
# provenance/ranking fields needed by the downloader and the split gate).
REQUIRED_FIELDS = (
    "image_id",
    "source",
    "source_version",
    "license",
    "relative_path",
    "source_file_name",
    "site_id",
    "sequence_id",
    "frame_num",
    "datetime",
    "illumination",
    "illumination_source",
    "species",
    "is_empty",
    "boxes",
    "width",
    "height",
    "split",
    "subset",
    "sample_rank",
    "sha256",
    "null_reasons",
)

VALID_SPLITS = {"train", "val", "test"}

# Kgalagadi follows the comparison paper's site-specific fine-tuning protocol:
# every site intentionally has train/val/test data, but a burst/sequence must
# still belong to exactly one split.  Other sources retain the stricter
# site-disjoint rule used for external-generalisation evaluation.
WITHIN_SITE_SPLIT_SOURCES = {"snapshot_kgalagadi"}


class SplitLeakageError(RuntimeError):
    """Raised when a manifest (or its derived list files) violate the contract."""


def load_manifest_rows(paths: Iterable[Path]) -> List[dict]:
    """Read one or more JSONL manifests into a flat list of row dicts.

    Each row gets an extra, non-contract ``_manifest_path`` key recording
    which file it came from; callers that only care about the contract
    fields should ignore it.
    """
    rows: List[dict] = []
    for path in paths:
        path = Path(path)
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                row["_manifest_path"] = str(path)
                rows.append(row)
    return rows


def _is_unsafe_relative_path(value) -> bool:
    if not isinstance(value, str) or not value:
        return True
    normalized = value.replace("\\", "/")
    if normalized.startswith("/"):
        return True
    if len(normalized) >= 2 and normalized[1] == ":":  # drive letter, e.g. C:/x.jpg
        return True
    if ".." in normalized.split("/"):
        return True
    return False


def find_violations(rows: Sequence[dict]) -> List[str]:
    """Return a list of human-readable contract/leakage violation messages."""
    violations: List[str] = []
    source_site_to_splits: Dict[tuple, set] = {}
    seq_to_splits: Dict[str, set] = {}
    image_id_counts: Dict[str, int] = {}

    for row in rows:
        image_id = row.get("image_id")
        label = image_id if image_id is not None else "<missing image_id>"

        missing = [f for f in REQUIRED_FIELDS if f not in row]
        if missing:
            violations.append(f"row {label}: missing required field(s) {missing}")

        null_reasons = row.get("null_reasons")
        if not isinstance(null_reasons, dict):
            null_reasons = {}
        for field in REQUIRED_FIELDS:
            if field == "null_reasons":
                continue
            if field in row and row[field] is None and field not in null_reasons:
                violations.append(
                    f"row {label}: field '{field}' is None with no null_reasons entry"
                )

        split = row.get("split")
        if split not in VALID_SPLITS:
            violations.append(f"row {label}: invalid split {split!r}")

        source = row.get("source")
        site_id = row.get("site_id")
        if site_id is not None and split in VALID_SPLITS:
            source_site_to_splits.setdefault((source, site_id), set()).add(split)

        sequence_id = row.get("sequence_id")
        if sequence_id is not None and split in VALID_SPLITS:
            seq_to_splits.setdefault(sequence_id, set()).add(split)

        if image_id is not None:
            image_id_counts[image_id] = image_id_counts.get(image_id, 0) + 1

        if "relative_path" in row and _is_unsafe_relative_path(row.get("relative_path")):
            violations.append(
                f"row {label}: unsafe relative_path {row.get('relative_path')!r}"
            )

    for (source, site_id), splits in source_site_to_splits.items():
        if source not in WITHIN_SITE_SPLIT_SOURCES and len(splits) > 1:
            violations.append(
                f"site '{site_id}' from source '{source}' present in multiple splits: "
                f"{sorted(splits)}"
            )
    for sequence_id, splits in seq_to_splits.items():
        if len(splits) > 1:
            violations.append(
                f"sequence '{sequence_id}' present in multiple splits: {sorted(splits)}"
            )
    for image_id, count in image_id_counts.items():
        if count > 1:
            violations.append(f"duplicate image_id '{image_id}' appears {count} times")

    return violations


def _match_list_paths(rows: Sequence[dict], list_files: Dict[str, str]) -> List[str]:
    violations: List[str] = []
    row_by_relpath = [
        (row["relative_path"].replace("\\", "/"), row)
        for row in rows
        if isinstance(row.get("relative_path"), str)
    ]

    for split, list_path in list_files.items():
        list_path = Path(list_path)
        if not list_path.exists():
            violations.append(f"list file for split '{split}' not found: {list_path}")
            continue
        with list_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                normalized = line.replace("\\", "/")
                matches = [row for rel, row in row_by_relpath if normalized.endswith(rel)]
                if not matches:
                    violations.append(
                        f"list file '{list_path}' path matches no manifest row: {line}"
                    )
                    continue
                if not any(row.get("split") == split for row in matches):
                    found_splits = sorted({row.get("split") for row in matches})
                    violations.append(
                        f"list file '{list_path}' declares split '{split}' but path "
                        f"belongs to split(s) {found_splits}: {line}"
                    )
    return violations


def _check_build_info(manifest_paths: Iterable[Path], build_info_path) -> List[str]:
    violations: List[str] = []
    build_info_path = Path(build_info_path)
    if not build_info_path.exists():
        return [f"build_info file not found: {build_info_path}"]

    info = json.loads(build_info_path.read_text(encoding="utf-8"))
    output_sha256 = info.get("output_sha256", {})
    for path in manifest_paths:
        path = Path(path)
        name = path.name
        expected = output_sha256.get(name)
        if expected is None:
            violations.append(f"build_info has no output_sha256 entry for '{name}'")
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            violations.append(
                f"manifest '{name}' sha256 mismatch with build_info: "
                f"expected {expected}, got {actual}"
            )
    return violations


def assert_no_leakage(
    manifest_paths: Iterable[Path],
    list_files: Optional[Dict[str, str]] = None,
    build_info=None,
) -> List[dict]:
    """Raise SplitLeakageError if any manifest/list-file/build_info check fails.

    Returns the loaded rows on success, so callers can reuse them.
    """
    manifest_paths = [Path(p) for p in manifest_paths]
    rows = load_manifest_rows(manifest_paths)

    violations = find_violations(rows)
    if list_files:
        violations.extend(_match_list_paths(rows, list_files))
    if build_info:
        violations.extend(_check_build_info(manifest_paths, build_info))

    if violations:
        total = len(violations)
        preview = violations[:20]
        message = "\n".join(f"  - {v}" for v in preview)
        if total > 20:
            message += f"\n  ... and {total - 20} more"
        raise SplitLeakageError(f"{total} leakage/contract violation(s):\n{message}")

    return rows


def _parse_kv_args(values) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for item in values or []:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"expected KEY=VALUE, got {item!r}")
        result[key] = value
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", nargs="+", help="Manifest JSONL path(s) to check")
    parser.add_argument(
        "--list",
        action="append",
        default=[],
        metavar="SPLIT=PATH",
        help="A .list file to cross-check against the manifest (repeatable)",
    )
    parser.add_argument(
        "--build-info",
        default=None,
        help="build_info.json path; its output_sha256 is checked against the manifests on disk",
    )
    return parser


def main(argv=None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    list_files = _parse_kv_args(args.list) or None
    try:
        rows = assert_no_leakage(args.manifests, list_files=list_files, build_info=args.build_info)
    except SplitLeakageError as exc:
        print(str(exc))
        print("split_check: FAILED")
        return 1

    sites = {r.get("site_id") for r in rows}
    sequences = {r.get("sequence_id") for r in rows}
    print(
        f"split_check: OK ({len(rows)} rows, {len(sites)} sites, {len(sequences)} sequences)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
