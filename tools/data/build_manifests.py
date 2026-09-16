"""Build the frozen Serengeti train/val (and later Kgalagadi test) manifests.

Reads the official LILA COCO-CT bbox JSON, emits one JSONL row per image
following the ``split_check.REQUIRED_FIELDS`` contract, splits by site so no
site or sequence crosses train/val, and gates its own output through
``split_check.assert_no_leakage`` before returning 0.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import io
import json
import random
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import split_check  # noqa: E402

_REPO_ROOT = _THIS_DIR.parents[1]


def load_coco_ct(path) -> Tuple[dict, dict]:
    """Load a COCO-CT style JSON, accepting a bare .json or a single-json .zip.

    Returns ``(data, meta)`` where meta holds the repo-relative posix path,
    the sha256 of the file on disk, and the json member name actually read.
    """
    path = Path(path)
    raw_bytes = path.read_bytes()
    sha256 = hashlib.sha256(raw_bytes).hexdigest()

    try:
        repo_relative = path.resolve().relative_to(_REPO_ROOT).as_posix()
    except ValueError:
        repo_relative = path.resolve().as_posix()

    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
            json_names = [n for n in zf.namelist() if n.lower().endswith(".json")]
            if len(json_names) != 1:
                raise ValueError(
                    f"{path} must contain exactly one .json member, found {json_names}"
                )
            member_name = json_names[0]
            data = json.loads(zf.read(member_name))
    else:
        member_name = path.name
        data = json.loads(raw_bytes)

    meta = {
        "repo_relative_path": repo_relative,
        "sha256": sha256,
        "member_name": member_name,
    }
    return data, meta


def illumination_from_datetime(value) -> Tuple[Optional[str], Optional[str]]:
    """Return (label, reason). label is None with a reason when unparseable."""
    if not value:
        return None, "datetime is null"
    text = str(value).strip().replace("T", " ")
    try:
        dt = datetime.datetime.strptime(text, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None, f"datetime {value!r} does not match 'YYYY-MM-DD HH:MM:SS'"
    label = "day" if 6 <= dt.hour <= 18 else "night"
    return label, None


_COPIED_FIELD_REASONS = {
    "frame_num": "missing frame_num in source image record",
    "datetime": "missing datetime in source image record",
    "width": "missing width in source image record",
    "height": "missing height in source image record",
}


def make_serengeti_row(image: dict, anns: List[dict], category_names: Dict[int, str], source_version: str) -> dict:
    illumination, illum_reason = illumination_from_datetime(image.get("datetime"))

    boxes = [
        {
            "category": category_names.get(ann["category_id"], str(ann["category_id"])),
            "bbox_xywh": ann["bbox"],
        }
        for ann in sorted(anns, key=lambda a: a["id"])
    ]

    row = {
        "image_id": f"SER:{image['id']}",
        "source": "snapshot_serengeti",
        "source_version": source_version,
        "license": "CDLA-Permissive-1.0",
        "relative_path": f"snapshot_serengeti/{image['file_name']}".replace("\\", "/"),
        "source_file_name": image["file_name"],
        "site_id": f"SER:{image['location']}",
        "sequence_id": f"SER:{image['seq_id']}",
        "frame_num": image.get("frame_num"),
        "datetime": image.get("datetime"),
        "width": image.get("width"),
        "height": image.get("height"),
        "illumination": illumination,
        "illumination_source": "datetime_hour_proxy",
        "species": None,
        "is_empty": None,
        "boxes": boxes,
        "subset": "bbox_subset",
        "split": None,
        "sample_rank": None,
        "sha256": None,
        "null_reasons": {},
    }

    null_reasons = row["null_reasons"]
    null_reasons["species"] = (
        "species labels live in the 5.5 GB SnapshotSerengeti_S1-11_v2.1.json, "
        "which is intentionally not loaded"
    )
    null_reasons["is_empty"] = (
        "the 20190903 bbox file removed boxes under 400 px^2, so zero boxes "
        "does not prove an empty frame"
    )
    null_reasons["sha256"] = "filled by download_images.py in the checksums file"
    if illumination is None:
        null_reasons["illumination"] = illum_reason
    for field, reason in _COPIED_FIELD_REASONS.items():
        if row[field] is None:
            null_reasons[field] = reason
    # split/sample_rank are assigned later by split_and_sample_serengeti.
    null_reasons["split"] = "not yet assigned"
    null_reasons["sample_rank"] = "not yet assigned"

    return row


def split_and_sample_serengeti(
    rows: List[dict],
    *,
    seed,
    train_size: int,
    val_size: int,
    val_site_fraction: float,
    max_per_sequence: int,
) -> List[dict]:
    """Tracer-level site-disjoint split (Task 2 replaces the internals)."""
    site_ids = sorted({r["site_id"] for r in rows})

    site_order = site_ids[:]
    random.Random(f"{seed}:val_sites").shuffle(site_order)
    n_val_sites = max(1, round(val_site_fraction * len(site_order)))
    val_sites = set(site_order[:n_val_sites])
    train_sites = set(site_order[n_val_sites:])

    def cap_per_sequence(candidate_rows, purpose):
        by_seq: Dict[str, List[dict]] = {}
        for row in candidate_rows:
            by_seq.setdefault(row["sequence_id"], []).append(row)
        rng = random.Random(f"{seed}:{purpose}:seq_cap")
        kept: List[dict] = []
        for seq_id in sorted(by_seq):
            frames = by_seq[seq_id][:]
            rng.shuffle(frames)
            kept.extend(frames[:max_per_sequence])
        return kept

    def pick(candidate_rows, n, purpose):
        rng = random.Random(f"{seed}:{purpose}:pick")
        pool = candidate_rows[:]
        rng.shuffle(pool)
        return pool[:n]

    val_candidates = cap_per_sequence([r for r in rows if r["site_id"] in val_sites], "val")
    train_candidates = cap_per_sequence([r for r in rows if r["site_id"] in train_sites], "train")

    val_rows = pick(val_candidates, val_size, "val")
    train_rows = pick(train_candidates, train_size, "train")

    for split_name, split_rows in (("val", val_rows), ("train", train_rows)):
        for rank, row in enumerate(split_rows):
            row["split"] = split_name
            row["sample_rank"] = rank
            row["null_reasons"].pop("split", None)
            row["null_reasons"].pop("sample_rank", None)

    return train_rows + val_rows


def write_jsonl(rows: List[dict], path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sorted_rows = sorted(
        rows,
        key=lambda r: (
            r.get("split") or "",
            r.get("sample_rank") if r.get("sample_rank") is not None else -1,
            r.get("image_id") or "",
        ),
    )
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in sorted_rows:
            clean = {k: v for k, v in row.items() if not k.startswith("_")}
            fh.write(json.dumps(clean, sort_keys=True, ensure_ascii=False))
            fh.write("\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--serengeti-bbox-json",
        default="data/raw/snapshot_serengeti/bboxes.json.zip",
    )
    parser.add_argument("--out-dir", default="data/manifests")
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--train-size", type=int, default=4000)
    parser.add_argument("--val-size", type=int, default=400)
    parser.add_argument("--val-site-fraction", type=float, default=0.1)
    parser.add_argument("--max-per-sequence", type=int, default=1)
    return parser


def main(argv=None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    data, meta = load_coco_ct(args.serengeti_bbox_json)
    category_names = {c["id"]: c["name"] for c in data["categories"]}
    anns_by_image: Dict[str, List[dict]] = {}
    for ann in data["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    rows = [
        make_serengeti_row(image, anns_by_image.get(image["id"], []), category_names, meta["member_name"])
        for image in data["images"]
    ]

    try:
        rows = split_and_sample_serengeti(
            rows,
            seed=args.seed,
            train_size=args.train_size,
            val_size=args.val_size,
            val_site_fraction=args.val_site_fraction,
            max_per_sequence=args.max_per_sequence,
        )
    except ValueError as exc:
        print(f"build_manifests: {exc}")
        return 1

    out_path = Path(args.out_dir) / "serengeti_trainval.jsonl"
    write_jsonl(rows, out_path)

    try:
        split_check.assert_no_leakage([out_path])
    except split_check.SplitLeakageError as exc:
        print(str(exc))
        return 1

    print(f"build_manifests: wrote {out_path} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
