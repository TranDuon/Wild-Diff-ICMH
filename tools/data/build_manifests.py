"""Build reproducible camera-trap manifests for Wild-Diff-ICMH.

Snapshot Kgalagadi is the primary experiment dataset.  It is split *within
each camera site* by sequence (70/15/15 by default), matching the
site-specific fine-tuning protocol of the comparison paper while preventing
frames from the same burst leaking across train/validation/test.  Snapshot
Serengeti remains an optional, site-disjoint external-generalisation set.

All output is gated through ``split_check.assert_no_leakage`` before the
builder returns successfully.
"""
from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import io
import json
import random
import sys
import zipfile
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import split_check  # noqa: E402

_REPO_ROOT = _THIS_DIR.parents[1]


# ---------------------------------------------------------------------------
# Source loading
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Small deterministic helpers shared by both sources
# ---------------------------------------------------------------------------


def _interleave_by_proportion(groups: List[List[dict]]) -> List[dict]:
    """Merge already-ordered groups, weighted round-robin by group size.

    Keeps any prefix of the result mixed across groups in proportion to
    their sizes (e.g. day/night, or nonempty/empty_check), instead of
    emitting one whole group before the next.
    """
    remaining = [list(g) for g in groups]
    total = sum(len(g) for g in remaining)
    if total == 0:
        return []
    weights = [len(g) / total for g in remaining]
    acc = [0.0] * len(remaining)
    result: List[dict] = []
    while any(remaining):
        for i in range(len(remaining)):
            acc[i] += weights[i]
        idx = max(
            (i for i in range(len(remaining)) if remaining[i]),
            key=lambda i: acc[i],
        )
        result.append(remaining[idx].pop(0))
        acc[idx] -= 1.0
    return result


def _cap_per_sequence(candidates: List[dict], max_per_sequence: int, seed_ns: str) -> List[dict]:
    by_seq: Dict[str, List[dict]] = {}
    for row in candidates:
        by_seq.setdefault(row["sequence_id"], []).append(row)
    kept: List[dict] = []
    for seq_id in sorted(by_seq):
        frames = by_seq[seq_id][:]
        random.Random(f"{seed_ns}:{seq_id}:frames").shuffle(frames)
        kept.extend(frames[:max_per_sequence])
    return kept


def _round_robin_by_site(pool: List[dict], quota: int, seed_ns: str) -> Tuple[List[dict], List[dict]]:
    """Pick up to quota rows from pool, cycling through sites in seeded order.

    Returns (selected, leftover) so callers can borrow leftovers to fill a
    shortfall in a sibling stratum.
    """
    by_site: Dict[str, List[dict]] = {}
    for row in pool:
        by_site.setdefault(row["site_id"], []).append(row)
    site_order = sorted(by_site)
    random.Random(f"{seed_ns}:site_order").shuffle(site_order)
    for site in site_order:
        random.Random(f"{seed_ns}:{site}:order").shuffle(by_site[site])

    remaining = {site: list(rows) for site, rows in by_site.items()}
    selected: List[dict] = []
    while len(selected) < quota:
        progressed = False
        for site in site_order:
            if remaining[site]:
                selected.append(remaining[site].pop(0))
                progressed = True
                if len(selected) >= quota:
                    break
        if not progressed:
            break
    leftover = [row for site in site_order for row in remaining[site]]
    return selected, leftover


def _largest_remainder_counts(total: int, fractions: Dict[str, float]) -> Dict[str, int]:
    """Convert fractions to deterministic integer counts that sum to total.

    When at least as many groups as splits are available, every non-zero
    split receives at least one group.  This is important for the smallest
    Kgalagadi sites: a site-specific model needs a validation and test split,
    not only a training split.
    """
    if total < 0:
        raise ValueError("total must be non-negative")
    if not fractions or any(v < 0 for v in fractions.values()):
        raise ValueError("split fractions must be non-negative")
    fraction_sum = sum(fractions.values())
    if not math.isclose(fraction_sum, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"split fractions must sum to 1.0, got {fraction_sum}")

    raw = {name: total * value for name, value in fractions.items()}
    counts = {name: int(math.floor(value)) for name, value in raw.items()}
    for name in sorted(fractions, key=lambda n: (-(raw[n] - counts[n]), n))[: total - sum(counts.values())]:
        counts[name] += 1

    nonzero = [name for name, value in fractions.items() if value > 0]
    if total >= len(nonzero):
        for empty_name in [name for name in nonzero if counts[name] == 0]:
            donors = [name for name in nonzero if counts[name] > 1]
            if not donors:
                break
            donor = max(donors, key=lambda n: (counts[n] - raw[n], counts[n], n))
            counts[donor] -= 1
            counts[empty_name] += 1
    return counts


# ---------------------------------------------------------------------------
# Snapshot Serengeti
# ---------------------------------------------------------------------------

_COPIED_FIELD_REASONS = {
    "frame_num": "missing frame_num in source image record",
    "datetime": "missing datetime in source image record",
    "width": "missing width in source image record",
    "height": "missing height in source image record",
}

_EXCLUDED_BOX_CATEGORIES = {"person", "vehicle"}


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


def sample_balanced(
    candidates: List[dict],
    n: int,
    seed_ns: str,
    max_per_sequence: int,
    night_fraction: Optional[float],
) -> List[dict]:
    """Site-diverse, day/night-stratified pick of n rows from candidates.

    ``seed_ns`` is a namespace string (not a Random instance): every draw
    inside this function derives its own purpose-scoped ``random.Random``
    from it, so results are reproducible and independent of call order.
    """
    capped = _cap_per_sequence(candidates, max_per_sequence, f"{seed_ns}:seq_cap")

    day_pool = [r for r in capped if r["illumination"] == "day"]
    night_pool = [r for r in capped if r["illumination"] == "night"]

    if night_fraction is None:
        total = len(day_pool) + len(night_pool)
        night_fraction_eff = (len(night_pool) / total) if total else 0.0
    else:
        night_fraction_eff = night_fraction

    night_quota = round(n * night_fraction_eff)
    day_quota = n - night_quota
    if n >= 2:
        if night_quota == 0 and night_pool:
            night_quota, day_quota = 1, n - 1
        elif day_quota == 0 and day_pool:
            day_quota, night_quota = 1, n - 1

    day_selected, day_leftover = _round_robin_by_site(day_pool, day_quota, f"{seed_ns}:day")
    night_selected, night_leftover = _round_robin_by_site(night_pool, night_quota, f"{seed_ns}:night")

    day_shortfall = day_quota - len(day_selected)
    if day_shortfall > 0 and night_leftover:
        day_selected.extend(night_leftover[:day_shortfall])
        night_leftover = night_leftover[day_shortfall:]
    night_shortfall = night_quota - len(night_selected)
    if night_shortfall > 0 and day_leftover:
        night_selected.extend(day_leftover[:night_shortfall])
        day_leftover = day_leftover[night_shortfall:]

    ordered = _interleave_by_proportion([day_selected, night_selected])
    return ordered[:n]


def split_and_sample_serengeti(
    rows: List[dict],
    *,
    seed,
    train_size: int,
    val_size: int,
    val_site_fraction: float,
    max_per_sequence: int,
    night_fraction: Optional[float] = None,
    report: Optional[dict] = None,
) -> List[dict]:
    excluded = {"illumination_none": 0, "person_or_vehicle_box": 0}
    eligible: List[dict] = []
    for row in rows:
        if row["illumination"] is None:
            excluded["illumination_none"] += 1
            continue
        box_categories = {b["category"] for b in row["boxes"]}
        if box_categories & _EXCLUDED_BOX_CATEGORIES:
            excluded["person_or_vehicle_box"] += 1
            continue
        eligible.append(row)

    site_order = sorted({r["site_id"] for r in eligible})
    random.Random(f"{seed}:val_sites").shuffle(site_order)

    min_val_sites = max(1, round(val_site_fraction * len(site_order)))

    def val_pool_for(n_sites: int) -> List[dict]:
        chosen = set(site_order[:n_sites])
        candidates = [r for r in eligible if r["site_id"] in chosen]
        return _cap_per_sequence(candidates, max_per_sequence, f"{seed}:val_probe:{n_sites}")

    n_val_sites = min_val_sites
    while True:
        pool = val_pool_for(n_val_sites)
        has_day = any(r["illumination"] == "day" for r in pool)
        has_night = any(r["illumination"] == "night" for r in pool)
        if n_val_sites >= min_val_sites and len(pool) >= val_size and has_day and has_night:
            break
        if n_val_sites >= len(site_order):
            break
        n_val_sites += 1

    val_sites = set(site_order[:n_val_sites])
    train_sites = set(site_order[n_val_sites:])

    train_candidates_pool = [r for r in eligible if r["site_id"] in train_sites]
    val_candidates_pool = [r for r in eligible if r["site_id"] in val_sites]

    def seq_capped_count(candidates: List[dict]) -> int:
        by_seq: Dict[str, int] = {}
        for row in candidates:
            by_seq[row["sequence_id"]] = by_seq.get(row["sequence_id"], 0) + 1
        return sum(min(count, max_per_sequence) for count in by_seq.values())

    train_pool_size = seq_capped_count(train_candidates_pool)
    if train_pool_size < train_size:
        raise ValueError(
            f"train pool has only {train_pool_size} eligible rows after the site split "
            f"and per-sequence cap, need {train_size} (shortfall "
            f"{train_size - train_pool_size})"
        )

    val_rows = sample_balanced(val_candidates_pool, val_size, f"{seed}:val", max_per_sequence, night_fraction)
    train_rows = sample_balanced(train_candidates_pool, train_size, f"{seed}:train", max_per_sequence, night_fraction)

    for split_name, split_rows in (("val", val_rows), ("train", train_rows)):
        for rank, row in enumerate(split_rows):
            row["split"] = split_name
            row["sample_rank"] = rank
            row["null_reasons"].pop("split", None)
            row["null_reasons"].pop("sample_rank", None)

    if report is not None:
        report["excluded"] = excluded
        report["val_sites"] = len(val_sites)
        report["train_sites"] = len(train_sites)
        report["val_site_ids"] = sorted(val_sites)

    return train_rows + val_rows


# ---------------------------------------------------------------------------
# Snapshot Kgalagadi
# ---------------------------------------------------------------------------


def build_kgalagadi_rows(
    coco: dict,
    meta: dict,
    *,
    seed,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    report: Optional[dict] = None,
) -> List[dict]:
    """Build the full non-human Kgalagadi corpus and split within each site.

    The official JSON contains 10,357 images.  Removing the 135 human-labelled
    images leaves the 10,222 images reported by the comparison paper.  Every
    frame of a sequence receives the same split.  Assignment is deterministic
    and balances each site's day/night and empty/non-empty sequence strata as
    closely as the site size permits.
    """
    fractions = {
        "train": train_fraction,
        "val": val_fraction,
        "test": test_fraction,
    }
    _largest_remainder_counts(0, fractions)  # validates the fractions

    category_names = {c["id"]: c["name"] for c in coco["categories"]}
    empty_category_ids = {cid for cid, name in category_names.items() if name == "empty"}
    human_category_ids = {cid for cid, name in category_names.items() if "human" in name.lower()}

    anns_by_image: Dict[str, List[dict]] = {}
    anns_by_seq: Dict[str, List[dict]] = {}
    for ann in coco["annotations"]:
        image_id = ann.get("image_id")
        if image_id:
            anns_by_image.setdefault(image_id, []).append(ann)
        seq_id = ann.get("seq_id")
        if seq_id:
            anns_by_seq.setdefault(seq_id, []).append(ann)

    source_version = meta["member_name"]
    if source_version.lower().endswith(".json"):
        source_version = source_version[: -len(".json")]

    excluded = {"human_label": 0, "no_annotation": 0, "illumination_none": 0}
    mapping_notes: List[str] = []
    seq_fallback_used = False

    candidates: List[dict] = []
    for image in coco["images"]:
        image_id_raw = image["id"]
        anns = anns_by_image.get(image_id_raw)
        if not anns:
            seq_anns = anns_by_seq.get(image.get("seq_id"))
            if seq_anns:
                anns = seq_anns
                seq_fallback_used = True
        if not anns:
            excluded["no_annotation"] += 1
            continue

        cat_ids = {a["category_id"] for a in anns}
        if cat_ids & human_category_ids:
            excluded["human_label"] += 1
            continue

        illumination, _reason = illumination_from_datetime(image.get("datetime"))
        if illumination is None:
            excluded["illumination_none"] += 1
            continue

        is_empty = bool(cat_ids) and cat_ids <= empty_category_ids
        species = sorted({category_names[cid] for cid in cat_ids if cid not in empty_category_ids})

        row = {
            "image_id": f"KGA:{image_id_raw}",
            "source": "snapshot_kgalagadi",
            "source_version": source_version,
            "license": "CDLA-Permissive-1.0",
            "relative_path": f"snapshot_kgalagadi/{image['file_name']}".replace("\\", "/"),
            "source_file_name": image["file_name"],
            "site_id": f"KGA:{image['location']}",
            "sequence_id": f"KGA:{image['seq_id']}",
            "frame_num": image.get("frame_num"),
            "datetime": image.get("datetime"),
            "width": image.get("width"),
            "height": image.get("height"),
            "illumination": illumination,
            "illumination_source": "datetime_hour_proxy",
            "species": species,
            "is_empty": is_empty,
            "boxes": None,
            "subset": "empty" if is_empty else "nonempty",
            "split": None,
            "sample_rank": None,
            "sha256": None,
            "null_reasons": {},
        }
        null_reasons = row["null_reasons"]
        null_reasons["boxes"] = (
            "Snapshot Kgalagadi has no bbox ground truth; MegaDetector boxes "
            "are produced at eval time"
        )
        null_reasons["sha256"] = "filled by download_images.py in the checksums file"
        for field in ("frame_num", "width", "height"):
            if row[field] is None:
                null_reasons[field] = f"missing {field} in source image record"
        null_reasons["split"] = "not yet assigned"
        null_reasons["sample_rank"] = "not yet assigned"
        candidates.append(row)

    if seq_fallback_used:
        mapping_notes.append(
            "some annotations matched by seq_id fallback (no per-image annotation found)"
        )

    rows_by_site_sequence: Dict[str, Dict[str, List[dict]]] = {}
    for row in candidates:
        rows_by_site_sequence.setdefault(row["site_id"], {}).setdefault(row["sequence_id"], []).append(row)

    assigned: List[dict] = []
    site_split_sequences: Dict[str, Dict[str, int]] = {}
    for site_id in sorted(rows_by_site_sequence):
        by_sequence = rows_by_site_sequence[site_id]
        targets = _largest_remainder_counts(len(by_sequence), fractions)
        assigned_counts = {name: 0 for name in fractions}
        assigned_strata = {name: collections.Counter() for name in fractions}

        sequence_records = []
        for sequence_id, sequence_rows in by_sequence.items():
            illumination_values = {r["illumination"] for r in sequence_rows}
            illumination = next(iter(illumination_values)) if len(illumination_values) == 1 else "mixed"
            if all(r["is_empty"] is True for r in sequence_rows):
                occupancy = "empty"
            elif any(r["is_empty"] is False for r in sequence_rows):
                occupancy = "nonempty"
            else:
                occupancy = "unknown"
            sequence_records.append((sequence_id, (illumination, occupancy), sequence_rows))

        random.Random(f"{seed}:{site_id}:sequences").shuffle(sequence_records)
        # Rare strata first; otherwise a rare night/non-empty sequence tends to
        # be consumed by train before validation/test receive representation.
        stratum_sizes = collections.Counter(stratum for _, stratum, _ in sequence_records)
        sequence_records.sort(key=lambda item: (stratum_sizes[item[1]], item[1]))

        for sequence_id, stratum, sequence_rows in sequence_records:
            available = [name for name in fractions if assigned_counts[name] < targets[name]]
            if not available:  # defensive; integer targets should make this unreachable
                raise RuntimeError(f"no split capacity left for {site_id}/{sequence_id}")
            split = min(
                available,
                key=lambda name: (
                    assigned_strata[name][stratum] / max(1, targets[name]),
                    assigned_counts[name] / max(1, targets[name]),
                    ("train", "val", "test").index(name),
                ),
            )
            assigned_counts[split] += 1
            assigned_strata[split][stratum] += 1
            for row in sequence_rows:
                row["split"] = split
                row["null_reasons"].pop("split", None)
                assigned.append(row)
        site_split_sequences[site_id] = dict(assigned_counts)

    # Rank each split with a site round-robin ordering.  Prefix downloads used
    # for smoke tests therefore cover many sites instead of only site A01.
    selected: List[dict] = []
    for split in ("train", "val", "test"):
        split_rows = [row for row in assigned if row["split"] == split]
        ordered, leftover = _round_robin_by_site(split_rows, len(split_rows), f"{seed}:kga:{split}:rank")
        assert not leftover
        for rank, row in enumerate(ordered):
            row["sample_rank"] = rank
            row["null_reasons"].pop("sample_rank", None)
        selected.extend(ordered)

    if report is not None:
        report["excluded"] = excluded
        report["mapping_notes"] = mapping_notes
        report["sites_total"] = len(rows_by_site_sequence)
        report["sites_covered"] = len({r["site_id"] for r in selected})
        report["site_split_sequences"] = site_split_sequences
        report["fractions"] = fractions
        report["shortfalls"] = {}

    return selected


# ---------------------------------------------------------------------------
# Output + build_info
# ---------------------------------------------------------------------------


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


def _manifest_stats(rows: List[dict]) -> dict:
    by_split: Dict[str, List[dict]] = {}
    for row in rows:
        by_split.setdefault(row["split"], []).append(row)

    stats = {}
    for split, split_rows in by_split.items():
        sites = {r["site_id"] for r in split_rows}
        sequences = {r["sequence_id"] for r in split_rows}
        day = sum(1 for r in split_rows if r.get("illumination") == "day")
        night = sum(1 for r in split_rows if r.get("illumination") == "night")
        boxes_lists = [r.get("boxes") for r in split_rows if isinstance(r.get("boxes"), list)]
        entry = {
            "images": len(split_rows),
            "sites": len(sites),
            "sequences": len(sequences),
            "day": day,
            "night": night,
            "zero_box_images": sum(1 for b in boxes_lists if len(b) == 0),
            "total_boxes": sum(len(b) for b in boxes_lists),
            "subset_counts": dict(collections.Counter(r.get("subset") for r in split_rows)),
        }
        species_lists = [r.get("species") for r in split_rows if isinstance(r.get("species"), list)]
        if species_lists:
            species_counter: collections.Counter = collections.Counter()
            for s_list in species_lists:
                species_counter.update(s_list)
            entry["species_counts"] = dict(species_counter)
        stats[split] = entry
    return stats


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--serengeti-bbox-json",
        default="data/raw/snapshot_serengeti/bboxes.json.zip",
    )
    parser.add_argument("--skip-serengeti", action="store_true")
    parser.add_argument("--out-dir", default="data/manifests")
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--train-size", type=int, default=4000)
    parser.add_argument("--val-size", type=int, default=400)
    parser.add_argument("--val-site-fraction", type=float, default=0.1)
    parser.add_argument("--max-per-sequence", type=int, default=1)
    parser.add_argument("--night-fraction", type=float, default=None)
    parser.add_argument(
        "--kgalagadi-json",
        default="data/raw/snapshot_kgalagadi/SnapshotKgalagadi_S1_v1.0.json.zip",
    )
    parser.add_argument("--kga-train-fraction", type=float, default=0.70)
    parser.add_argument("--kga-val-fraction", type=float, default=0.15)
    parser.add_argument("--kga-test-fraction", type=float, default=0.15)
    parser.add_argument("--skip-kgalagadi", action="store_true")
    return parser


def main(argv=None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)

    manifest_paths = []
    build_info = {
        "generator": "tools/data/build_manifests.py",
        "seed": args.seed,
        "args": {
            "train_size": args.train_size,
            "val_size": args.val_size,
            "val_site_fraction": args.val_site_fraction,
            "max_per_sequence": args.max_per_sequence,
            "night_fraction": args.night_fraction,
            "kga_train_fraction": args.kga_train_fraction,
            "kga_val_fraction": args.kga_val_fraction,
            "kga_test_fraction": args.kga_test_fraction,
        },
        "sources": {},
        "manifests": {},
        "excluded": {},
        "mapping_notes": [],
        "shortfalls": {},
    }

    if not args.skip_serengeti:
        data, meta = load_coco_ct(args.serengeti_bbox_json)
        category_names = {c["id"]: c["name"] for c in data["categories"]}
        anns_by_image: Dict[str, List[dict]] = {}
        for ann in data["annotations"]:
            anns_by_image.setdefault(ann["image_id"], []).append(ann)
        rows = [
            make_serengeti_row(image, anns_by_image.get(image["id"], []), category_names, meta["member_name"])
            for image in data["images"]
        ]
        serengeti_report: dict = {}
        try:
            rows = split_and_sample_serengeti(
                rows,
                seed=args.seed,
                train_size=args.train_size,
                val_size=args.val_size,
                val_site_fraction=args.val_site_fraction,
                max_per_sequence=args.max_per_sequence,
                night_fraction=args.night_fraction,
                report=serengeti_report,
            )
        except ValueError as exc:
            print(f"build_manifests: {exc}")
            return 1
        serengeti_path = out_dir / "serengeti_trainval.jsonl"
        write_jsonl(rows, serengeti_path)
        manifest_paths.append(serengeti_path)
        build_info["sources"]["snapshot_serengeti"] = {**meta, "info": data.get("info", {})}
        build_info["manifests"][serengeti_path.name] = _manifest_stats(rows)
        build_info["excluded"]["snapshot_serengeti"] = serengeti_report.get("excluded", {})

    if not args.skip_kgalagadi:
        kga_data, kga_meta = load_coco_ct(args.kgalagadi_json)
        kga_report: dict = {}
        kga_rows = build_kgalagadi_rows(
            kga_data,
            kga_meta,
            seed=args.seed,
            train_fraction=args.kga_train_fraction,
            val_fraction=args.kga_val_fraction,
            test_fraction=args.kga_test_fraction,
            report=kga_report,
        )
        kga_path = out_dir / "kgalagadi_site_split.jsonl"
        write_jsonl(kga_rows, kga_path)
        manifest_paths.append(kga_path)

        build_info["sources"]["snapshot_kgalagadi"] = {**kga_meta, "info": kga_data.get("info", {})}
        build_info["manifests"][kga_path.name] = _manifest_stats(kga_rows)
        build_info["excluded"]["snapshot_kgalagadi"] = kga_report.get("excluded", {})
        build_info["kga_sites_total"] = kga_report.get("sites_total")
        build_info["kga_sites_covered"] = kga_report.get("sites_covered")
        build_info["kga_site_split_sequences"] = kga_report.get("site_split_sequences", {})
        build_info["shortfalls"] = kga_report.get("shortfalls", {})
        build_info["mapping_notes"] = kga_report.get("mapping_notes", [])

    try:
        split_check.assert_no_leakage(manifest_paths)
    except split_check.SplitLeakageError as exc:
        print(str(exc))
        return 1

    output_sha256 = {}
    for path in manifest_paths:
        output_sha256[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    build_info["output_sha256"] = output_sha256

    build_info_path = out_dir / "build_info.json"
    with build_info_path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(build_info, indent=2, sort_keys=True))
        fh.write("\n")

    print(f"build_manifests: wrote {len(manifest_paths)} manifest(s) to {out_dir}")
    for path in manifest_paths:
        print(f"  {path.name}: {output_sha256[path.name]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
