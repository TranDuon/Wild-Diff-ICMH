"""Freeze the development set (EVAL-16) from the validation split.

Every tuning decision (lambda, decode knobs, processing resolution, H2 alpha,
H3 prompts) is made on this list; the test split is scored once in Phase 6.

Selection is deterministic: one frame per sequence (frames of a burst are
near-duplicates), every animal and every night sequence (they are scarce), and
the remaining budget filled with day-empty sequences round-robin across sites.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import split_check  # noqa: E402

# Scarce strata are taken whole; day-empty fills the rest of the budget.
PRIORITY_STRATA = (("day", "animal"), ("night", "animal"), ("night", "empty"))
FILL_STRATUM = ("day", "empty")


def _rank(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def _stratum(row: dict) -> tuple[str, str]:
    return str(row.get("illumination") or "unknown"), "empty" if row.get("is_empty") else "animal"


def select_dev_rows(rows: list[dict], *, size: int, seed: int) -> list[dict]:
    sequences: dict[str, list[dict]] = collections.defaultdict(list)
    for row in rows:
        if row.get("split") == "val":
            sequences[row["sequence_id"]].append(row)
    # One deterministic representative frame per sequence.
    representatives = [
        min(frames, key=lambda row: _rank(seed, row["image_id"]))
        for _, frames in sorted(sequences.items())
    ]
    by_stratum: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    for row in representatives:
        by_stratum[_stratum(row)].append(row)

    selected = [row for stratum in PRIORITY_STRATA for row in by_stratum.get(stratum, [])]
    if len(selected) > size:
        raise ValueError(
            f"scarce strata alone hold {len(selected)} sequences; raise --size above that"
        )

    # Round-robin over sites so the large site B06 cannot dominate the fill.
    by_site: dict[str, list[dict]] = collections.defaultdict(list)
    for row in by_stratum.get(FILL_STRATUM, []):
        by_site[row["site_id"]].append(row)
    queues = {
        site: sorted(site_rows, key=lambda row: _rank(seed, row["image_id"]))
        for site, site_rows in sorted(by_site.items())
    }
    while len(selected) < size and any(queues.values()):
        for site in sorted(queues):
            if queues[site] and len(selected) < size:
                selected.append(queues[site].pop(0))
    return sorted(selected, key=lambda row: row["image_id"])


def describe(selected: list[dict]) -> dict:
    return {
        "images": len(selected),
        "sequences": len({row["sequence_id"] for row in selected}),
        "sites": len({row["site_id"] for row in selected}),
        "strata": {
            f"{illumination}/{content}": count
            for (illumination, content), count in sorted(
                collections.Counter(_stratum(row) for row in selected).items()
            )
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/manifests/kgalagadi_site_split.jsonl")
    parser.add_argument("--build-info", default="data/manifests/build_info.json")
    parser.add_argument("--size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--output", default="data/manifests/kgalagadi_dev.txt")
    args = parser.parse_args(argv)

    rows = split_check.assert_no_leakage([Path(args.manifest)], build_info=args.build_info)
    selected = select_dev_rows(rows, size=args.size, seed=args.seed)
    stats = describe(selected)
    header = [
        "# Wild-Diff-ICMH frozen development set (EVAL-16). Do not edit by hand.",
        f"# generator: tools/data/build_dev_set.py --size {args.size} --seed {args.seed}",
        f"# source: {Path(args.manifest).as_posix()} split=val, one frame per sequence",
        f"# stats: {json.dumps(stats, sort_keys=True)}",
        "# strata use the manifest capture-hour illumination; evaluation groups by illumination_ir when present",
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join([*header, *(row["image_id"] for row in selected)]) + "\n",
        encoding="utf-8", newline="\n",
    )
    print(json.dumps(stats, indent=2, sort_keys=True))
    print(f"Wrote {len(selected)} image ids to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
