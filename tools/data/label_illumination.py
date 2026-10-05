"""Write the light-source day/night sidecar for every image of a manifest.

Uses ``utils.illumination`` (EXIF flash -> IR grayscale -> solar when the
dataset supplies locations -> pixel brightness), so the same command labels
any camera-trap dataset.  The frozen manifest is not modified; consumers call
``utils.illumination.apply_sidecar``.  Resumable: re-running only labels the
image_ids not yet written.

Optional ``--locations`` JSON maps ``site_id`` to
``{"latitude": .., "longitude": .., "utc_offset_hours": ..}``; it is only used
for frames without an EXIF flash tag and without IR grayscale.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.data import split_check  # noqa: E402
from utils.illumination import RULES, classify_file, load_sidecar  # noqa: E402

DEFAULT_SIDECAR = "data/manifests/kgalagadi_illumination.jsonl"


def _capture_time(row):
    try:
        return datetime.strptime(str(row.get("datetime")), "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/manifests/kgalagadi_site_split.jsonl")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--locations", default=None)
    parser.add_argument("--output", default=DEFAULT_SIDECAR)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)

    rows = split_check.assert_no_leakage([Path(args.manifest)])
    locations = json.loads(Path(args.locations).read_text(encoding="utf-8")) if args.locations else {}
    done = load_sidecar(args.output)
    pending = [row for row in rows if row["image_id"] not in done]
    data_root = Path(args.data_root)

    def work(row):
        record = classify_file(
            data_root / row["relative_path"],
            capture_time=_capture_time(row),
            location=locations.get(row.get("site_id")),
        )
        return {"image_id": row["image_id"], "illumination_hour_proxy": row.get("illumination"), **record}

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8", newline="\n") as stream, ThreadPoolExecutor(args.workers) as pool:
        for index, record in enumerate(pool.map(work, pending), 1):
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            done[record["image_id"]] = record
            if index % 500 == 0 or index == len(pending):
                stream.flush()
                print(f"labelled {index}/{len(pending)}", flush=True)

    summary = {
        "images": len(done),
        "rules": RULES,
        "labels": dict(collections.Counter(r["illumination"] for r in done.values())),
        "sources": dict(collections.Counter(r["illumination_source"] for r in done.values())),
        "hour_proxy->label": dict(collections.Counter(
            f"{r.get('illumination_hour_proxy')}->{r['illumination']}" for r in done.values()
        )),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
