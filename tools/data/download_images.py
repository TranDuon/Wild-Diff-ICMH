"""Resumable, mirror-fallback image downloader for the frozen manifests.

Reads one or more ``data/manifests/*.jsonl`` files, fetches each selected
image into ``<root>/<relative_path>``, records a sha256 checksum per image,
and (after the split gate passes) writes one absolute-path ``.list`` file per
(manifest, split) that ``utils.file.load_file_list`` can read directly.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable, Dict, List, Optional

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import split_check  # noqa: E402

# Verified live (see PLAN context): GCS first, since Colab runs inside GCP.
DEFAULT_MIRRORS: Dict[str, List[str]] = {
    "snapshot_serengeti": [
        "https://storage.googleapis.com/public-datasets-lila/snapshotserengeti-unzipped/{file_name}",
        "https://lilawildlife.blob.core.windows.net/lila-wildlife/snapshotserengeti-unzipped/{file_name}",
    ],
}

MAX_BYTES_DEFAULT = 50 * 1024 * 1024


def safe_join(root, relative_path: str) -> Path:
    """Join root and relative_path, rejecting escapes (zip-slip mitigation)."""
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError(f"relative_path must be a non-empty string, got {relative_path!r}")
    normalized = relative_path.replace("\\", "/")
    if normalized.startswith("/"):
        raise ValueError(f"relative_path must not be absolute: {relative_path!r}")
    if len(normalized) >= 2 and normalized[1] == ":":
        raise ValueError(f"relative_path must not include a drive letter: {relative_path!r}")
    if ".." in normalized.split("/"):
        raise ValueError(f"relative_path must not contain '..': {relative_path!r}")

    # Pure lexical join (no filesystem access): root may not exist yet, and
    # concurrent worker threads creating it mid-resolve() have been observed
    # to make Path.resolve() misjudge containment on Windows. The absolute /
    # drive-letter / ".." checks above already rule out escapes, so a plain
    # join is both correct and race-free.
    root_path = Path(root)
    dest = root_path.joinpath(*normalized.split("/"))
    return dest


def candidate_urls(row: dict, mirrors: Dict[str, List[str]]) -> List[str]:
    templates = mirrors.get(row["source"], [])
    quoted = urllib.parse.quote(row["source_file_name"], safe="/")
    return [template.format(file_name=quoted) for template in templates]


def fetch_url(url: str, timeout: float = 60, max_bytes: int = MAX_BYTES_DEFAULT) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "wild-diff-icmh-downloader/1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        chunks: List[bytes] = []
        total = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise ValueError(f"response from {url} exceeded max_bytes={max_bytes}")
            chunks.append(chunk)
        return b"".join(chunks)


def download_row(row: dict, root, mirrors: Dict[str, List[str]], fetch: Callable = fetch_url) -> dict:
    dest = safe_join(root, row["relative_path"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    part_path = dest.parent / f"{dest.name}.part"

    urls = candidate_urls(row, mirrors)
    last_error: Optional[Exception] = None
    for url in urls:
        try:
            payload = fetch(url)
        except Exception as exc:  # noqa: BLE001 - tracer: try each candidate once
            last_error = exc
            continue
        part_path.write_bytes(payload)
        os.replace(part_path, dest)
        return {
            "image_id": row["image_id"],
            "relative_path": row["relative_path"],
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
            "url": url,
            "status": "downloaded",
        }

    return {
        "image_id": row["image_id"],
        "relative_path": row["relative_path"],
        "sha256": None,
        "bytes": 0,
        "url": None,
        "status": "failed",
        "error": str(last_error) if last_error is not None else "no candidate urls",
    }


def load_checksums(path) -> Dict[str, dict]:
    path = Path(path)
    records: Dict[str, dict] = {}
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            records[record["image_id"]] = record
    return records


def write_list_files(manifest_paths, rows: List[dict], root, checksums: Dict[str, dict], list_dir) -> List[Path]:
    """Write <list_dir>/<manifest-stem>.<split>.list, gated by the split check."""
    split_check.assert_no_leakage(manifest_paths)

    list_dir = Path(list_dir)
    list_dir.mkdir(parents=True, exist_ok=True)

    grouped: Dict[tuple, List[dict]] = {}
    for row in rows:
        if row["image_id"] not in checksums:
            continue
        stem = Path(row["_manifest_path"]).stem
        grouped.setdefault((stem, row["split"]), []).append(row)

    written: List[Path] = []
    for (stem, split), split_rows in grouped.items():
        list_path = list_dir / f"{stem}.{split}.list"
        with list_path.open("w", encoding="utf-8", newline="\n") as fh:
            for row in split_rows:
                dest = safe_join(root, row["relative_path"])
                fh.write(str(dest))
                fh.write("\n")
        written.append(list_path)
    return written


def _apply_limit(rows: List[dict], limit: int) -> List[dict]:
    grouped: Dict[tuple, List[dict]] = {}
    for row in rows:
        key = (row.get("_manifest_path"), row.get("split"))
        grouped.setdefault(key, []).append(row)
    result: List[dict] = []
    for group in grouped.values():
        ordered = sorted(group, key=lambda r: r.get("sample_rank") if r.get("sample_rank") is not None else 0)
        result.extend(ordered[:limit])
    return result


def run_download(args) -> dict:
    manifest_paths = [Path(p) for p in args.manifest]
    root = Path(args.root)

    mirrors = {source: list(templates) for source, templates in DEFAULT_MIRRORS.items()}
    for item in getattr(args, "mirror_template", None) or []:
        source, _, template = item.partition("=")
        mirrors[source] = [template]

    rows = split_check.load_manifest_rows(manifest_paths)
    if getattr(args, "splits", None):
        allowed = set(args.splits)
        rows = [r for r in rows if r.get("split") in allowed]
    if getattr(args, "limit", None) is not None:
        rows = _apply_limit(rows, args.limit)

    checksums_path = Path(args.checksums) if args.checksums else root / "_meta" / "checksums.jsonl"
    checksums = load_checksums(checksums_path)

    if args.dry_run:
        for row in rows:
            for url in candidate_urls(row, mirrors):
                print(url)
        return {"downloaded": 0, "skipped": 0, "failed": 0}

    # Create root up front: concurrent workers each resolving/creating
    # subdirectories of a not-yet-existing root can race on Windows and
    # cause Path.resolve() to intermittently misjudge containment.
    root.mkdir(parents=True, exist_ok=True)
    checksums_path.parent.mkdir(parents=True, exist_ok=True)
    counters = {"downloaded": 0, "skipped": 0, "failed": 0}

    def process(row):
        dest = safe_join(root, row["relative_path"])
        record = checksums.get(row["image_id"])
        if record and dest.exists():
            return "skipped", record
        result = download_row(row, root, mirrors)
        return result["status"], result

    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process, row): row for row in rows}
        with checksums_path.open("a", encoding="utf-8", newline="\n") as fh:
            for future in concurrent.futures.as_completed(futures):
                status, record = future.result()
                counters[status] = counters.get(status, 0) + 1
                if status == "downloaded":
                    fh.write(json.dumps(record, sort_keys=True))
                    fh.write("\n")
                done += 1
                if done % 100 == 0:
                    print(f"download_images: progress {done}/{len(rows)}")

    print(f"download_images: summary {counters}")

    checksums = load_checksums(checksums_path)
    list_dir = Path(args.list_dir) if args.list_dir else root / "_lists"
    write_list_files(manifest_paths, rows, root, checksums, list_dir)

    return counters


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", required=True, dest="manifest")
    parser.add_argument("--root", required=True)
    parser.add_argument("--checksums", default=None)
    parser.add_argument("--list-dir", default=None)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--splits", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--mirror-template", action="append", default=[])
    return parser


def main(argv=None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        result = run_download(args)
    except split_check.SplitLeakageError as exc:
        print(str(exc))
        return 1

    return 1 if result.get("failed", 0) else 0


if __name__ == "__main__":
    sys.exit(main())
