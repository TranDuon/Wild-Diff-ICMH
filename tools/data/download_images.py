"""Resumable, mirror-fallback image downloader for the frozen manifests.

Reads one or more ``data/manifests/*.jsonl`` files, fetches each selected
image into ``<root>/<relative_path>``, records a sha256 checksum plus basic
metadata (width/height, IR-grayscale flag, EXIF datetime) per image, and
(after the split gate passes) writes one absolute-path ``.list`` file per
(manifest, split) that ``utils.file.load_file_list`` can read directly.

Kgalagadi retrieval strategy (probed live, see PLAN Task 3 Step A): both LILA
mirrors answer per-image HEAD requests with 200 under a ``KGA_public/``
prefix, so the default strategy is per-image URL, not the 10.56 GB season
zip. The zip strategy (``HttpRangeFile`` + range-read ``zipfile``) is kept as
a documented, tested fallback via ``--strategy SOURCE=zip``.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import io
import json
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import split_check  # noqa: E402

# --- Mirror templates and retrieval strategy --------------------------------
# Verified live (see PLAN context / Task 3 probe): GCS first, since Colab
# runs inside GCP; Azure as fallback.
DEFAULT_MIRRORS: Dict[str, List[str]] = {
    "snapshot_serengeti": [
        "https://storage.googleapis.com/public-datasets-lila/snapshotserengeti-unzipped/{file_name}",
        "https://lilawildlife.blob.core.windows.net/lila-wildlife/snapshotserengeti-unzipped/{file_name}",
    ],
    "snapshot_kgalagadi": [
        "https://storage.googleapis.com/public-datasets-lila/snapshot-safari/KGA/KGA_public/{file_name}",
        "https://lilawildlife.blob.core.windows.net/lila-wildlife/snapshot-safari/KGA/KGA_public/{file_name}",
    ],
}

SOURCE_STRATEGY: Dict[str, str] = {
    "snapshot_serengeti": "url",
    "snapshot_kgalagadi": "url",
}

# Only used when --strategy SOURCE=zip overrides the (probed) default above.
DEFAULT_ZIP_URLS: Dict[str, str] = {
    "snapshot_kgalagadi": (
        "https://storage.googleapis.com/public-datasets-lila/snapshot-safari/KGA/KGA_S1.lila.zip"
    ),
}

# The live probe never needed to inspect the zip's central directory (the
# per-image URL strategy already returned 200), so this prefix is a best
# guess, not a verified mapping. Override with --strategy/--zip-* if the zip
# path is ever exercised for real.
KGA_ZIP_MEMBER_PREFIX = ""

MAX_BYTES_DEFAULT = 50 * 1024 * 1024
RETRYABLE_HTTP_STATUS = {429, 500, 502, 503, 504}
NO_RETRY_HTTP_STATUS = {403, 404}
JPEG_SOI = b"\xff\xd8"
USER_AGENT = "wild-diff-icmh-downloader/1"


class RetryableError(Exception):
    """A transient fetch failure that is worth retrying."""


# ---------------------------------------------------------------------------
# Path safety (zip-slip / traversal mitigation)
# ---------------------------------------------------------------------------


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
    # to make Path.resolve() misjudge containment on Windows. The checks
    # above already rule out escapes, so a plain join is correct and race-free.
    root_path = Path(root)
    return root_path.joinpath(*normalized.split("/"))


def candidate_urls(row: dict, mirrors: Dict[str, List[str]]) -> List[str]:
    templates = mirrors.get(row["source"], [])
    quoted = urllib.parse.quote(row["source_file_name"], safe="/")
    return [template.format(file_name=quoted) for template in templates]


# ---------------------------------------------------------------------------
# Retry/backoff HTTP fetch
# ---------------------------------------------------------------------------


def _sleep_backoff(attempt: int, backoff: float) -> None:
    if backoff <= 0:
        return
    delay = backoff * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
    time.sleep(delay)


def fetch_url(
    url: str,
    *,
    timeout: float = 60,
    max_bytes: int = MAX_BYTES_DEFAULT,
    retries: int = 4,
    backoff: float = 2.0,
) -> Tuple[bytes, int]:
    """Fetch url with retry/backoff. Returns (payload, attempts_for_this_url).

    Retries on URLError/timeout/429/5xx and short bodies. On 403/404 raises
    immediately (the caller moves to the next mirror without spending
    retries here).
    """
    last_exc: Optional[Exception] = None
    attempt = 0
    while attempt < retries:
        attempt += 1
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                content_length = response.headers.get("Content-Length")
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
                if content_length is not None and total < int(content_length):
                    raise RetryableError(
                        f"short body from {url}: got {total} bytes, expected {content_length}"
                    )
            return b"".join(chunks), attempt
        except urllib.error.HTTPError as exc:
            if exc.code in NO_RETRY_HTTP_STATUS:
                raise
            last_exc = exc
        except (urllib.error.URLError, TimeoutError, ConnectionError, RetryableError) as exc:
            last_exc = exc
        if attempt < retries:
            _sleep_backoff(attempt, backoff)
    raise last_exc if last_exc is not None else RuntimeError(f"failed to fetch {url}")


# ---------------------------------------------------------------------------
# Zip strategy: range-read a remote (or local) zip, one entry at a time
# ---------------------------------------------------------------------------


class HttpRangeFile(io.RawIOBase):
    """A seekable, read-only file-like object backed by HTTP Range requests.

    Wrap in ``io.BufferedReader`` before handing to ``zipfile.ZipFile`` so
    zipfile's small central-directory reads get coalesced into fewer HTTP
    requests.
    """

    def __init__(self, url: str, *, timeout: float = 60, retries: int = 4, backoff: float = 2.0):
        self._url = url
        self._timeout = timeout
        self._retries = retries
        self._backoff = backoff
        self._pos = 0
        self._size = self._head_size()

    def _head_size(self) -> int:
        request = urllib.request.Request(self._url, method="HEAD", headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=self._timeout) as response:
            length = response.headers.get("Content-Length")
            if length is None:
                raise ValueError(f"HEAD {self._url} did not return a Content-Length header")
            return int(length)

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self._pos = offset
        elif whence == io.SEEK_CUR:
            self._pos += offset
        elif whence == io.SEEK_END:
            self._pos = self._size + offset
        else:
            raise ValueError(f"unsupported whence {whence}")
        return self._pos

    def tell(self) -> int:
        return self._pos

    def readinto(self, b) -> int:  # noqa: D102 - RawIOBase protocol
        length = len(b)
        if length == 0 or self._pos >= self._size:
            return 0
        end = min(self._pos + length, self._size) - 1
        range_header = f"bytes={self._pos}-{end}"

        last_exc: Optional[Exception] = None
        attempt = 0
        while attempt < self._retries:
            attempt += 1
            request = urllib.request.Request(
                self._url,
                headers={"User-Agent": USER_AGENT, "Range": range_header},
            )
            try:
                with urllib.request.urlopen(request, timeout=self._timeout) as response:
                    status = getattr(response, "status", getattr(response, "code", 200))
                    if status != 206:
                        raise ValueError(
                            f"expected 206 Partial Content for a Range request, got {status} "
                            f"from {self._url}"
                        )
                    data = response.read()
                n = len(data)
                b[:n] = data
                self._pos += n
                return n
            except (urllib.error.URLError, ConnectionError) as exc:
                last_exc = exc
                if attempt < self._retries:
                    _sleep_backoff(attempt, self._backoff)
                    continue
                raise
        raise last_exc if last_exc is not None else RuntimeError(f"range read failed: {self._url}")


_ZIP_THREAD_LOCAL = threading.local()


def _get_zip_reader(
    source: str,
    *,
    zip_urls: Dict[str, str],
    zip_locals: Dict[str, str],
    reader_factory=None,
) -> zipfile.ZipFile:
    """One zipfile.ZipFile per (thread, source, location), so workers never
    share a handle, and a source whose zip location changes between calls
    (as in tests exercising both a local zip and a remote one) never reuses
    a stale reader pointed at the wrong archive.
    """
    location = zip_locals.get(source) or zip_urls.get(source)
    cache_key = (source, location)
    cache = getattr(_ZIP_THREAD_LOCAL, "readers", None)
    if cache is None:
        cache = {}
        _ZIP_THREAD_LOCAL.readers = cache
    if cache_key in cache:
        return cache[cache_key]

    if reader_factory is not None:
        zf = reader_factory(source)
    elif source in zip_locals:
        zf = zipfile.ZipFile(zip_locals[source])
    else:
        range_file = HttpRangeFile(zip_urls[source])
        buffered = io.BufferedReader(range_file, buffer_size=1024 * 1024)
        zf = zipfile.ZipFile(buffered)
    cache[cache_key] = zf
    return zf


def _fetch_from_zip(
    row: dict,
    source: str,
    *,
    zip_urls: Dict[str, str],
    zip_locals: Dict[str, str],
    member_prefixes: Dict[str, str],
    reader_factory=None,
) -> Tuple[bytes, str, int]:
    zf = _get_zip_reader(source, zip_urls=zip_urls, zip_locals=zip_locals, reader_factory=reader_factory)
    prefix = member_prefixes.get(source, "")
    member_name = prefix + row["source_file_name"]
    payload = zf.read(member_name)  # never extractall; destination always comes from safe_join
    return payload, f"zip:{member_name}", 1


# ---------------------------------------------------------------------------
# Integrity + metadata
# ---------------------------------------------------------------------------


def _validate_image_bytes(payload: bytes) -> bool:
    if payload[:2] != JPEG_SOI:
        return False
    try:
        from PIL import Image
    except ImportError:
        return True  # SOI check is all we can do without Pillow
    try:
        with Image.open(io.BytesIO(payload)) as img:
            img.verify()
        return True
    except Exception:
        return False


def _extract_image_metadata(payload: bytes) -> dict:
    meta = {
        "width": None,
        "height": None,
        "is_grayscale": None,
        "exif_datetime": None,
        "exif_status": "unreadable",
    }
    try:
        from PIL import Image
    except ImportError:
        return meta
    try:
        with Image.open(io.BytesIO(payload)) as img:
            meta["width"], meta["height"] = img.size

            small = img.convert("RGB").resize((64, 64))
            total = 0.0
            pixels = list(small.getdata())
            for r, g, b in pixels:
                total += max(abs(r - g), abs(g - b), abs(r - b))
            meta["is_grayscale"] = (total / len(pixels)) <= 2.0 if pixels else None

            exif = img.getexif()
            dt = None
            if exif:
                dt = exif.get(36867) or exif.get(306)  # DateTimeOriginal, else DateTime
                if dt is None:
                    try:
                        exif_ifd = exif.get_ifd(0x8769)
                        dt = exif_ifd.get(36867)
                    except Exception:
                        dt = None
            if dt:
                meta["exif_datetime"] = dt
                meta["exif_status"] = "present"
            else:
                meta["exif_status"] = "absent"
    except Exception:
        meta["exif_status"] = "unreadable"
    return meta


# ---------------------------------------------------------------------------
# Per-row download
# ---------------------------------------------------------------------------


def download_row(
    row: dict,
    root,
    mirrors: Dict[str, List[str]],
    *,
    strategies: Optional[Dict[str, str]] = None,
    zip_urls: Optional[Dict[str, str]] = None,
    zip_locals: Optional[Dict[str, str]] = None,
    zip_member_prefixes: Optional[Dict[str, str]] = None,
    retries: int = 4,
    backoff: float = 2.0,
    timeout: float = 60,
    max_bytes: int = MAX_BYTES_DEFAULT,
    fetch: Callable = fetch_url,
    zip_reader_factory=None,
) -> dict:
    dest = safe_join(root, row["relative_path"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    part_path = dest.parent / f"{dest.name}.part"
    if part_path.exists():
        part_path.unlink()  # always overwrite a stale .part

    source = row["source"]
    strategy = (strategies or SOURCE_STRATEGY).get(source, "url")

    payload: Optional[bytes] = None
    winning_url: Optional[str] = None
    attempts = 0
    last_error: Optional[Exception] = None

    if strategy == "zip":
        try:
            payload, winning_url, attempts = _fetch_from_zip(
                row,
                source,
                zip_urls=zip_urls or DEFAULT_ZIP_URLS,
                zip_locals=zip_locals or {},
                member_prefixes=zip_member_prefixes or {},
                reader_factory=zip_reader_factory,
            )
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    else:
        for url in candidate_urls(row, mirrors):
            try:
                payload, attempts = fetch(url, timeout=timeout, max_bytes=max_bytes, retries=retries, backoff=backoff)
                winning_url = url
                break
            except Exception as exc:  # noqa: BLE001 - try next mirror
                last_error = exc
                continue

    base = {
        "image_id": row["image_id"],
        "relative_path": row["relative_path"],
    }

    if payload is None:
        return {
            **base,
            "sha256": None,
            "bytes": 0,
            "url": None,
            "status": "failed",
            "attempts": attempts,
            "error": str(last_error) if last_error is not None else "no candidate source",
        }

    if not _validate_image_bytes(payload):
        return {
            **base,
            "sha256": None,
            "bytes": 0,
            "url": winning_url,
            "status": "failed",
            "attempts": attempts,
            "error": "payload failed image integrity check (bad SOI or Pillow verify)",
        }

    part_path.write_bytes(payload)
    os.replace(part_path, dest)

    metadata = _extract_image_metadata(payload)
    return {
        **base,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "url": winning_url,
        "status": "downloaded",
        "attempts": attempts,
        "width": metadata["width"],
        "height": metadata["height"],
        "is_grayscale": metadata["is_grayscale"],
        "exif_datetime": metadata["exif_datetime"],
        "exif_status": metadata["exif_status"],
    }


# ---------------------------------------------------------------------------
# Checksums / list files
# ---------------------------------------------------------------------------


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


def _parse_kv(values) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for item in values or []:
        key, _, value = item.partition("=")
        result[key] = value
    return result


def _mirror_key(url: str) -> str:
    if url.startswith("zip:"):
        return "zip"
    parsed = urllib.parse.urlparse(url)
    return parsed.netloc or url


def _print_exif_report(rows: List[dict], checksums: Dict[str, dict], n: int) -> None:
    subset = rows[:n]
    have_exif = 0
    have_manifest_dt = 0
    disagree = 0
    checked = 0
    for row in subset:
        record = checksums.get(row["image_id"])
        if not record:
            continue
        checked += 1
        if record.get("exif_datetime"):
            have_exif += 1
        if row.get("datetime"):
            have_manifest_dt += 1
        is_gray = record.get("is_grayscale")
        illumination = row.get("illumination")
        if is_gray is not None and illumination in ("day", "night"):
            proxy_says_night = illumination == "night"
            if bool(is_gray) != proxy_says_night:
                disagree += 1
    print(
        f"download_images: exif-report over first {n} manifest rows ({checked} recorded): "
        f"{have_exif} have exif_datetime, {have_manifest_dt} have a manifest datetime, "
        f"{disagree} is_grayscale/illumination disagreements"
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_download(args) -> dict:
    manifest_paths = [Path(p) for p in args.manifest]
    root = Path(args.root)

    mirrors = {source: list(templates) for source, templates in DEFAULT_MIRRORS.items()}
    for item in getattr(args, "mirror_template", None) or []:
        source, _, template = item.partition("=")
        mirrors[source] = [template]

    strategies = dict(SOURCE_STRATEGY)
    strategies.update(_parse_kv(getattr(args, "strategy", None)))

    zip_urls = dict(DEFAULT_ZIP_URLS)
    zip_urls.update(_parse_kv(getattr(args, "zip_url", None)))
    zip_locals = _parse_kv(getattr(args, "zip_local", None))
    zip_member_prefixes = {"snapshot_kgalagadi": KGA_ZIP_MEMBER_PREFIX}

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
            source = row["source"]
            if strategies.get(source, "url") == "zip":
                prefix = zip_member_prefixes.get(source, "")
                print(f"zip:{prefix}{row['source_file_name']}")
            else:
                for url in candidate_urls(row, mirrors):
                    print(url)
        return {"downloaded": 0, "skipped": 0, "failed": 0, "recovered": 0, "bytes": 0, "by_mirror": {}}

    # Create root up front: concurrent workers each resolving/creating
    # subdirectories of a not-yet-existing root can race; do it once here.
    root.mkdir(parents=True, exist_ok=True)
    checksums_path.parent.mkdir(parents=True, exist_ok=True)

    counters = {"downloaded": 0, "skipped": 0, "failed": 0, "recovered": 0}
    bytes_total = 0
    by_mirror: Dict[str, int] = {}

    def process(row):
        dest = safe_join(root, row["relative_path"])
        record = checksums.get(row["image_id"])

        if record is not None and not args.verify_existing:
            if dest.exists() and dest.stat().st_size == record.get("bytes"):
                return "skipped", record

        if dest.exists():
            payload = dest.read_bytes()
            if _validate_image_bytes(payload):
                digest = hashlib.sha256(payload).hexdigest()
                if record is None:
                    metadata = _extract_image_metadata(payload)
                    new_record = {
                        "image_id": row["image_id"],
                        "relative_path": row["relative_path"],
                        "sha256": digest,
                        "bytes": len(payload),
                        "url": None,
                        "status": "recovered",
                        "attempts": 0,
                        "width": metadata["width"],
                        "height": metadata["height"],
                        "is_grayscale": metadata["is_grayscale"],
                        "exif_datetime": metadata["exif_datetime"],
                        "exif_status": metadata["exif_status"],
                    }
                    return "recovered", new_record
                if digest == record.get("sha256"):
                    return "skipped", record
                # verify-existing found a mismatch -> fall through to re-download

        result = download_row(
            row,
            root,
            mirrors,
            strategies=strategies,
            zip_urls=zip_urls,
            zip_locals=zip_locals,
            zip_member_prefixes=zip_member_prefixes,
            retries=args.retries,
            backoff=args.backoff,
            timeout=args.timeout,
            max_bytes=args.max_bytes,
        )
        return result["status"], result

    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process, row): row for row in rows}
        with checksums_path.open("a", encoding="utf-8", newline="\n") as fh:
            for future in concurrent.futures.as_completed(futures):
                status, record = future.result()
                counters[status] = counters.get(status, 0) + 1
                if status in ("downloaded", "recovered"):
                    bytes_total += record.get("bytes", 0) or 0
                    url = record.get("url")
                    if url:
                        key = _mirror_key(url)
                        by_mirror[key] = by_mirror.get(key, 0) + 1
                    fh.write(json.dumps(record, sort_keys=True))
                    fh.write("\n")
                    fh.flush()
                    os.fsync(fh.fileno())
                done += 1
                if done % 100 == 0:
                    print(f"download_images: progress {done}/{len(rows)}")

    print(f"download_images: summary {counters}, bytes={bytes_total}, by_mirror={by_mirror}")

    checksums = load_checksums(checksums_path)
    list_dir = Path(args.list_dir) if args.list_dir else root / "_lists"
    write_list_files(manifest_paths, rows, root, checksums, list_dir)

    if args.exif_report:
        _print_exif_report(rows, checksums, args.exif_report)

    return {**counters, "bytes": bytes_total, "by_mirror": by_mirror}


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
    parser.add_argument("--mirror-template", action="append", default=[], metavar="SOURCE=TEMPLATE")
    parser.add_argument("--strategy", action="append", default=[], metavar="SOURCE=url|zip")
    parser.add_argument("--zip-url", action="append", default=[], metavar="SOURCE=URL")
    parser.add_argument("--zip-local", action="append", default=[], metavar="SOURCE=PATH")
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--backoff", type=float, default=2.0)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--max-bytes", type=int, default=MAX_BYTES_DEFAULT)
    parser.add_argument("--verify-existing", action="store_true")
    parser.add_argument("--exif-report", type=int, default=100)
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
