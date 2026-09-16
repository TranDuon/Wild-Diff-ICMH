"""Hardening tests for tools/data/download_images.py (Task 3 behaviors).

Uses a local http.server.ThreadingHTTPServer for retries/mirrors/Range, and
in-memory Pillow JPEGs / zipfile archives. The one real-download test is
opt-in via WILD_DATA_NET_TESTS=1.
"""
from __future__ import annotations

import hashlib
import io
import json
import sys
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import download_images as di  # noqa: E402
import split_check  # noqa: E402


# ---------------------------------------------------------------------------
# Local HTTP server fixture
# ---------------------------------------------------------------------------


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # silence
        pass

    def _lookup(self):
        path = self.path.split("?", 1)[0]
        return self.server.files.get(path)

    def do_HEAD(self):
        body = self._lookup()
        if body is None:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        self.server.request_counts[path] = self.server.request_counts.get(path, 0) + 1

        fail_count = self.server.fail_paths.get(path, 0)
        if fail_count > 0 and self.server.fail_hits.get(path, 0) < fail_count:
            self.server.fail_hits[path] = self.server.fail_hits.get(path, 0) + 1
            self.send_response(503)
            self.end_headers()
            return

        if path in self.server.always_404:
            self.send_response(404)
            self.end_headers()
            return

        body = self._lookup()
        if body is None:
            self.send_response(404)
            self.end_headers()
            return

        range_header = self.headers.get("Range")
        if range_header:
            unit, _, rng = range_header.partition("=")
            start_s, _, end_s = rng.partition("-")
            start = int(start_s)
            end = int(end_s) if end_s else len(body) - 1
            chunk = body[start : end + 1]
            self.send_response(206)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(chunk)))
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(body)}")
            self.end_headers()
            self.wfile.write(chunk)
            return

        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def http_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.files = {}
    server.fail_paths = {}
    server.fail_hits = {}
    server.always_404 = set()
    server.request_counts = {}
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join(timeout=5)


def _base_url(server):
    host, port = server.server_address
    return f"http://{host}:{port}"


def _make_jpeg(grayscale=False, exif_datetime=None, size=(16, 16)):
    from PIL import Image

    color = (128, 128, 128) if grayscale else (200, 40, 40)
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    save_kwargs = {}
    if exif_datetime:
        exif = img.getexif()
        exif[36867] = exif_datetime
        save_kwargs["exif"] = exif.tobytes()
    img.save(buf, format="JPEG", **save_kwargs)
    return buf.getvalue()


def _row(image_id="SER:img1", source="snapshot_serengeti", source_file_name="a.jpg", split="train"):
    return {
        "image_id": image_id,
        "source": source,
        "source_version": "v1",
        "license": "CDLA-Permissive-1.0",
        "relative_path": f"{source}/{source_file_name}",
        "source_file_name": source_file_name,
        "site_id": "SER:D01",
        "sequence_id": "SER:SEQ1",
        "frame_num": 1,
        "datetime": "2010-10-08 12:00:00",
        "illumination": "day",
        "illumination_source": "datetime_hour_proxy",
        "species": None,
        "is_empty": None,
        "boxes": [],
        "width": None,
        "height": None,
        "split": split,
        "subset": "bbox_subset",
        "sample_rank": 0,
        "sha256": None,
        "null_reasons": {"species": "x", "is_empty": "x", "sha256": "x", "width": "x", "height": "x"},
    }


# ---------------------------------------------------------------------------
# Retry / mirror-fallback / integrity
# ---------------------------------------------------------------------------


def test_retries_then_succeeds_records_attempts(http_server, tmp_path):
    payload = _make_jpeg()
    server = http_server
    server.files["/a.jpg"] = payload
    server.fail_paths["/a.jpg"] = 2  # first two GETs return 503

    row = _row()
    mirrors = {"snapshot_serengeti": [f"{_base_url(server)}/{{file_name}}"]}
    root = tmp_path / "images"
    record = di.download_row(row, root, mirrors, retries=5, backoff=0)
    assert record["status"] == "downloaded"
    assert record["attempts"] == 3
    assert record["sha256"] == hashlib.sha256(payload).hexdigest()


def test_first_mirror_404_falls_back_to_second_no_retries(http_server, tmp_path):
    payload = _make_jpeg()
    server = http_server
    server.always_404.add("/missing.jpg")
    server.files["/present.jpg"] = payload

    row = _row(source_file_name="x.jpg")
    # Fixed URLs with no {file_name} placeholder: str.format() with an unused
    # kwarg is a no-op, so candidate_urls() returns these two URLs unchanged.
    mirrors = {"snapshot_serengeti": [f"{_base_url(server)}/missing.jpg", f"{_base_url(server)}/present.jpg"]}

    root = tmp_path / "images"
    record = di.download_row(row, root, mirrors, retries=5, backoff=0)

    assert record["status"] == "downloaded"
    assert record["url"].endswith("/present.jpg")
    assert record["attempts"] == 1
    assert server.request_counts.get("/missing.jpg", 0) == 1  # no retries spent on the 404


def test_non_jpeg_payload_fails_no_file_no_record(http_server, tmp_path):
    server = http_server
    server.files["/bad.jpg"] = b"not a jpeg at all"
    row = _row(source_file_name="bad.jpg")
    mirrors = {"snapshot_serengeti": [f"{_base_url(server)}/{{file_name}}"]}
    root = tmp_path / "images"
    record = di.download_row(row, root, mirrors, retries=1, backoff=0)
    assert record["status"] == "failed"
    dest = di.safe_join(root, row["relative_path"])
    assert not dest.exists()
    assert not (dest.parent / f"{dest.name}.part").exists()


def test_payload_above_max_bytes_fails(http_server, tmp_path):
    payload = _make_jpeg(size=(64, 64))
    server = http_server
    server.files["/big.jpg"] = payload
    row = _row(source_file_name="big.jpg")
    mirrors = {"snapshot_serengeti": [f"{_base_url(server)}/{{file_name}}"]}
    root = tmp_path / "images"
    record = di.download_row(row, root, mirrors, retries=1, backoff=0, max_bytes=10)
    assert record["status"] == "failed"
    dest = di.safe_join(root, row["relative_path"])
    assert not dest.exists()


def test_stale_part_file_is_replaced(http_server, tmp_path):
    payload = _make_jpeg()
    server = http_server
    server.files["/a.jpg"] = payload
    row = _row(source_file_name="a.jpg")
    mirrors = {"snapshot_serengeti": [f"{_base_url(server)}/{{file_name}}"]}
    root = tmp_path / "images"

    dest = di.safe_join(root, row["relative_path"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.parent / f"{dest.name}.part"
    part.write_bytes(b"stale garbage")

    record = di.download_row(row, root, mirrors, retries=1, backoff=0)
    assert record["status"] == "downloaded"
    assert dest.read_bytes() == payload
    assert not part.exists()


@pytest.mark.parametrize("bad_path", ["../x.jpg", "/abs.jpg", "C:/x.jpg"])
def test_safe_join_rejects_escapes(tmp_path, bad_path):
    with pytest.raises(ValueError):
        di.safe_join(tmp_path, bad_path)


# ---------------------------------------------------------------------------
# Zip strategy over HttpRangeFile
# ---------------------------------------------------------------------------


def test_zip_strategy_reads_only_selected_member(http_server, tmp_path):
    payload_a = _make_jpeg()
    payload_b = _make_jpeg(grayscale=True)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("KGA/a.jpg", payload_a)
        zf.writestr("KGA/b.jpg", payload_b)
    zip_bytes = buf.getvalue()

    server = http_server
    server.files["/kga.zip"] = zip_bytes

    row = _row(image_id="KGA:img1", source="snapshot_kgalagadi", source_file_name="a.jpg")
    root = tmp_path / "images"
    record = di.download_row(
        row,
        root,
        mirrors={},
        strategies={"snapshot_kgalagadi": "zip"},
        zip_urls={"snapshot_kgalagadi": f"{_base_url(server)}/kga.zip"},
        zip_member_prefixes={"snapshot_kgalagadi": "KGA/"},
        retries=3,
        backoff=0,
    )
    assert record["status"] == "downloaded"
    assert record["sha256"] == hashlib.sha256(payload_a).hexdigest()
    dest = di.safe_join(root, row["relative_path"])
    assert dest.exists()
    assert dest.read_bytes() == payload_a

    b_dest = root / "snapshot_kgalagadi" / "b.jpg"
    assert not b_dest.exists()  # unselected member never written


def test_zip_member_parent_dir_never_escapes_root(tmp_path):
    payload = _make_jpeg()
    zip_path = tmp_path / "evil.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("../../evil/a.jpg", payload)

    row = _row(image_id="KGA:img1", source="snapshot_kgalagadi", source_file_name="a.jpg")
    # The manifest's relative_path is safe even though the archive member name
    # is malicious: the destination always comes from the manifest, not the
    # zip member name.
    root = tmp_path / "images"
    record = di.download_row(
        row,
        root,
        mirrors={},
        strategies={"snapshot_kgalagadi": "zip"},
        zip_locals={"snapshot_kgalagadi": str(zip_path)},
        zip_member_prefixes={"snapshot_kgalagadi": "../../evil/"},
        retries=1,
        backoff=0,
    )
    assert record["status"] == "downloaded"
    dest = di.safe_join(root, row["relative_path"])
    assert dest.exists()
    # nothing was written outside root
    for path in tmp_path.rglob("*"):
        if path.is_file() and path != zip_path:
            assert str(path.resolve()).startswith(str(root.resolve()))


# ---------------------------------------------------------------------------
# Metadata: grayscale / EXIF
# ---------------------------------------------------------------------------


def test_grayscale_and_exif_metadata(http_server, tmp_path):
    gray_payload = _make_jpeg(grayscale=True)
    color_payload = _make_jpeg(grayscale=False, exif_datetime="2018:10:27 07:50:14")
    server = http_server
    server.files["/gray.jpg"] = gray_payload
    server.files["/color.jpg"] = color_payload

    mirrors = {"snapshot_serengeti": [f"{_base_url(server)}/{{file_name}}"]}
    root = tmp_path / "images"

    gray_row = _row(image_id="SER:gray", source_file_name="gray.jpg")
    color_row = _row(image_id="SER:color", source_file_name="color.jpg")

    gray_record = di.download_row(gray_row, root, mirrors, retries=1, backoff=0)
    color_record = di.download_row(color_row, root, mirrors, retries=1, backoff=0)

    assert gray_record["is_grayscale"] is True
    assert color_record["is_grayscale"] is False
    assert color_record["exif_datetime"] == "2018:10:27 07:50:14"
    assert color_record["exif_status"] == "present"


# ---------------------------------------------------------------------------
# run_download: resume / recovered / leakage / dry-run
# ---------------------------------------------------------------------------


def _manifest(tmp_path, rows, name="m.jsonl"):
    path = tmp_path / name
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps({k: v for k, v in row.items() if not k.startswith("_")}, sort_keys=True))
            fh.write("\n")
    return path


def test_second_run_makes_zero_get_requests(http_server, tmp_path):
    payload = _make_jpeg()
    server = http_server
    server.files["/a.jpg"] = payload

    row = _row(source_file_name="a.jpg")
    manifest = _manifest(tmp_path, [row])
    root = tmp_path / "images"

    common = [
        "--manifest", str(manifest),
        "--root", str(root),
        "--mirror-template", f"snapshot_serengeti={_base_url(server)}/{{file_name}}",
        "--retries", "2",
        "--backoff", "0",
        "--workers", "2",
    ]
    rc = di.main(common)
    assert rc == 0
    assert server.request_counts.get("/a.jpg", 0) == 1

    rc2 = di.main(common)
    assert rc2 == 0
    assert server.request_counts.get("/a.jpg", 0) == 1  # unchanged: no new GET


def test_existing_file_no_record_is_recovered(tmp_path):
    payload = _make_jpeg()
    row = _row(source_file_name="a.jpg")
    manifest = _manifest(tmp_path, [row])
    root = tmp_path / "images"
    dest = di.safe_join(root, row["relative_path"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(payload)

    args = di.build_arg_parser().parse_args(
        [
            "--manifest", str(manifest),
            "--root", str(root),
            "--workers", "2",
        ]
    )
    result = di.run_download(args)
    assert result["recovered"] == 1
    assert result["downloaded"] == 0

    checksums = di.load_checksums(root / "_meta" / "checksums.jsonl")
    record = checksums[row["image_id"]]
    assert record["sha256"] == hashlib.sha256(payload).hexdigest()
    assert record["status"] == "recovered"


def test_leaking_manifest_blocks_list_files(http_server, tmp_path):
    payload = _make_jpeg()
    server = http_server
    server.files["/a.jpg"] = payload
    server.files["/b.jpg"] = payload

    row_train = _row(image_id="SER:img1", source_file_name="a.jpg", split="train")
    row_train["site_id"] = "SER:D01"
    row_val = _row(image_id="SER:img2", source_file_name="b.jpg", split="val")
    row_val["site_id"] = "SER:D01"  # shared site -> leak

    manifest = _manifest(tmp_path, [row_train, row_val])
    root = tmp_path / "images"

    args = di.build_arg_parser().parse_args(
        [
            "--manifest", str(manifest),
            "--root", str(root),
            "--mirror-template", f"snapshot_serengeti={_base_url(server)}/{{file_name}}",
            "--retries", "1",
            "--backoff", "0",
        ]
    )
    with pytest.raises(split_check.SplitLeakageError):
        di.run_download(args)

    list_dir = root / "_lists"
    assert not list_dir.exists() or not any(list_dir.iterdir())

    rc = di.main(
        [
            "--manifest", str(manifest),
            "--root", str(root),
            "--mirror-template", f"snapshot_serengeti={_base_url(server)}/{{file_name}}",
            "--retries", "1",
            "--backoff", "0",
        ]
    )
    assert rc != 0


def test_dry_run_creates_nothing(tmp_path):
    row = _row(source_file_name="a.jpg")
    manifest = _manifest(tmp_path, [row])
    root = tmp_path / "images"
    rc = di.main(["--manifest", str(manifest), "--root", str(root), "--dry-run"])
    assert rc == 0
    assert not root.exists()


# ---------------------------------------------------------------------------
# Opt-in network test
# ---------------------------------------------------------------------------


@pytest.mark.network
def test_real_download_from_committed_manifests(tmp_path):
    manifests_dir = _REPO_ROOT / "data" / "manifests"
    serengeti = manifests_dir / "serengeti_trainval.jsonl"
    kgalagadi = manifests_dir / "kgalagadi_test.jsonl"
    assert serengeti.exists() and kgalagadi.exists()

    ser_rows = sorted(split_check.load_manifest_rows([serengeti]), key=lambda r: (r["split"], r["sample_rank"]))
    kga_rows = sorted(split_check.load_manifest_rows([kgalagadi]), key=lambda r: r["sample_rank"])

    root = tmp_path / "images"
    for row, mirrors in (
        (ser_rows[0], di.DEFAULT_MIRRORS),
        (kga_rows[0], di.DEFAULT_MIRRORS),
    ):
        record = di.download_row(row, root, mirrors, retries=3, backoff=1.0)
        assert record["status"] == "downloaded", record
        dest = di.safe_join(root, row["relative_path"])
        assert dest.exists()
        assert record["sha256"] == hashlib.sha256(dest.read_bytes()).hexdigest()
