"""JPEG / WebP baselines on the EVAL-11 protocol (EVAL-14), CPU only.

Writes the same archive layout as ``inference_partition.py`` so that
``tools/evaluate_kgalagadi.py`` scores every method identically:

    <output>/<relative>.png          reconstruction at the original resolution
    <output>/<dir>/data/<stem>       the exact transmitted bytes (full file)
    <output>/decode_log.jsonl        bytes, sizes, encode/decode seconds
    <output>/run_info.json           codec, quality, library versions

``--processing-long-side`` optionally shrinks the frame before coding (and the
reconstruction is restored afterwards), which lets classical codecs reach the
very low rates of the learned codecs.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import PIL
from PIL import Image, features

from utils.image_geometry import (
    PROCESSING_RESAMPLING,
    resize_for_processing,
    restore_original_size,
)

CODECS = {
    # PIL save kwargs per codec; quality is added per run.
    "jpeg": {"format": "JPEG", "optimize": True, "subsampling": 2},  # 4:2:0
    "webp": {"format": "WEBP", "method": 6},
}


def encode(image: Image.Image, codec: str, quality: int) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, quality=int(quality), **CODECS[codec])
    return buffer.getvalue()


def decode(payload: bytes) -> Image.Image:
    with Image.open(io.BytesIO(payload)) as image:
        return image.convert("RGB")


def archived_paths(archive_root) -> set[str]:
    """Relative paths with a complete record in ``<archive>/decode_log.jsonl``.

    A decode run may cover only part of the dev set (``--limit``) or have been
    interrupted; evaluation then scores exactly the archived images.
    """
    log = Path(archive_root) / "decode_log.jsonl"
    if not log.is_file():
        raise FileNotFoundError(f"no decode_log.jsonl in {archive_root}")
    paths = set()
    for line in log.read_text(encoding="utf-8").splitlines():
        try:
            paths.add(json.loads(line)["relative_path"])
        except (json.JSONDecodeError, KeyError):
            continue
    return paths


def load_rows(manifest: str, split: str, dev_list: str | None, limit: int | None) -> list[dict]:
    rows = [json.loads(line) for line in Path(manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [row for row in rows if row.get("split") == split]
    if dev_list:
        wanted = {
            line.strip() for line in Path(dev_list).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        }
        rows = [row for row in rows if row["image_id"] in wanted]
    rows.sort(key=lambda row: row["image_id"])
    return rows[:limit] if limit else rows


def code_one(source: Path, output_root: Path, relative: str, codec: str, quality: int,
             processing_long_side: int | None) -> dict:
    with Image.open(source) as opened:
        original = opened.convert("RGB")
    coded = resize_for_processing(original, processing_long_side)
    started = time.perf_counter()
    payload = encode(coded, codec, quality)
    encoded = time.perf_counter()
    reconstruction = restore_original_size(decode(payload), original.size)
    decoded = time.perf_counter()

    target = output_root / Path(relative).with_suffix(".png")
    stream = target.parent / "data" / target.stem
    stream.parent.mkdir(parents=True, exist_ok=True)
    stream.write_bytes(payload)
    reconstruction.save(target, compress_level=1)
    return {
        "relative_path": relative,
        "bitstream_bytes": len(payload),
        "bpp": len(payload) * 8 / (original.width * original.height),
        "original_size": list(original.size),
        "coded_size": list(coded.size),
        "encode_seconds": encoded - started,
        "decode_seconds": decoded - encoded,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/manifests/kgalagadi_site_split.jsonl")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--dev-list", default="data/manifests/kgalagadi_dev.txt")
    parser.add_argument("--codec", choices=sorted(CODECS), required=True)
    parser.add_argument("--quality", type=int, required=True)
    parser.add_argument("--processing-long-side", type=int, default=None)
    parser.add_argument("--output", required=True, help="archive directory for this codec/quality")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args(argv)
    if args.codec == "webp" and not features.check("webp"):
        parser.error("this Pillow build has no WebP support")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    info = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": args.codec,
        "codec_options": {**CODECS[args.codec], "quality": args.quality},
        "pillow": PIL.__version__,
        "libjpeg_turbo": features.check_feature("libjpeg_turbo"),
        "webp_version": features.version("webp"),
        "protocol": "original_resolution",
        "processing_long_side": args.processing_long_side,
        "resampling": PROCESSING_RESAMPLING,
        "split": args.split,
        "dev_list": args.dev_list,
    }
    (output / "run_info.json").write_text(json.dumps(info, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    log_path = output / "decode_log.jsonl"
    done = set()
    if args.skip_existing and log_path.is_file():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["relative_path"])
            except (json.JSONDecodeError, KeyError):
                continue
    rows = load_rows(args.manifest, args.split, args.dev_list, args.limit)
    total_bytes = total_pixels = 0
    with log_path.open("a", encoding="utf-8", newline="\n") as log:
        for index, row in enumerate(rows, 1):
            relative = row["relative_path"]
            if relative in done:
                continue
            record = code_one(
                Path(args.data_root) / relative, output, relative,
                args.codec, args.quality, args.processing_long_side,
            )
            record["image_id"] = row["image_id"]
            log.write(json.dumps(record, sort_keys=True) + "\n")
            log.flush()
            total_bytes += record["bitstream_bytes"]
            total_pixels += record["original_size"][0] * record["original_size"][1]
            if index % 50 == 0 or index == len(rows):
                print(f"{args.codec} q={args.quality}: {index}/{len(rows)}", flush=True)
    if total_pixels:
        print(f"{args.codec} q={args.quality} side={args.processing_long_side}: "
              f"{total_bytes * 8 / total_pixels:.4f} bpp over {len(rows)} images -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
