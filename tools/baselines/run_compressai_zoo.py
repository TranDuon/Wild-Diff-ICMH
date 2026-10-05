"""Pretrained CompressAI codecs on the EVAL-11 protocol (EVAL-14), inference only.

Real entropy coding: ``model.compress`` produces the strings that are written
to ``data/<stem>``, and the reconstruction comes from ``model.decompress`` of
those bytes, so bpp counts every transmitted byte (shape header included).
Archive layout is identical to ``inference_partition.py`` and
``run_classical.py``.  ``bmshj2018-hyperprior`` is the codec family used by the
comparison candidate Xie et al. 2025, pretrained on generic images.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PIL import Image

from tools.baselines.run_classical import load_rows
from utils.image_geometry import PROCESSING_RESAMPLING, resize_for_processing, restore_original_size

MODELS = ("bmshj2018-hyperprior", "mbt2018", "cheng2020-attn")
MAGIC = b"CAZ1"
PAD = 64  # every zoo model downsamples by at most 64


def pack(strings: list[list[bytes]], shape, padded_size: tuple[int, int]) -> bytes:
    """Serialize compress() output: magic, padded W/H, latent shape, string lists."""
    header = MAGIC + struct.pack(">HHHHB", padded_size[0], padded_size[1], shape[0], shape[1], len(strings))
    body = b""
    for group in strings:
        if len(group) != 1:
            raise ValueError("one image per call is expected")
        body += struct.pack(">I", len(group[0])) + group[0]
    return header + body


def unpack(payload: bytes):
    if payload[:4] != MAGIC:
        raise ValueError("not a CompressAI zoo bitstream")
    width, height, shape_h, shape_w, count = struct.unpack(">HHHHB", payload[4:13])
    offset, strings = 13, []
    for _ in range(count):
        (length,) = struct.unpack(">I", payload[offset:offset + 4])
        offset += 4
        strings.append([payload[offset:offset + length]])
        offset += length
    if offset != len(payload):
        raise ValueError("trailing bytes in bitstream")
    return strings, (shape_h, shape_w), (width, height)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/manifests/kgalagadi_site_split.jsonl")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--dev-list", default="data/manifests/kgalagadi_dev.txt")
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--quality", type=int, required=True, help="CompressAI quality index (1 = lowest rate)")
    parser.add_argument("--metric", default="mse", choices=("mse", "ms-ssim"))
    parser.add_argument("--processing-long-side", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args(argv)

    import compressai
    import numpy as np
    import torch
    import torch.nn.functional as F
    from compressai.zoo import models as zoo

    torch.backends.cudnn.deterministic = True
    compressai.set_entropy_coder("ans")
    net = zoo[args.model](quality=args.quality, metric=args.metric, pretrained=True).eval().to(args.device)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "run_info.json").write_text(json.dumps({
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": f"compressai:{args.model}",
        "quality": args.quality,
        "metric": args.metric,
        "compressai": compressai.__version__,
        "torch": torch.__version__,
        "entropy_coder": "ans",
        "protocol": "original_resolution",
        "processing_long_side": args.processing_long_side,
        "resampling": PROCESSING_RESAMPLING,
        "padding": f"replicate to a multiple of {PAD}",
        "split": args.split,
        "dev_list": args.dev_list,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    log_path = output / "decode_log.jsonl"
    done = set()
    if args.skip_existing and log_path.is_file():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["relative_path"])
            except (json.JSONDecodeError, KeyError):
                continue

    def sync():
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        return time.perf_counter()

    rows = load_rows(args.manifest, args.split, args.dev_list, args.limit)
    with log_path.open("a", encoding="utf-8", newline="\n") as log, torch.no_grad():
        for index, row in enumerate(rows, 1):
            relative = row["relative_path"]
            if relative in done:
                continue
            with Image.open(Path(args.data_root) / relative) as opened:
                original = opened.convert("RGB")
            coded = resize_for_processing(original, args.processing_long_side)
            x = torch.from_numpy(np.asarray(coded, dtype=np.float32) / 255.0).permute(2, 0, 1)[None].to(args.device)
            height, width = x.shape[-2:]
            pad_h, pad_w = (-height) % PAD, (-width) % PAD
            x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

            started = sync()
            compressed = net.compress(x_padded)
            payload = pack(compressed["strings"], compressed["shape"], (width + pad_w, height + pad_h))
            encoded = sync()
            strings, shape, _ = unpack(payload)
            x_hat = net.decompress(strings, shape)["x_hat"][..., :height, :width].clamp(0, 1)
            decoded = sync()

            array = (x_hat[0].permute(1, 2, 0).cpu().numpy() * 255.0).round().astype("uint8")
            reconstruction = restore_original_size(Image.fromarray(array), original.size)
            target = output / Path(relative).with_suffix(".png")
            stream = target.parent / "data" / target.stem
            stream.parent.mkdir(parents=True, exist_ok=True)
            stream.write_bytes(payload)
            reconstruction.save(target)
            log.write(json.dumps({
                "relative_path": relative,
                "image_id": row["image_id"],
                "bitstream_bytes": len(payload),
                "bpp": len(payload) * 8 / (original.width * original.height),
                "original_size": list(original.size),
                "coded_size": list(coded.size),
                "encode_seconds": encoded - started,
                "decode_seconds": decoded - encoded,
            }, sort_keys=True) + "\n")
            log.flush()
            if index % 50 == 0 or index == len(rows):
                print(f"{args.model} q={args.quality}: {index}/{len(rows)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
