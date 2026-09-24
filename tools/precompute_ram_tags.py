"""Cache RAM++ tags once so codec training does not carry RAM++ in VRAM."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from model.lfgcm import TagGCM


def _rows(path, site_id):
    with open(path, "r", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    return [row for row in rows if site_id is None or row.get("site_id") == site_id]


def _tensor(path, device):
    image = Image.open(path).convert("RGB")
    value = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(value).permute(2, 0, 1).unsqueeze(0).to(device)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/manifests/kgalagadi_site_split.jsonl")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--site-id", default=None)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    args = parser.parse_args(argv)

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    rows = _rows(args.manifest, args.site_id)
    if not rows:
        parser.error("manifest filter selected no rows")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if output.is_file():
        with output.open("r", encoding="utf-8") as stream:
            completed = {json.loads(line)["image_id"] for line in stream if line.strip()}

    model = TagGCM(enabled=True, pretrained=args.checkpoint).to(args.device).eval()
    data_root = Path(args.data_root)
    with output.open("a", encoding="utf-8", newline="\n") as stream:
        for position, row in enumerate(rows, 1):
            if row["image_id"] in completed:
                continue
            image = _tensor(data_root / row["relative_path"], args.device)
            with torch.inference_mode():
                tag_ids, _ = model(image, return_ids=True)
                flat_ids = [int(value) for value in tag_ids[0].reshape(-1)]
                tags = model.model.index2tag([np.asarray(flat_ids).reshape(-1, 1)])[0][0]
                tags = tags.replace(" |", ",")
            stream.write(json.dumps({
                "image_id": row["image_id"],
                "site_id": row["site_id"],
                "relative_path": row["relative_path"],
                "tags": tags,
                "tag_ids": flat_ids,
            }, sort_keys=True) + "\n")
            stream.flush()
            if position % 100 == 0:
                print(f"{position}/{len(rows)}")
    print(f"RAM++ tags ready: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
