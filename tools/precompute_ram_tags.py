"""Cache RAM++ tags once so codec training does not carry RAM++ in VRAM."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from model.lfgcm import TagGCM


def _rows(path, site_id):
    with open(path, "r", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    return [row for row in rows if site_id is None or row.get("site_id") == site_id]


class _ImageRows(Dataset):
    def __init__(self, rows, data_root):
        self.rows = rows
        self.data_root = Path(data_root)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.data_root / row["relative_path"]) as image:
            value = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        return row, torch.from_numpy(value).permute(2, 0, 1)


def _collate(batch):
    rows, images = zip(*batch)
    images = [
        F.interpolate(
            image.unsqueeze(0), size=(384, 384), mode="bilinear", align_corners=False
        ).squeeze(0)
        for image in images
    ]
    return list(rows), torch.stack(images)


def _completed_ids(output):
    completed = set()
    if not output.is_file():
        return completed
    valid_lines = []
    needs_repair = False
    with output.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                completed.add(json.loads(line)["image_id"])
                valid_lines.append(line.rstrip("\n"))
            except (json.JSONDecodeError, KeyError):
                # A stopped Colab runtime can leave one truncated final line.
                needs_repair = True
    if needs_repair:
        repaired = output.with_name(output.name + ".repair")
        repaired.write_text("\n".join(valid_lines) + "\n", encoding="utf-8")
        os.replace(repaired, output)
        print(f"Removed a truncated JSONL line from {output}")
    return completed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/manifests/kgalagadi_site_split.jsonl")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--site-id", default=None)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args(argv)

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    rows = _rows(args.manifest, args.site_id)
    if not rows:
        parser.error("manifest filter selected no rows")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = _completed_ids(output)
    pending_rows = [row for row in rows if row["image_id"] not in completed]
    if not pending_rows:
        print(f"RAM++ tags already complete: {len(completed)}/{len(rows)} ({output})")
        return 0

    if output.is_file() and output.stat().st_size:
        with output.open("rb+") as stream:
            stream.seek(-1, 2)
            if stream.read(1) != b"\n":
                stream.write(b"\n")

    model = TagGCM(enabled=True, pretrained=args.checkpoint).to(args.device).eval()
    loader = DataLoader(
        _ImageRows(pending_rows, args.data_root),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=_collate,
    )

    with output.open("a", encoding="utf-8", newline="\n") as stream:
        with tqdm(
            total=len(rows),
            initial=len(rows) - len(pending_rows),
            unit="image",
            desc="RAM++ tags",
        ) as progress:
            for batch_rows, images in loader:
                images = images.to(args.device, non_blocking=True)
                with torch.inference_mode():
                    batch_tag_ids, _ = model(images, return_ids=True)

                compact_ids = [
                    [int(value) for value in ids.reshape(-1)] for ids in batch_tag_ids
                ]
                original_ids = [model.expand_tag_ids(ids) for ids in compact_ids]
                index_arrays = [
                    np.asarray(ids, dtype=np.int64).reshape(-1, 1)
                    for ids in original_ids
                ]
                batch_tags = model.model.index2tag(index_arrays)[0]

                for row, flat_ids, tags in zip(batch_rows, compact_ids, batch_tags):
                    stream.write(json.dumps({
                        "image_id": row["image_id"],
                        "site_id": row["site_id"],
                        "relative_path": row["relative_path"],
                        "tags": tags.replace(" |", ","),
                        "tag_ids": flat_ids,
                    }, sort_keys=True) + "\n")
                stream.flush()
                progress.update(len(batch_rows))
    print(f"RAM++ tags ready: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
