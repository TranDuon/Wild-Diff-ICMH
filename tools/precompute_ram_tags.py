"""Cache RAM++ tags once so codec training does not carry RAM++ in VRAM."""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

# When this file is launched directly (``python tools/precompute_ram_tags.py``),
# Python puts ``tools/`` rather than the repository root on ``sys.path``.  The
# notebook deliberately invokes it that way, so make the project's top-level
# packages (notably ``model``) discoverable independently of the caller's cwd.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/manifests/kgalagadi_site_split.jsonl")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--site-id", default=None)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser


# Show CLI help without importing the multi-gigabyte ML stack.  This also gives
# the direct-script regression a dependency-independent bootstrap path.
if __name__ == "__main__" and any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
    _build_parser().parse_args()

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

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


def _validate_inputs(rows, data_root, checkpoint):
    """Fail before model allocation with a useful, bounded diagnostic."""
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"RAM++ checkpoint not found: {checkpoint}")
    if checkpoint.stat().st_size < 1024 * 1024:
        raise RuntimeError(
            f"RAM++ checkpoint looks incomplete ({checkpoint.stat().st_size} bytes): "
            f"{checkpoint}"
        )

    data_root = Path(data_root)
    missing = [
        data_root / row["relative_path"]
        for row in rows
        if not (data_root / row["relative_path"]).is_file()
    ]
    if missing:
        sample = "\n".join(f"  - {path}" for path in missing[:5])
        raise FileNotFoundError(
            f"Missing {len(missing)}/{len(rows)} selected images under {data_root}. "
            f"First missing paths:\n{sample}"
        )


def main(argv=None):
    parser = _build_parser()
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

    print(
        f"[RAM tags 1/4] Preflight: {len(pending_rows)} pending images, "
        f"batch={args.batch_size}, workers={args.num_workers}",
        flush=True,
    )
    _validate_inputs(pending_rows, args.data_root, args.checkpoint)
    print(
        f"[RAM tags 2/4] Loading checkpoint: {args.checkpoint}",
        flush=True,
    )
    # Keep the heavyweight project import after argparse and input validation.
    # This makes ``--help`` useful even in a partially installed environment,
    # while REPO_ROOT above makes the import reliable for direct script runs.
    from model.lfgcm import TagGCM

    model = TagGCM(enabled=True, pretrained=args.checkpoint).to(args.device).eval()
    print(f"[RAM tags 3/4] Model ready on {args.device}", flush=True)
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
        print(f"[RAM tags 4/4] Writing resumable output: {output}", flush=True)
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
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        print("\nRAM++ TAGGING FAILED — full traceback follows:", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
