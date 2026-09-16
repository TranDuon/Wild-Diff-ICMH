"""End-to-end tracer: synthetic COCO-CT -> manifest -> split gate -> download -> .list."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import build_manifests  # noqa: E402
import download_images  # noqa: E402
import split_check  # noqa: E402
from utils.file import load_file_list  # noqa: E402


def test_pipeline_end_to_end(tmp_path, make_coco_ct, jpeg_mirror):
    bbox_path, _data = make_coco_ct(n_sites=12, seqs_per_site=4, frames_per_seq=3, night_fraction=0.3)

    out_dir = tmp_path / "manifests"
    rc = build_manifests.main(
        [
            "--serengeti-bbox-json",
            str(bbox_path),
            "--out-dir",
            str(out_dir),
            "--seed",
            "20260916",
            "--train-size",
            "20",
            "--val-size",
            "4",
            "--val-site-fraction",
            "0.25",
            "--max-per-sequence",
            "1",
        ]
    )
    assert rc == 0

    manifest_path = out_dir / "serengeti_trainval.jsonl"
    assert manifest_path.exists()

    rc = split_check.main([str(manifest_path)])
    assert rc == 0

    rows = split_check.load_manifest_rows([manifest_path])
    assert len(rows) == 24

    mirror_dir = jpeg_mirror(rows)

    root = tmp_path / "images"
    checksums_path = root / "_meta" / "checksums.jsonl"
    list_dir = root / "_lists"

    download_args = [
        "--manifest",
        str(manifest_path),
        "--root",
        str(root),
        "--checksums",
        str(checksums_path),
        "--list-dir",
        str(list_dir),
        "--mirror-template",
        f"snapshot_serengeti={mirror_dir.as_uri()}/{{file_name}}",
        "--workers",
        "4",
    ]

    rc = download_images.main(download_args)
    assert rc == 0

    checksums = download_images.load_checksums(checksums_path)
    assert len(checksums) == len(rows)
    for row in rows:
        record = checksums[row["image_id"]]
        source_bytes = (mirror_dir / row["source_file_name"]).read_bytes()
        assert record["sha256"] == hashlib.sha256(source_bytes).hexdigest()

    train_list = list_dir / "serengeti_trainval.train.list"
    val_list = list_dir / "serengeti_trainval.val.list"
    assert train_list.exists() and val_list.exists()

    train_paths = load_file_list(str(train_list))
    val_paths = load_file_list(str(val_list))

    train_count = sum(1 for r in rows if r["split"] == "train")
    val_count = sum(1 for r in rows if r["split"] == "val")
    assert len(train_paths) == train_count
    assert len(val_paths) == val_count
    for p in train_paths + val_paths:
        assert Path(p).is_absolute()
        assert Path(p).exists()

    args = download_images.build_arg_parser().parse_args(download_args)
    result = download_images.run_download(args)
    assert result["downloaded"] == 0
    assert result["skipped"] == len(rows)
