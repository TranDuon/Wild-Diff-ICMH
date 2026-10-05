import json

import numpy as np
from PIL import Image

from tools.baselines.run_classical import code_one, main


def _fixture(tmp_path):
    rng = np.random.default_rng(0)
    pixels = (rng.random((200, 260, 3)) * 255).astype(np.uint8)
    relative = "snapshot_kgalagadi/S/a.jpg"
    source = tmp_path / "images" / relative
    source.parent.mkdir(parents=True)
    Image.fromarray(pixels).save(source, quality=95)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({
        "image_id": "KGA:a", "relative_path": relative, "split": "val",
    }) + "\n", encoding="utf-8")
    return tmp_path / "images", manifest, relative


def test_jpeg_archive_matches_evaluator_layout(tmp_path):
    data_root, _, relative = _fixture(tmp_path)
    record = code_one(data_root / relative, tmp_path / "out", relative, "jpeg", 20, 128)
    png = tmp_path / "out" / "snapshot_kgalagadi" / "S" / "a.png"
    stream = tmp_path / "out" / "snapshot_kgalagadi" / "S" / "data" / "a"
    assert Image.open(png).size == (260, 200)            # restored to the original frame
    assert record["coded_size"] == [128, 98]              # coded at the processing size
    assert stream.stat().st_size == record["bitstream_bytes"]
    assert record["bpp"] == record["bitstream_bytes"] * 8 / (260 * 200)


def test_lower_quality_means_fewer_bytes_and_cli_is_resumable(tmp_path):
    data_root, manifest, relative = _fixture(tmp_path)
    low = code_one(data_root / relative, tmp_path / "low", relative, "webp", 5, None)
    high = code_one(data_root / relative, tmp_path / "high", relative, "webp", 90, None)
    assert low["bitstream_bytes"] < high["bitstream_bytes"]

    arguments = [
        "--manifest", str(manifest), "--data-root", str(data_root), "--dev-list", "",
        "--codec", "jpeg", "--quality", "10", "--output", str(tmp_path / "cli"), "--skip-existing",
    ]
    assert main(arguments) == 0
    assert main(arguments) == 0
    log = (tmp_path / "cli" / "decode_log.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(log) == 1
    assert json.loads((tmp_path / "cli" / "run_info.json").read_text())["method"] == "jpeg"


def test_compressai_bitstream_round_trips_every_byte():
    from tools.baselines.run_compressai_zoo import pack, unpack

    strings = [[b"\x01\x02\x03"], [b""], [b"hyper"]]
    payload = pack(strings, (13, 16), (1024, 832))
    assert len(payload) == 13 + 3 * 4 + 3 + 0 + 5
    assert unpack(payload) == (strings, (13, 16), (1024, 832))
