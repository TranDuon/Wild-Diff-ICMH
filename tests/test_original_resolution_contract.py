"""Static contracts for the EVAL-11 original-resolution decode/evaluate path."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _main_source():
    source = (ROOT / "inference_partition.py").read_text(encoding="utf-8")
    return source[source.index("def main()"):]


def test_decode_codes_downscaled_frame_and_saves_original_size():
    main = _main_source()
    resize = main.index("resize_for_processing(img, args.processing_long_side)")
    pad = main.index("pad(np.array(coded), scale=64)")
    restore = main.index("restore_original_size(pred_image, img.size)")
    save = main.index("pred_image.save(save_path, compress_level=1)")
    assert resize < pad < restore < save


def test_decode_bpp_uses_original_pixels_not_coded_size():
    main = _main_source()
    assert "bpp = bitstream_bytes * 8.0 / (img.width * img.height)" in main
    assert "coded.width * coded.height" not in main


def test_decode_archives_protocol_timings_and_supports_resume():
    main = _main_source()
    assert "_write_run_info(args, model_config)" in main
    assert "decode_log.jsonl" in main
    assert "'encode_seconds'" in main and "'decode_seconds'" in main
    assert "args.skip_existing and relative_file_path in logged_paths" in main
    assert "zlib.crc32(relative_file_path" in main


def test_evaluator_never_crops_unless_asked():
    source = (ROOT / "tools" / "evaluate_kgalagadi.py").read_text(encoding="utf-8")
    assert "resolve_geometry(args.crop_size, None, manifest_supplied=False)" in source
    assert "resolve_crop_size" not in source


def test_evaluator_reports_eval13_metrics_with_protocol_and_ci():
    source = (ROOT / "tools" / "evaluate_kgalagadi.py").read_text(encoding="utf-8")
    for metric in ("ssim_fullres", "ms_ssim", "dists", "encode_seconds", "decode_seconds"):
        assert f'"{metric}"' in source
    assert "bootstrap_ci(" in source
    assert "stratified_groups(" in source
    assert "protocol=protocol" in source
    assert "apply_sidecar(rows, illumination_sidecar)" in source
