"""CPU evidence contract: never infer from unequal or mismatched caches."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tools.review_h1_cached import (
    EXPECTED_PROTOCOL, load_drive_bundle, main, review_bundle, review_loss_logs,
)


def evidence():
    ids = [f"KGA_img_{i:02d}" for i in range(30)]
    box = {"category": "1", "conf": .9, "bbox": [.1, .2, .2, .2]}
    originals = [{"image_id": i, "width": 1000, "height": 500,
                  "detections": [box] if n % 3 else []} for n, i in enumerate(ids)]
    points = {}
    for name, nbytes in (("B0_2", 1000), ("B0_8", 600), ("B0_32", 300),
                         ("H1_500", 700), ("H1_1000", 650)):
        quality = [{"image_id": i, "width": 1000, "height": 500,
                    "coded_size": [512, 256], "bitstream_bytes": nbytes,
                    "bpp": nbytes * 8 / 500000, "rate_source": "coded", "split": "val",
                    "site_id": f"KGA:S{n % 3}", "is_empty": n % 3 == 0,
                    "illumination": "day" if n % 2 else "night",
                    "relative_path": f"images/{i}.jpg", "psnr": 22., "lpips": .5,
                    "ssim": .8, "ms_ssim": .9} for n, i in enumerate(ids)]
        points[name] = {"quality": quality, "pred": copy.deepcopy(originals),
                        "protocol": dict(EXPECTED_PROTOCOL),
                        "protocol_source": {"path": f"cache/{name}/contract.json", "sha256": "a" * 64}}
    return {"ids": ids, "points": points, "originals": originals,
            "provenance": {"originals": {"path": "originals.jsonl", "selected": 30}}}


def test_missing_protocol_closes_gate_but_keeps_descriptive_scores():
    bundle = evidence()
    del bundle["points"]["B0_8"]["protocol"]
    report = review_bundle(bundle)
    assert report["protocol_verified"] is False
    assert report["training_allowed"] is False
    assert report["recommendation"] == "stop_review"
    assert "B0_8" in " ".join(report["missing_evidence"])


@pytest.mark.parametrize("mutation", [
    lambda b: b["ids"].append(b["ids"][0]),
    lambda b: b["points"]["H1_500"]["quality"].pop(),
    lambda b: b["points"]["H1_500"]["quality"].append(b["points"]["H1_500"]["quality"][0]),
    lambda b: b["points"]["H1_500"]["quality"][0].update(split="test"),
    lambda b: b["points"]["H1_500"]["quality"][0].update(width=1024),
    lambda b: b["points"]["H1_500"]["pred"][0].update(height=256),
    lambda b: b["points"]["H1_500"]["quality"][0].update(coded_size=[1024, 512]),
    lambda b: b["points"]["H1_500"]["quality"][0].update(bpp=.99),
    lambda b: b["points"]["H1_500"]["protocol"].update(ddim_steps=5),
])
def test_comparability_errors_fail_closed(mutation):
    bundle = evidence()
    mutation(bundle)
    with pytest.raises(ValueError):
        review_bundle(bundle)


def test_same_subset_recomputes_machine_and_metric_specific_brackets():
    bundle = evidence()
    bundle["points"]["H1_1000"]["pred"][1]["detections"] = []
    report = review_bundle(bundle)
    assert report["points"]["H1_1000"]["machine"]["missed_animal_rate"] == pytest.approx(1 / 20)
    assert report["points"]["B0_2"]["machine"]["map"] == pytest.approx(1.)
    assert report["points"]["H1_1000"]["missed_case_ids"] == [bundle["ids"][1]]
    assert report["rate_comparison"]["H1_1000"]["bracket"] == ["B0_8", "B0_2"]
    assert "B0_8" in report["rate_comparison"]["H1_1000"]["metrics"]["map"]["dominated_by"]
    assert report["rate_comparison"]["H1_1000"]["measured_matched_rate"] is False
    assert report["selection"]["n"] == 30
    assert len(report["evidence_sha256"]) == 64
    assert report["loss_review"]["status"] == "unavailable"


def test_evidence_hash_changes_and_is_deterministic():
    bundle = evidence()
    before = review_bundle(bundle)["evidence_sha256"]
    assert review_bundle(copy.deepcopy(bundle))["evidence_sha256"] == before
    bundle["points"]["H1_500"]["quality"][0]["psnr"] += .1
    assert review_bundle(bundle)["evidence_sha256"] != before


def test_cli_replays_without_model_or_subprocess_inference(tmp_path):
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps(evidence()), encoding="utf-8")
    output = tmp_path / "review.json"
    assert main(["--bundle", str(path), "--output", str(output)]) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["schema_version"] == 1
    assert output.with_suffix(".md").is_file()
    assert report["operations"] == ["cached_json_read", "cpu_aggregation", "cpu_coco_scoring"]
    source = (Path(__file__).parents[1] / "tools/review_h1_cached.py").read_text(encoding="utf-8")
    assert "import torch" not in source


def test_drive_loader_rejects_missing_caches_without_regeneration(tmp_path):
    with pytest.raises(FileNotFoundError, match="dev30"):
        load_drive_bundle(tmp_path)


def test_unsafe_relative_image_paths_are_rejected():
    bundle = evidence()
    bundle["points"]["H1_1000"]["quality"][0]["relative_path"] = "../../secret.jpg"
    with pytest.raises(ValueError, match="relative_path"):
        review_bundle(bundle)


def test_csv_review_keeps_actual_component_names_and_optimizer_lr(tmp_path):
    (tmp_path / "metrics.csv").write_text(
        "global_step,T/l_bpp,T/l_guide,T/l_simple,optimizer/main_lr\n"
        "900,0.3,1.5,0.4,0.0001\n1000,0.2,1.2,0.3,0.0001\n", encoding="utf-8")
    report = review_loss_logs(tmp_path)
    assert report["status"] == "available"
    assert report["scalars"]["T/l_bpp"]["last"] == ["1000", .2]
    assert report["scalars"]["T/l_guide"]["n"] == 2
    assert report["scalars"]["T/l_simple"]["last"][1] == .3
    assert report["scalars"]["optimizer/main_lr"]["last"][1] == .0001
    assert "rate loss component" not in report["missing"]
    assert "actual optimizer learning rates" not in report["missing"]
