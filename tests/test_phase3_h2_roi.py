"""Bounded paired H2 workflow: no GPU models or paid-for control inference."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from tools import phase3_h2_roi as api


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    p = api.paths(tmp_path / "drive")
    p["images"] = tmp_path / "local_images"
    p["checkpoints"] = tmp_path / "local_checkpoints"
    selected = [f"img{i:02d}" for i in range(30)]
    p["quick"].parent.mkdir(parents=True)
    p["quick"].write_text("\n".join(selected) + "\n")
    p["full"].parent.mkdir(parents=True)
    p["full"].write_text("\n".join(selected) + "\n")
    manifest = tmp_path / "manifest.jsonl"
    rows, originals, quality = [], [], []
    for i, image_id in enumerate(selected):
        empty = i >= 20
        relative = f"snapshot_kgalagadi/{image_id}.JPG"
        rows.append({"image_id": image_id, "split": "val", "relative_path": relative,
                     "site_id": "KGA:A01", "is_empty": empty})
        detections = [] if empty else [{"category": "1", "conf": .9, "bbox": [.1, .1, .2, .2]}]
        originals.append({"image_id": image_id, "width": 1000, "height": 1000, "detections": detections})
        quality.append({**rows[-1], "method": api.CONTROL_METHOD, "width": 1000, "height": 1000,
                        "illumination": "day", "coded_size": [512, 512], "bitstream_bytes": 1000,
                        "rate_source": "coded", "bpp": .008, "psnr": 25., "ssim": .8,
                        "ms_ssim": .8, "lpips": .3, "foreground_ssim": None if empty else .7})
    jsonl(manifest, rows)
    monkeypatch.setattr(api.old, "MANIFEST", str(manifest))
    jsonl(p["detections"], originals)
    jsonl(p["tags"], [{"image_id": i, "tags": "animal"} for i in selected])
    jsonl(p["illumination"], [{"image_id": i, "illumination": "day"} for i in selected])
    assets = {"manifest_sha256": api.digest(manifest), "tags_sha256": api.digest(p["tags"]),
              "detections_sha256": api.digest(p["detections"]),
              "illumination_sha256": api.digest(p["illumination"]),
              "quick_sha256": api.digest(p["quick"]), "full_sha256": api.digest(p["full"])}
    monkeypatch.setattr(api, "DEV_SHA", assets["quick_sha256"])
    contract = {"target_steps": 200, "optimizer_lrs": api.EXPECTED_LRS,
                "initialization": "trusted_author_weights_only_fresh_optimizer",
                "author_relative": api.AUTHOR_RELATIVE, "author_sha256": api.AUTHOR_SHA,
                "config_chain_sha256": {name: api.digest(api.REPO / name) for name in api.h1.CONFIG_CHAIN},
                "protocol": {**api.old.PROTOCOL, **assets}, "review_sha256": "a" * 64}
    dump(p["control_run"] / "experiment_contract.json", contract)
    monkeypatch.setattr(api, "CONTROL_CONTRACT_SHA", api.digest(p["control_run"] / "experiment_contract.json"))
    dump(p["quick"].parent / "protocol.json", {**api.old.PROTOCOL, **assets})
    done = {"recovery_contract_sha256": api.CONTROL_CONTRACT_SHA, "checkpoint_step": 200,
            "checkpoint_sha256": api.CONTROL_CHECKPOINT_SHA, "dev_sha256": api.DEV_SHA,
            "scope": "quick", "steps": 50, "protocol": api.old.PROTOCOL, "review_sha256": "a" * 64}
    dump(p["control_eval"] / "done.json", done)
    dump(p["control_archive"] / "evaluation_contract.json", done)
    info = {"processing_long_side": 512, "sampler": "ddim", "steps": 50, "seed": 231,
            "c_cfg_scale": 3., "evaluation_resolution": "original", "split": "val",
            "domain_metadata": False, "crop_size": None, "resampling": "PIL.BICUBIC"}
    dump(p["control_archive"] / "run_info.json", info)
    dump(p["control_eval"] / "run_info.json", info)
    jsonl(p["control_eval"] / "H1.jsonl", quality)
    jsonl(p["control_eval"] / "H1.detections.jsonl", originals)
    dump(p["control_eval"] / "comparison.json", {"protocol_verified": True,
         "selection": {"image_ids": selected}, "points": {api.CONTROL_METHOD: {
             "quality_records_sha256": api.audit.canonical_hash(quality),
             "prediction_records_sha256": api.audit.canonical_hash(originals)}}})
    return p, selected, quality, originals, info


def acknowledged(evidence, monkeypatch):
    p = evidence[0]
    source = api.source_evidence(p)
    dump(p["overlay_report"], {"source_identity": api.audit.canonical_hash(source)})
    report = {"schema_version": 1, "experiment": api.METHOD, "target_steps": 200,
              "protocol_verified": True, "training_allowed": True,
              "source": source, "source_sha256": api.audit.canonical_hash(source),
              "overlay_report_sha256": api.digest(p["overlay_report"]),
              "roi": {"foreground_alpha": 2., "background_weight": 1., "semantic": False},
              "initialization": "trusted_author_weights_only_fresh_optimizer"}
    dump(p["preflight"], report)
    from tools import h2_roi_preflight
    monkeypatch.setattr(h2_roi_preflight, "validate_overlay_report", lambda *a, **k: None)
    return p, report, api.digest(p["preflight"])


def test_config_is_exact_paired_guide_only():
    assert api.validate_paired_configs()["roi_weight"] == 2
    value = api._load_config(api.REPO / api.CONFIG)
    assert value["lightning"]["trainer"]["max_steps"] == 200
    assert value["model"]["resume_checkpoint"] is None
    assert any(c["target"] == "utils.h2_training_diagnostics.H2LossCSV" for c in value["lightning"]["callbacks"])


def test_paths_do_not_adopt_old_h1(tmp_path):
    p = api.paths(tmp_path)
    assert p["run"] != api.h1.paths(tmp_path)["run"]
    assert p["phase"] != api.h1.paths(tmp_path)["phase"]
    assert p["quick"] == api.h1.paths(tmp_path)["quick"]


def test_verified_control_evidence_is_source_bound(evidence):
    p = evidence[0]
    assert api.source_evidence(p)["selection"] == evidence[1]
    value = json.loads((p["control_eval"] / "done.json").read_text())
    value["checkpoint_step"] = 199
    dump(p["control_eval"] / "done.json", value)
    with pytest.raises(RuntimeError, match="marker"):
        api.source_evidence(p)


@pytest.mark.parametrize("name", ["tags", "detections", "illumination"])
def test_changed_asset_rejected(evidence, name):
    p = evidence[0]
    with p[name].open("a") as stream:
        stream.write("\n")
    with pytest.raises(RuntimeError, match="assets changed"):
        api.source_evidence(p)


def test_changed_cached_control_quality_rejected(evidence):
    p, selected, quality, _, _ = evidence
    quality[0]["psnr"] = 9
    jsonl(p["control_eval"] / "H1.jsonl", quality)
    with pytest.raises(RuntimeError, match="per-image evidence"):
        api.source_evidence(p)


def test_ordered_selection_cannot_change(evidence):
    p, selected, *_ = evidence
    p["quick"].write_text("\n".join(reversed(selected)) + "\n")
    with pytest.raises(RuntimeError, match="ordered"):
        api.source_evidence(p)


def test_expensive_actions_off_by_default(tmp_path):
    p = api.paths(tmp_path)
    with pytest.raises(RuntimeError, match="explicitly"):
        api.train(p, .5, 1.54)
    with pytest.raises(RuntimeError, match="explicitly"):
        api.evaluate(p, 1.54, "/wrong/env")


def test_acknowledgement_requires_unchanged_sources_and_overlays(evidence, monkeypatch):
    p, report, ack = acknowledged(evidence, monkeypatch)
    assert api.validate_acknowledgement(p, ack, enabled=True) == report
    dump(p["overlay_report"], {"changed": True})
    with pytest.raises(RuntimeError, match="overlay evidence changed"):
        api.validate_acknowledgement(p, ack, enabled=True)


def test_new_training_uses_author_weights_only_capped_and_reuses_staging(evidence, monkeypatch):
    p, report, ack = acknowledged(evidence, monkeypatch)
    author = p["checkpoints"] / api.AUTHOR_RELATIVE
    author.parent.mkdir(parents=True)
    author.write_bytes(b"trusted author")
    monkeypatch.setattr(api, "AUTHOR_SHA", api.digest(author))
    # Existing report source remains bound; replace only the ack gate fixture.
    monkeypatch.setattr(api, "validate_acknowledgement", lambda *a, **k: report)
    commands, staged = [], []
    monkeypatch.setattr(api, "_require_environment", lambda **k: None)
    monkeypatch.setattr(api, "_stage", lambda p, scope: staged.append(scope))
    monkeypatch.setattr(api, "run_logged", lambda command, *a: commands.append([str(x) for x in command]))
    monkeypatch.setattr(api, "checkpoint_step", lambda *a: 200)
    api.train(p, .5, 1.54, enabled=True, acknowledgement=ack)
    assert staged == ["train"]
    assert "--init-checkpoint" in commands[0] and "--resume" not in commands[0]
    assert "lightning.trainer.max_steps=200" in commands[0]
    assert api.CONFIG in commands[0]
    own = json.loads((p["run"] / "experiment_contract.json").read_text())
    assert own["source"]["control_contract_sha256"] == api.CONTROL_CONTRACT_SHA


def test_nonempty_orphan_run_is_not_adopted(evidence, monkeypatch):
    p, report, ack = acknowledged(evidence, monkeypatch)
    p["run"].mkdir(parents=True)
    (p["run"] / "unbound.ckpt").touch()
    with pytest.raises(RuntimeError, match="Nonempty"):
        api.train(p, .5, 1.54, enabled=True, acknowledgement=ack)


def test_checkpoint_key_rates_and_step_bounds():
    cp = {"wild_diff_checkpoint_contract_version": 2, "global_step": 100,
          api.CHECKPOINT_IDENTITY_KEY: "a" * 64,
          "optimizer_states": [{"param_groups": [{"name": "main", "lr": 1e-5}, {"name": "entropy_aux", "lr": .001}]}]}
    assert api.validate_h2_checkpoint(cp, "a" * 64) == 100
    for mutation in (lambda c: c.update(global_step=201),
                     lambda c: c.update({api.CHECKPOINT_IDENTITY_KEY: "b" * 64}),
                     lambda c: c["optimizer_states"][0]["param_groups"][0].update(lr=1e-4)):
        altered = deepcopy(cp)
        mutation(altered)
        with pytest.raises(RuntimeError):
            api.validate_h2_checkpoint(altered, "a" * 64)


@pytest.mark.parametrize("hours", [0, float("nan"), float("inf"), 3])
def test_invalid_budget_stops_before_staging(tmp_path, monkeypatch, hours):
    monkeypatch.setattr(api, "_stage", lambda *a: pytest.fail("No copying before valid consent/budget"))
    with pytest.raises(ValueError):
        api.train(api.paths(tmp_path), hours, 1.54)


def test_completed_h2_training_does_not_stage_or_run(evidence, monkeypatch):
    p, report, ack = acknowledged(evidence, monkeypatch)
    dump(p["run"] / "experiment_contract.json", api._experiment_contract(p, report, ack))
    last = p["run"] / "checkpoints/last.ckpt"
    last.parent.mkdir(parents=True)
    last.touch()
    monkeypatch.setattr(api, "checkpoint_step", lambda *a: 200)
    monkeypatch.setattr(api, "_stage", lambda *a: pytest.fail("No recopy"))
    monkeypatch.setattr(api, "run_logged", lambda *a: pytest.fail("No retrain"))
    api.train(p, .5, 1.54, enabled=True, acknowledgement=ack)


def test_paired_report_reuses_h1_and_correct_denominators(evidence):
    p, selected, quality, originals, info = evidence
    output = p["phase"] / "eval/step_000200_quick"
    candidate = deepcopy(quality)
    for row in candidate:
        row.update(method=api.METHOD, bitstream_bytes=900, bpp=.0072, psnr=26.)
    predictions = deepcopy(originals)
    predictions[0]["detections"] = []
    jsonl(output / "H2.jsonl", candidate)
    jsonl(output / "H2.detections.jsonl", predictions)
    dump(output / "run_info.json", info)
    report = api.paired_report(p, output)
    assert report["bpp_change_percent"] == pytest.approx(-10)
    assert report["points"][api.METHOD]["machine"]["missed_animal_rate"] == .05
    assert report["points"][api.METHOD]["machine"]["denominators"] == {"empty_fp": 10, "hallucination": 10, "missed_animal": 20}
    assert report["roi_logged_last"]["T/roi_fraction"] is None
    assert report["phase_complete"] is False
    assert report["recommendation"] != "conditional_low_lr_trial"
    candidate[0]["coded_size"] = [256, 256]
    jsonl(output / "H2.jsonl", candidate)
    with pytest.raises(RuntimeError, match="geometry"):
        api.paired_report(p, output)


def test_cache_stage_markers_bind_exact_files(evidence):
    p, selected, quality, originals, _ = evidence
    output = p["phase"] / "eval/step_000200_quick"
    jsonl(output / "H2.detections.jsonl", originals)
    info = {"checkpoint_sha256": "f" * 64}
    assert not api._stage_ready(output, "detect", info, selected)
    api._mark_stage(output, "detect", info)
    assert api._stage_ready(output, "detect", info, selected)
    with (output / "H2.detections.jsonl").open("a") as stream:
        stream.write("\n")
    with pytest.raises(RuntimeError, match="marker/files"):
        api._stage_ready(output, "detect", info, selected)


def test_interrupted_detector_tail_preserves_complete_records_and_backup(evidence):
    p, selected, _, originals, _ = evidence
    path = p["phase"] / "eval/step_000200_quick/H2.detections.jsonl"
    jsonl(path, originals[:2])
    complete = path.read_bytes()
    damaged = complete + b'{"image_id": "img02", "detections": ['
    path.write_bytes(damaged)
    api.repair_detection_tail(path, set(selected))
    assert path.read_bytes() == complete
    assert next(path.parent.glob("*.bak")).read_bytes() == damaged
    api.repair_detection_tail(path, set(selected))
    assert path.read_bytes() == complete


def test_valid_detector_final_record_gets_newline_before_append(evidence):
    p, selected, _, originals, _ = evidence
    path = p["phase"] / "predictions.jsonl"
    jsonl(path, originals[:1])
    path.write_bytes(path.read_bytes().rstrip(b"\n"))
    api.repair_detection_tail(path, set(selected))
    assert path.read_bytes().endswith(b"\n")


def test_detector_interior_corruption_is_not_silently_repaired(evidence):
    p, selected, _, originals, _ = evidence
    path = p["phase"] / "predictions.jsonl"
    jsonl(path, originals[:1])
    damaged = b'{"image_id": bad}\n' + path.read_bytes()
    path.write_bytes(damaged)
    with pytest.raises(RuntimeError, match="interior"):
        api.repair_detection_tail(path, set(selected))
    assert path.read_bytes() == damaged


def test_status_does_not_load_checkpoint_or_stage(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "checkpoint_step", lambda *a: pytest.fail("Cheap status cannot load weights"))
    monkeypatch.setattr(api, "_stage", lambda *a: pytest.fail("Cheap status cannot copy"))
    api.status(api.paths(tmp_path), 1.54)


def test_cli_does_not_accept_sweeps_extensions_or_test():
    for arg in ("test", "full", "--alpha", "--target-steps"):
        with pytest.raises(SystemExit):
            api.main([arg])
