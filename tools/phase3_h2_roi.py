"""One explicitly approved, paired ROI guide-loss pilot; never reruns H1/B0.

Preflight/status are CPU-only. Training and dev30 evaluation require the exact
source-bound preflight acknowledgement. No alpha sweep, test or extension API.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import re
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tools import phase3_h1 as old
from tools import phase3_h1_recovery as h1
from tools import review_h1_cached as audit

CONFIG = "configs/train_kgalagadi_h2_roi_pilot.yaml"
CONFIG_CHAIN = (CONFIG, *h1.CONFIG_CHAIN)
METHOD = "H2_roi_v1_alpha2_200"
CONTROL_METHOD = "H1_low_lr_200"
TARGET_STEPS = 200
EXPECTED_LRS = h1.EXPECTED_LRS
CHECKPOINT_IDENTITY_KEY = "wild_diff_h2_roi_contract_sha256"
AUTHOR_RELATIVE = h1.AUTHOR_RELATIVE
# This pilot is explicitly paired to the already-paid-for completed run.
CONTROL_CONTRACT_SHA = "73061ee145b3f2010c692daa43587ccf281edb5a9fedcb1fa8b51a3cdaa1dbe0"
CONTROL_CHECKPOINT_SHA = "85c2eb81be5edd44102fb7bbbe836bcae8a1c4e1082fcef13e8623bd6f93b84f"
AUTHOR_SHA = "e71eaef9e55a33a7051a528acc64c832823bdad10244ebd55b064de75e7972b5"
DEV_SHA = "f1f175c1df398f624ec101565fdd12a382cd618d0b4f22ad7fff411fbe9609e8"
digest = old.digest
run_logged = old.run_logged


def paths(root):
    p = old.paths(root)
    p.update(phase=p["root"] / "phase3/h2_roi_v1_alpha2",
             run=p["root"] / "runs/h2_roi_v1_alpha2_ls512/lambda_2")
    p["preflight"] = p["phase"] / "preflight.json"
    p["overlay_report"] = p["phase"] / "overlays/report.json"
    p["control_run"] = p["root"] / "runs/h1_low_lr_recovery_ls512/lambda_2"
    p["control_eval"] = p["root"] / "phase3/h1_low_lr_recovery/eval/step_000200_quick"
    p["control_archive"] = p["root"] / "phase3/h1_low_lr_recovery/archive/step_000200_quick_ddim50"
    return p


def _contained(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Path escapes the approved root: {path}")
    return path


def _json(path):
    return audit.read_json(path)


def _freeze_json(path, value):
    old.freeze(path, json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def _exact_jsonl(path, selected, label):
    rows, source = audit.selected_jsonl(path, set(selected))
    indexed = audit.index_exact(rows, selected, label)
    return [indexed[i] for i in selected], source


def validate_decode_info(info):
    if (audit.normalize_protocol(info) != audit.EXPECTED_PROTOCOL
            or info.get("split") != "val" or info.get("domain_metadata") is not False
            or info.get("crop_size") is not None or info.get("resampling") != "PIL.BICUBIC"):
        raise RuntimeError("Actual decoder metadata does not match frozen DDIM50 dev30")


def _load_config(path):
    """The existing base-file/list-replacement semantics, without model imports."""
    import yaml
    def merge(base, update):
        value = dict(base)
        for key, item in update.items():
            value[key] = merge(value[key], item) if isinstance(value.get(key), dict) and isinstance(item, dict) else item
        return value
    path = Path(path)
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    base = value.pop("base", None)
    return merge(_load_config(path.parent / base), value) if base else value


def validate_paired_configs():
    """Check consumed config values, not just intended alpha printed in a report."""
    from copy import deepcopy
    control = _load_config(REPO / h1.CONFIG)
    candidate = _load_config(REPO / CONFIG)
    params = candidate["model"]["params"]
    expected = {"roi_loss_enabled": True, "roi_weight": 2.0,
                "roi_background_weight": 1.0, "roi_semantic_enabled": False,
                "l_semantic_weight": 0.0}
    if any(params.get(k) != v for k, v in expected.items()):
        raise RuntimeError("Only ROI-V1 alpha2/background1 guide weighting is approved")
    if candidate.get("preflight", {}).get("asset_splits") != ["train", "val"]:
        raise RuntimeError("H2 asset preflight must require exactly train/val, preserving the full split gate")
    copies = [deepcopy(control), deepcopy(candidate)]
    for config in copies:
        for key in expected:
            config["model"]["params"].pop(key, None)
        config["lightning"]["trainer"].pop("default_root_dir", None)
        config["lightning"].pop("callbacks", None)
        config.get("preflight", {}).pop("asset_splits", None)
    if copies[0] != copies[1]:
        raise RuntimeError("H2 changes more than the paired guide loss/run/diagnostics")
    expected_callbacks = deepcopy(control["lightning"]["callbacks"])
    for item in expected_callbacks:
        if item["target"] == "utils.training_diagnostics.LossCSV":
            item["target"] = "utils.h2_training_diagnostics.H2LossCSV"
    if candidate["lightning"]["callbacks"] != expected_callbacks:
        raise RuntimeError("H2 must retain bounded checkpoint/progress callbacks and its own ROI diagnostics")
    transforms = candidate["data"]["params"]["dataset_overrides"]
    if (transforms.get("processing_long_side") != 512 or transforms.get("bbox_crop_probability") != 0
            or transforms.get("site_id") is not None or candidate["lightning"]["seed"] != 20260916
            or candidate["lightning"]["trainer"]["accumulate_grad_batches"] != 8):
        raise RuntimeError("Paired pooled seed/crop/accumulation protocol changed")
    return expected


def source_evidence(p):
    """Cheap source checks: no checkpoint deserialization or model forward pass."""
    validate_paired_configs()
    selected = old.ids(p["quick"])
    if len(selected) != 30 or len(set(selected)) != 30 or digest(p["quick"]) != DEV_SHA:
        raise RuntimeError("Exactly the approved ordered frozen dev30 is required")
    contract_path = _contained(p["control_run"] / "experiment_contract.json", p["root"])
    contract = _json(contract_path)
    if digest(contract_path) != CONTROL_CONTRACT_SHA:
        raise RuntimeError("Completed H1 control contract differs from the approved run")
    if (contract.get("target_steps") != 200 or contract.get("optimizer_lrs") != EXPECTED_LRS
            or contract.get("author_relative") != AUTHOR_RELATIVE
            or contract.get("author_sha256") != AUTHOR_SHA
            or contract.get("initialization") != "trusted_author_weights_only_fresh_optimizer"):
        raise RuntimeError("H1 control is not the same-source 200-step low-LR experiment")
    chain = {name: digest(REPO / name) for name in h1.CONFIG_CHAIN}
    if contract.get("config_chain_sha256") != chain:
        raise RuntimeError("Protected H1 config chain changed; pairing is invalid")
    protocol = _json(p["quick"].parent / "protocol.json")
    if any(protocol.get(k) != v for k, v in old.PROTOCOL.items()):
        raise RuntimeError("Frozen training/decode protocol changed")
    assets = {"manifest_sha256": digest(REPO / old.MANIFEST),
              "tags_sha256": digest(p["tags"]), "detections_sha256": digest(p["detections"]),
              "illumination_sha256": digest(p["illumination"]),
              "quick_sha256": digest(p["quick"]), "full_sha256": digest(p["full"])}
    if any(protocol.get(k) != v or contract.get("protocol", {}).get(k) != v for k, v in assets.items()):
        raise RuntimeError("Manifest/tags/detections/selection assets changed from H1 control")
    done = _json(p["control_eval"] / "done.json")
    archived = _json(p["control_archive"] / "evaluation_contract.json")
    if (done != archived or done.get("recovery_contract_sha256") != CONTROL_CONTRACT_SHA
            or done.get("checkpoint_step") != 200 or done.get("checkpoint_sha256") != CONTROL_CHECKPOINT_SHA
            or done.get("dev_sha256") != DEV_SHA or done.get("scope") != "quick" or done.get("steps") != 50
            or done.get("review_sha256") != contract.get("review_sha256")
            or done.get("protocol") != old.PROTOCOL):
        raise RuntimeError("Completed H1 evaluation marker/checkpoint/protocol is contradictory")
    observed = _json(p["control_archive"] / "run_info.json")
    validate_decode_info(observed)
    if _json(p["control_eval"] / "run_info.json") != observed:
        raise RuntimeError("H1 cached and archived decoder metadata differ")
    quality, qsrc = _exact_jsonl(p["control_eval"] / "H1.jsonl", selected, "H1 quality")
    predictions, psrc = _exact_jsonl(p["control_eval"] / "H1.detections.jsonl", selected, "H1 predictions")
    comparison = _json(p["control_eval"] / "comparison.json")
    point = comparison.get("points", {}).get(CONTROL_METHOD, {})
    if (comparison.get("protocol_verified") is not True
            or comparison.get("selection", {}).get("image_ids") != selected
            or point.get("quality_records_sha256") != audit.canonical_hash(quality)
            or point.get("prediction_records_sha256") != audit.canonical_hash(predictions)):
        raise RuntimeError("H1 cached per-image evidence differs from its verified comparison")
    # Full train/val detections must include explicit empty records, not missing masks.
    manifest = old.read_jsonl(REPO / old.MANIFEST)
    required = [row for row in manifest if row.get("split") in {"train", "val"}]
    if not required or len({row["image_id"] for row in required}) != len(required):
        raise RuntimeError("Train/val manifest is missing or duplicated")
    ids = [r["image_id"] for r in required]
    det, _ = _exact_jsonl(p["detections"], ids, "train/val ROI coverage")
    tags, _ = _exact_jsonl(p["tags"], ids, "train/val tag coverage")
    for row, record, tag in zip(required, det, tags):
        audit.safe_relative(row["relative_path"])
        dimensions = (record.get("width"), record.get("height"))
        if not all(isinstance(x, int) and x > 0 for x in dimensions):
            raise RuntimeError("ROI source dimensions unavailable")
        audit.validate_detection(record, dimensions, "ROI source")
        if record.get("error") or not isinstance(tag.get("tags"), str):
            raise RuntimeError("Failed detector/tag record cannot become an empty ROI")
    source = {"control_contract_sha256": CONTROL_CONTRACT_SHA,
              "control_checkpoint_sha256": CONTROL_CHECKPOINT_SHA,
              "control_done_sha256": digest(p["control_eval"] / "done.json"),
              "control_quality_sha256": qsrc["selected_records_sha256"],
              "control_predictions_sha256": psrc["selected_records_sha256"],
              "control_run_info_sha256": audit.canonical_hash(observed),
              "assets": assets, "config_chain_sha256": {name: digest(REPO / name) for name in CONFIG_CHAIN},
              "author_sha256": AUTHOR_SHA, "selection": selected}
    return source


def preflight(p):
    from tools.h2_roi_preflight import generate_overlays, select_overlay_cases, stage_assets
    print("H2 CPU preflight: paired control + real mask overlays; no GPU jobs.", flush=True)
    source = source_evidence(p)
    cases = select_overlay_cases(REPO / old.MANIFEST, p["detections"], p["illumination"], limit=10)
    # Helper returns selected manifest rows; only these originals are staged.
    relative = [case["relative_path"] for case in cases]
    print("Overlay assets:", stage_assets(p["root"], p["images"], relative), flush=True)
    overlay = generate_overlays(manifest=REPO / old.MANIFEST, detections=p["detections"], tags=p["tags"],
                illumination=p["illumination"], image_root=p["images"], output_dir=p["phase"] / "overlays",
                source_identity=audit.canonical_hash(source), seed=20260916, limit=10)
    if not p["overlay_report"].is_file():
        raise RuntimeError("Real overlay evidence report was not written")
    value = {"schema_version": 1, "experiment": METHOD, "protocol_verified": True,
             "training_allowed": True, "source": source, "source_sha256": audit.canonical_hash(source),
             "overlay_report_sha256": digest(p["overlay_report"]),
             "overlay_report": overlay, "target_steps": 200,
             "roi": {"foreground_alpha": 2.0, "background_weight": 1.0, "semantic": False},
             "initialization": "trusted_author_weights_only_fresh_optimizer",
             "limitations": "dev30 pilot; different achieved bitrate; detector pseudo-reference; no phase completion"}
    _freeze_json(p["preflight"], value)
    print("Preflight:", p["preflight"], "\nAcknowledge hash:", digest(p["preflight"]), flush=True)
    print("Read actual overlays, then explicitly enable ONE H2 200-step trial. H1 is not rerun.", flush=True)
    return value


def validate_acknowledgement(p, acknowledgement, *, enabled):
    if not enabled:
        raise RuntimeError("H2 heavy actions are off; explicitly enable after viewing real overlays")
    if not re.fullmatch(r"[0-9a-f]{64}", str(acknowledgement)) or not p["preflight"].is_file():
        raise RuntimeError("Run preflight and explicitly acknowledge its SHA256")
    if digest(p["preflight"]) != acknowledgement:
        raise RuntimeError("Acknowledged preflight changed; inspect its new overlays first")
    report = _json(p["preflight"])
    source = source_evidence(p)
    if (report.get("protocol_verified") is not True or report.get("training_allowed") is not True
            or report.get("experiment") != METHOD or report.get("target_steps") != 200
            or report.get("source") != source
            or report.get("source_sha256") != audit.canonical_hash(source)
            or report.get("overlay_report_sha256") != digest(p["overlay_report"])):
        raise RuntimeError("Source-bound H2 preflight or overlay evidence changed")
    # Recheck visual file/source integrity without new image generation.
    from tools.h2_roi_preflight import validate_overlay_report
    validate_overlay_report(p["overlay_report"], source_identity=audit.canonical_hash(source),
                            image_root=p["images"])
    return report


def validate_h2_checkpoint(checkpoint, identity):
    from utils.checkpoint_contract import validate_project_resume_checkpoint
    validate_project_resume_checkpoint(checkpoint)
    if checkpoint.get(CHECKPOINT_IDENTITY_KEY) != identity:
        raise RuntimeError("H2 checkpoint identity mismatch; never adopt an H1/foreign optimizer")
    optimizers = checkpoint["optimizer_states"]
    if len(optimizers) != 1:
        raise RuntimeError("H2 requires one optimizer with two groups")
    groups = optimizers[0].get("param_groups", [])
    if len(groups) != 2 or {g.get("name") for g in groups} != set(EXPECTED_LRS):
        raise RuntimeError("H2 optimizer groups mismatch")
    for group in groups:
        rate = float(group.get("lr", float("nan")))
        if not math.isfinite(rate) or not math.isclose(rate, EXPECTED_LRS[group["name"]], rel_tol=1e-9):
            raise RuntimeError("H2 optimizer learning rates changed")
    step = checkpoint["global_step"]
    if isinstance(step, bool) or not isinstance(step, int) or not 0 <= step <= 200:
        raise RuntimeError("H2 has a fixed 200-step cap")
    return step


def checkpoint_step(path, identity, run):
    path = _contained(path, run)
    print("Checking own H2 checkpoint on CPU:", path, flush=True)
    import torch
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    try:
        return validate_h2_checkpoint(checkpoint, identity)
    finally:
        del checkpoint
        gc.collect()


def _stage(p, scope):
    from tools.h2_roi_preflight import stage_assets
    rows = old.read_jsonl(REPO / old.MANIFEST)
    selected = set(old.ids(p["quick"]))
    rows = [r for r in rows if (r.get("split") in {"train", "val"} if scope == "train" else r["image_id"] in selected)]
    print(f"Stage {scope}: {len(rows)} required originals; skip valid existing files.", flush=True)
    return stage_assets(p["root"], p["images"], [r["relative_path"] for r in rows],
                        include_checkpoints=True, checkpoint_root=p["checkpoints"])


def _require_environment(*, detect_python=None):
    # No pip installation, kernel reset, dependency replacement or model loading.
    import importlib.util
    missing = [m for m in ("torch", "omegaconf", "lightning", "compressai", "ram")
               if importlib.util.find_spec(m) is None]
    if missing:
        raise RuntimeError(f"H2 training environment is not ready: {missing}. Run H2-A/B in the Recovery notebook; do not run H1 Step 5.")
    if detect_python is not None:
        expected = Path("/content/envs/h1-recovery-detect/bin/python")
        if Path(detect_python).resolve() != expected.resolve() or not expected.is_file() or not (expected.parent.parent / ".ready").is_file():
            raise RuntimeError("H2 detector environment is not ready. Run H2-A/C in the Recovery notebook; do not run H1 Step 8 or H1 evaluation.")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Current runtime has no CUDA GPU; keep the existing L4 runtime selected")


def _experiment_contract(p, report, acknowledgement):
    return {"schema_version": 1, "experiment": METHOD, "target_steps": 200,
            "preflight_sha256": acknowledgement, "source": report["source"],
            "source_sha256": report["source_sha256"], "optimizer_lrs": EXPECTED_LRS,
            "roi": report["roi"], "initialization": report["initialization"],
            "author_relative": AUTHOR_RELATIVE, "author_sha256": AUTHOR_SHA,
            "config_chain_sha256": {name: digest(REPO / name) for name in CONFIG_CHAIN}}


def train(p, hours, rate, *, enabled=False, acknowledgement=""):
    seconds = h1._session_seconds(hours)
    report = validate_acknowledgement(p, acknowledgement, enabled=enabled)
    contract = p["run"] / "experiment_contract.json"
    if p["run"].exists() and not contract.exists() and any(p["run"].iterdir()):
        raise RuntimeError("Nonempty H2 run has no contract; preserve it and inspect")
    value = _experiment_contract(p, report, acknowledgement)
    _freeze_json(contract, value)
    _freeze_json(p["run"] / "acknowledged_preflight.json", report)
    identity = digest(contract)
    last = p["run"] / "checkpoints/last.ckpt"
    step = checkpoint_step(last, identity, p["run"]) if last.is_file() else 0
    if step == 200:
        print("H2 already has 200/200 steps; no training/re-copy was launched.", flush=True)
        return
    if not last.exists() and list((p["run"] / "checkpoints").glob("*.ckpt")):
        raise RuntimeError("Own last.ckpt is missing; do not silently restart")
    _require_environment()
    print("Asset reuse:", _stage(p, "train"), flush=True)
    author = _contained(p["checkpoints"] / AUTHOR_RELATIVE, p["checkpoints"])
    if digest(author) != AUTHOR_SHA:
        raise RuntimeError("Authored lambda2 weights do not match the paired H1 source")
    duration = f"00:{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"
    command = [sys.executable, "-u", "train.py", "--config", CONFIG, "--session-time-limit",
               "lightning.trainer.max_steps=200", f"lightning.trainer.max_time={duration}"]
    command.extend(["--resume", str(last)] if last.is_file() else ["--init-checkpoint", str(author)])
    print(f"ONE paired H2: {step} → 200, alpha2/background1 guide-only; same authored weights, not H1 continuation.", flush=True)
    run_logged(command, p, "h2_roi_v1_alpha2_train_200", rate)
    step = checkpoint_step(last, identity, p["run"])
    print(f"H2 checkpoint {step}/200. " + ("Evaluation is a separate explicit action." if step == 200 else "Rerun train to resume ONLY H2."), flush=True)


def paired_report(p, output):
    """Validate exact per-image evidence and recompute CPU metrics; no inference."""
    from tools.eval_machine import summarize, image_flags
    from utils.eval_stats import aggregate
    selected = old.ids(p["quick"])
    source = source_evidence(p)
    h2_info = _json(output / "run_info.json")
    validate_decode_info(h2_info)
    originals, _ = _exact_jsonl(p["detections"], selected, "original reference")
    original = dict(zip(selected, originals))
    metrics, ref = {}, None
    for method, directory, quality_name, prediction_name in (
            (CONTROL_METHOD, p["control_eval"], "H1.jsonl", "H1.detections.jsonl"),
            (METHOD, output, "H2.jsonl", "H2.detections.jsonl")):
        quality, qsrc = _exact_jsonl(directory / quality_name, selected, method + ".quality")
        pred, psrc = _exact_jsonl(directory / prediction_name, selected, method + ".pred")
        if ref is None:
            ref = quality
        for row, base, prediction in zip(quality, ref, pred):
            dimensions = (row.get("width"), row.get("height"))
            if (row.get("split") != "val" or row.get("method") != method
                    or dimensions != (base.get("width"), base.get("height"))
                    or not all(isinstance(n, int) and n > 0 for n in dimensions)
                    or any(row.get(k) != base.get(k) for k in ("relative_path", "site_id", "is_empty", "illumination", "coded_size"))):
                raise RuntimeError("Paired image membership/metadata/geometry differs")
            coded = row.get("coded_size")
            if (not isinstance(coded, list) or len(coded) != 2 or max(coded) != 512
                    or not all(isinstance(n, int) and n > 0 for n in coded)):
                raise RuntimeError("Expected uncropped 512-long-side coded geometry")
            nbytes = row.get("bitstream_bytes")
            if (not isinstance(nbytes, int) or nbytes <= 0 or row.get("rate_source") != "coded"
                    or not math.isclose(float(row.get("bpp", -1)), nbytes * 8 / math.prod(dimensions), rel_tol=1e-8, abs_tol=1e-12)):
                raise RuntimeError("Actual coded rate/original-pixel denominator is invalid")
            if not isinstance(row.get("is_empty"), bool) or row.get("illumination") not in {"day", "night"}:
                raise RuntimeError("Missing human-empty/illumination metadata")
            audit.safe_relative(row["relative_path"])
            audit.validate_detection(original[row["image_id"]], dimensions, "original")
            audit.validate_detection(prediction, dimensions, method)
            for key in audit.QUALITY_METRICS:
                number = row.get(key)
                if number is not None and (not isinstance(number, (float, int)) or not math.isfinite(number)):
                    raise RuntimeError(f"Invalid quality component {key}")
        items = [{"row": row, "gt": original[row["image_id"]], "pred": prediction, "site_id": row["site_id"]}
                 for row, prediction in zip(quality, pred)]
        flags = [image_flags(item, .2) for item in items]
        machine = summarize(items, threshold=.2, gt_threshold=.2)
        machine["denominators"] = {k: sum(f[k] is not None for f in flags) for k in ("empty_fp", "hallucination", "missed_animal")}
        metrics[method] = {"quality": aggregate(quality, audit.QUALITY_METRICS), "machine": machine,
                          "quality_records_sha256": qsrc["selected_records_sha256"],
                          "prediction_records_sha256": psrc["selected_records_sha256"],
                          "missed_case_ids": [i for i, f in zip(selected, flags) if f["missed_animal"]],
                          "hallucination_candidate_ids": [i for i, f in zip(selected, flags) if f["hallucination"]]}
    control, candidate = metrics[CONTROL_METHOD], metrics[METHOD]
    delta = {}
    for section, names in (("quality", ("bpp", "psnr", "lpips", "foreground_ssim")),
                           ("machine", ("map", "missed_animal_rate", "empty_fp_rate", "hallucination_rate"))):
        for name in names:
            a, b = control[section].get(name), candidate[section].get(name)
            delta[name] = b - a if a is not None and b is not None else None
    losses = audit.review_loss_logs(p["run"])
    # ROI fraction is not in the legacy scalar-token set, so read it explicitly.
    import csv
    roi = {k: None for k in ("T/l_guide_roi", "T/l_guide_bg", "T/roi_fraction")}
    loss_file = p["run"] / "loss_components.csv"
    if loss_file.is_file():
        with loss_file.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                if row.get("kind") == "train":
                    for key in roi:
                        raw = row.get(key)
                        if raw:
                            roi[key] = float(raw)
    resources = old.read_jsonl(p["phase"] / "resource_usage.jsonl") if (p["phase"] / "resource_usage.jsonl").exists() else []
    report = {"schema_version": 1, "experiment": METHOD, "protocol_verified": True,
              "source": source, "selection": {"n": 30, "split": "val", "image_ids": selected},
              "protocol": audit.EXPECTED_PROTOCOL, "points": metrics, "delta_h2_minus_h1": delta,
              "bpp_change_percent": 100 * (candidate["quality"]["bpp"] / control["quality"]["bpp"] - 1),
              "loss_review": losses, "roi_logged_last": roi,
              "resources": {"jobs": resources, "cu_estimate": sum(r.get("cu_estimate", 0) for r in resources),
                            "warning": "Job estimates exclude setup/copy/idle time; not a Colab invoice"},
              "recommendation": "review_single_roi_pilot_no_automatic_extension",
              "phase_complete": False,
              "limitations": ["Frozen stratified dev30 is pilot selection evidence, not a population estimate",
                              "Different achieved bitrates; no measured matched-rate/whole-curve superiority",
                              "Detection AP uses original-image MegaDetector pseudo-reference, not human boxes",
                              "Hallucination candidates require manual inspection; frame rates are not counts of lost animals"]}
    old.write_json(output / "comparison.json", report)
    return report


def _stage_ready(output, name, info, selected):
    marker = output / f"{name}.done.json"
    files = {"quality": ("H2.jsonl", "H2.summary.json"), "detect": ("H2.detections.jsonl",)}[name]
    if not marker.exists():
        return False
    value = _json(marker)
    if value.get("identity") != info or value.get("files") != {f: digest(output / f) for f in files}:
        raise RuntimeError(f"Cached H2 {name} marker/files changed")
    _exact_jsonl(output / files[0], selected, name)
    return True


def _mark_stage(output, name, info):
    files = {"quality": ("H2.jsonl", "H2.summary.json"), "detect": ("H2.detections.jsonl",)}[name]
    _freeze_json(output / f"{name}.done.json", {"identity": info, "files": {f: digest(output / f) for f in files}})


def repair_detection_tail(path, selected):
    """Keep complete cached predictions; recover only a demonstrably cut tail.

    Interior corruption and invalid complete lines remain errors. Preserve the
    original bytes beside the file before any atomic repair, never touch H1.
    """
    import hashlib
    import os
    import tempfile
    path = Path(path)
    if not path.is_file():
        return
    payload = path.read_bytes()
    lines = payload.splitlines(keepends=True)
    kept, seen, repaired = [], set(), False
    for index, raw in enumerate(lines):
        if not raw.strip():
            kept.append(raw)
            continue
        terminal = index == len(lines) - 1 and not raw.endswith(b"\n")
        try:
            text = raw.decode("utf-8")
            record = json.loads(text)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            cut = (isinstance(exc, UnicodeDecodeError) and exc.reason == "unexpected end of data") or (
                isinstance(exc, json.JSONDecodeError) and (
                    exc.pos >= len(text.rstrip()) - 1 or exc.msg.startswith("Unterminated string")))
            if not terminal or raw.endswith(b"\r") or not raw.lstrip().startswith(b"{") or not cut:
                raise RuntimeError("Malformed complete/interior H2 detection record; preserve and inspect") from exc
            repaired = True
            break
        if not isinstance(record, dict) or record.get("image_id") not in selected or record["image_id"] in seen:
            raise RuntimeError("Foreign/duplicate H2 detection record; preserve and inspect")
        audit.validate_detection(record, (record.get("width"), record.get("height")), "cached H2")
        seen.add(record["image_id"])
        kept.append(raw if not terminal else raw + b"\n")
        repaired = repaired or terminal
    if not repaired:
        return
    backup = path.with_name(path.name + ".interrupted-" + hashlib.sha256(payload).hexdigest()[:16] + ".bak")
    if backup.exists() and backup.read_bytes() != payload:
        raise RuntimeError("Interrupted detection backup differs; preserve and inspect")
    if not backup.exists():
        backup.write_bytes(payload)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".partial-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(b"".join(kept))
        os.replace(temporary, path)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()
    print("Recovered interrupted H2 detection tail; complete predictions reused. Original:", backup, flush=True)


def evaluate(p, rate, detect_python, *, enabled=False, acknowledgement=""):
    report = validate_acknowledgement(p, acknowledgement, enabled=enabled)
    contract = p["run"] / "experiment_contract.json"
    if not contract.is_file() or _json(contract) != _experiment_contract(p, report, acknowledgement):
        raise RuntimeError("H2 experiment contract is missing/changed")
    identity = digest(contract)
    last = p["run"] / "checkpoints/last.ckpt"
    if checkpoint_step(last, identity, p["run"]) != 200:
        raise RuntimeError("Complete exactly 200 H2 steps before evaluation")
    selected = old.ids(p["quick"])
    output = p["phase"] / "eval/step_000200_quick"
    folder = p["phase"] / "archive/step_000200_quick_ddim50"
    # Freeze immutable snapshot before inference; never decode H1/control again.
    checkpoint, step = old.snapshot(p)
    info = {"protocol": old.PROTOCOL, "scope": "quick", "steps": 50, "checkpoint_step": step,
            "checkpoint_sha256": digest(checkpoint), "dev_sha256": DEV_SHA,
            "h2_contract_sha256": identity, "preflight_sha256": acknowledgement,
            "control_contract_sha256": CONTROL_CONTRACT_SHA}
    _freeze_json(folder / "evaluation_contract.json", info)
    _freeze_json(output / "evaluation_contract.json", info)
    if (output / "done.json").exists():
        done = _json(output / "done.json")
        if done.get("identity") != info or done.get("comparison_sha256") != digest(output / "comparison.json"):
            raise RuntimeError("H2 evaluation completion marker changed")
        if not _stage_ready(output, "quality", info, selected) or not _stage_ready(output, "detect", info, selected):
            raise RuntimeError("H2 done marker exists but a verified cached stage is missing")
        print("H2 evaluation already complete; no GPU/copy/inference:", output / "comparison.json", flush=True)
        return
    _require_environment(detect_python=detect_python)
    print("Evaluation assets (only frozen30):", _stage(p, "evaluate"), flush=True)
    if not h1._archive_ready(folder, selected):
        run_logged([sys.executable, "-u", "inference_partition.py", "--ckpt_sd",
                    p["checkpoints"] / "sd2p1/v2-1_512-ema-pruned.ckpt", "--ckpt_lc", checkpoint,
                    "--config", "configs/model/diffeic.yaml", "--input", p["images"], "--output", folder,
                    "--manifest", old.MANIFEST, "--dev-list", p["quick"], "--tag-cache", p["tags"],
                    "--split", "val", "--sampler", "ddim", "--steps", "50", "--seed", "231",
                    "--device", "cuda", "--processing-long-side", "512", "--skip-existing",
                    "params.control_stage_config.params.control_model_ratio=1.0", "params.c_cfg_scale=3.0"],
                   p, "h2_roi_v1_alpha2_decode_200", rate)
    else:
        print("Reuse complete H2 archive; preserve actual decoder metadata.", flush=True)
    old.assert_archive_complete(folder, selected)
    observed = _json(folder / "run_info.json")
    validate_decode_info(observed)
    if Path(observed.get("checkpoint", "")).resolve() != checkpoint.resolve():
        raise RuntimeError("Actual H2 decoder checkpoint does not match its immutable snapshot")
    old.freeze(output / "run_info.json", (folder / "run_info.json").read_text(encoding="utf-8"))
    common = ["--manifest", old.MANIFEST, "--split", "val", "--dev-list", p["quick"],
              "--illumination-sidecar", p["illumination"]]
    registry = ["--results-registry", p["phase"] / "results.jsonl", "--lambda-rate", "2",
                "--ddim-steps", "50", "--exp-id", "p3_H2_roi_v1_alpha2_200_quick"]
    if not _stage_ready(output, "quality", info, selected):
        run_logged([sys.executable, "-u", "tools/evaluate_kgalagadi.py", *common,
                    "--data-root", p["images"], "--reconstruction-root", folder, "--detections", p["detections"],
                    "--method", METHOD, "--lpips", "--bootstrap-resamples", "0",
                    "--output", output / "H2.jsonl", *registry], p, "h2_roi_v1_alpha2_quality_200", rate)
        _exact_jsonl(output / "H2.jsonl", selected, "H2 quality")
        _mark_stage(output, "quality", info)
    if not _stage_ready(output, "detect", info, selected):
        repair_detection_tail(output / "H2.detections.jsonl", set(selected))
        run_logged([detect_python, "-u", "tools/detect/run_megadetector.py", "--manifest", old.MANIFEST,
                    "--split", "val", "--dev-list", p["quick"], "--image-root", folder,
                    "--suffix", ".png", "--output", output / "H2.detections.jsonl"],
                   p, "h2_roi_v1_alpha2_detect_200", rate)
        _exact_jsonl(output / "H2.detections.jsonl", selected, "H2 predictions")
        _mark_stage(output, "detect", info)
    paired_report(p, output)
    _freeze_json(output / "done.json", {"identity": info, "comparison_sha256": digest(output / "comparison.json")})
    print("ONE H2 paired evaluation complete:", output / "comparison.json", flush=True)


def status(p, rate):
    print("H2 run:", p["run"], "\nPreflight:", p["preflight"], flush=True)
    print("Own checkpoint:", p["run"] / "checkpoints/last.ckpt", flush=True)
    print("Result:", p["phase"] / "eval/step_000200_quick/comparison.json", flush=True)
    if p["preflight"].is_file():
        print("Acknowledgement SHA256:", digest(p["preflight"]), flush=True)
    throughput = p["run"] / "throughput.json"
    if throughput.is_file():
        value = _json(throughput)
        hours = max(0, 200 - int(value["global_step_end"])) * float(value["seconds_per_optimizer_step"]) / 3600
        print(f"Remaining training ~{hours:.2f} hours / ~{hours * rate:.2f} CU (excludes setup/eval)", flush=True)
    result = p["phase"] / "eval/step_000200_quick/comparison.json"
    if result.is_file():
        report = _json(result)
        for method, point in report["points"].items():
            print(method, point["quality"], point["machine"], flush=True)
        print("H2 minus cached H1:", report["delta_h2_minus_h1"], flush=True)
    print("No automatic extensions. Formal Phase3/4 remains incomplete; GPU/visual acceptance is manual.", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "train", "evaluate", "status"))
    parser.add_argument("--drive-root", default="/content/drive/MyDrive/wild_diff_icmh")
    parser.add_argument("--enable", action="store_true")
    parser.add_argument("--acknowledge-preflight", default="")
    parser.add_argument("--session-hours", type=float, default=.5)
    parser.add_argument("--cu-rate", type=float, default=1.54)
    parser.add_argument("--detect-python", default="/content/envs/h1-recovery-detect/bin/python")
    args = parser.parse_args(argv)
    if not math.isfinite(args.cu_rate) or args.cu_rate <= 0:
        parser.error("cu-rate must be positive and finite")
    p = paths(args.drive_root)
    if args.action == "preflight":
        preflight(p)
    elif args.action == "status":
        status(p, args.cu_rate)
    elif args.action == "train":
        train(p, args.session_hours, args.cu_rate, enabled=args.enable, acknowledgement=args.acknowledge_preflight)
    else:
        evaluate(p, args.cu_rate, args.detect_python, enabled=args.enable, acknowledgement=args.acknowledge_preflight)


if __name__ == "__main__":
    main()
