"""Review cached dev30 results on CPU; never decode, train or run a detector.

The evidence hash covers selected per-image records and their source contracts,
not a full-development-set summary. Missing protocol evidence closes the trial
gate but still permits a clearly labelled descriptive audit.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

EXPECTED_PROTOCOL = {"processing_long_side": 512, "sampler": "ddim", "ddim_steps": 50,
                     "decode_seed": 231, "cfg_scale": 3.0, "evaluation_size": "original"}
REQUIRED_POINTS = ("B0_2", "B0_8", "B0_32", "H1_500", "H1_1000")
QUALITY_METRICS = ("psnr", "ssim", "ms_ssim", "lpips", "foreground_ssim")
DIRECTIONS = {"psnr": 1, "lpips": -1, "map": 1, "missed_animal_rate": -1,
              "hallucination_rate": -1}
MAX_FILE_BYTES = 8 * 1024 * 1024
MAX_RECORDS = 20000
SCALAR_TOKENS = ("loss", "lr", "learning_rate", "objective", "l_bpp", "l_guide",
                 "l_simple", "l_semantic", "l_roi")


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def ids_hash(ids):
    return hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest()


def read_text(path):
    path = Path(path)
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError(f"Evidence file exceeds {MAX_FILE_BYTES} bytes: {path}")
    return path.read_text(encoding="utf-8")


def read_json(path):
    return json.loads(read_text(path))


def selected_jsonl(path, ids):
    """Bounded streaming: validate duplicates before filtering a full cache."""
    selected, seen, count = [], set(), 0
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if len(line) > 1024 * 1024:
                raise ValueError(f"Oversized JSONL record: {path}")
            if not line.strip():
                continue
            row = json.loads(line)
            count += 1
            if count > MAX_RECORDS:
                raise ValueError(f"Too many cache records: {path}")
            image_id = row.get("image_id")
            if not isinstance(image_id, str) or image_id in seen:
                raise ValueError(f"Duplicate or invalid image_id in {path}: {image_id}")
            seen.add(image_id)
            if image_id in ids:
                selected.append(row)
    return selected, {"path": str(path), "total": count, "selected": len(selected),
                      "selected_records_sha256": canonical_hash(selected)}


def normalize_protocol(info):
    return {"processing_long_side": info.get("processing_long_side"),
            "sampler": info.get("sampler"), "ddim_steps": info.get("steps"),
            "decode_seed": info.get("seed"), "cfg_scale": info.get("c_cfg_scale"),
            "evaluation_size": info.get("evaluation_resolution")}


def attach_protocol_evidence(bundle, evidence):
    """Attach observed run_info/contracts; never fill absent values with defaults."""
    bundle["protocol_evidence"] = evidence
    phase = evidence.get("phase_protocol", {}).get("value")
    for name, point in bundle["points"].items():
        record = evidence.get("points", {}).get(name)
        if not record:
            continue
        info = record.get("run_info")
        if not info:
            continue
        point["protocol"] = normalize_protocol(info)
        point["protocol_source"] = {
            **record.get("source", {}), "sha256": canonical_hash(record),
            "hash_kind": "canonical_json", "git_commit": info.get("git_commit")}
        if info.get("split") != "val":
            raise ValueError(f"{name}: source run_info split is not val")
        if info.get("domain_metadata") is not False:
            raise ValueError(f"{name}: unexpected domain conditioning")
        if info.get("crop_size") is not None or info.get("resampling") != "PIL.BICUBIC":
            raise ValueError(f"{name}: unexpected evaluation geometry/resampling")
        if name.startswith("H1"):
            contract = record.get("evaluation_contract")
            if not contract or contract.get("scope") != "quick" or contract.get("steps") != 50:
                raise ValueError(f"{name}: missing/mismatched quick evaluation contract")
            expected_step = int(name.split("_")[1]) if name != "H1_low_lr_200" else 200
            if contract.get("checkpoint_step") != expected_step:
                raise ValueError(f"{name}: checkpoint step mismatch")
            if contract.get("dev_sha256") != ids_hash(bundle["ids"]):
                raise ValueError(f"{name}: frozen selection hash mismatch")
            cp = contract.get("protocol", {})
            keys = {"processing_long_side": "processing_long_side", "ddim_steps": "ddim_steps",
                    "decode_seed": "decode_seed", "cfg_scale": "cfg_scale"}
            if any(cp.get(k) != EXPECTED_PROTOCOL[v] for k, v in keys.items()):
                raise ValueError(f"{name}: conflicting evaluation contract protocol")
            if phase and phase.get("quick_sha256") != contract["dev_sha256"]:
                raise ValueError(f"{name}: phase selection hash mismatch")
    return bundle


def load_drive_bundle(root, trial_dir=None):
    """Read known caches from mounted Drive. No network or model inference."""
    root = Path(root)
    selection = root / "phase3/dev30.txt"
    ids = [line.strip() for line in read_text(selection).splitlines()
           if line.strip() and not line.lstrip().startswith("#")]
    bundle = {"ids": ids, "points": {}, "provenance": {},
              "loss_logs_dir": str(root / "runs/h1_pooled_pilot_ls512/lambda_2/lightning_logs")}
    selected = set(ids)
    originals, source = selected_jsonl(root / "phase2/detections/originals.jsonl", selected)
    bundle["originals"] = originals
    bundle["provenance"]["originals"] = source
    evidence = {"phase_protocol": {"value": read_json(root / "phase3/protocol.json"),
                                   "source": str(root / "phase3/protocol.json")}, "points": {}}
    for name in REQUIRED_POINTS:
        if name.startswith("B0"):
            lam = name.split("_")[1]
            stem = f"B0_ls512_ddim50__lambda_{lam}"
            quality = root / "phase2/eval" / f"{stem}.jsonl"
            pred = root / "phase2/detections" / f"{stem}.jsonl"
            archive = root / "phase2/archive/B0_ls512_ddim50" / f"lambda_{lam}"
        else:
            step = int(name.split("_")[1])
            directory = root / "phase3/eval" / f"step_{step:06d}_quick"
            quality, pred = directory / "H1.jsonl", directory / "H1.detections.jsonl"
            archive = root / "phase3/archive" / f"step_{step:06d}_quick_ddim50"
        rows, qsrc = selected_jsonl(quality, selected)
        detections, psrc = selected_jsonl(pred, selected)
        bundle["points"][name] = {"quality": rows, "pred": detections}
        bundle["provenance"][f"{name}_quality"] = qsrc
        bundle["provenance"][f"{name}_pred"] = psrc
        record = {"run_info": read_json(archive / "run_info.json"), "source": {"path": str(archive)}}
        if name.startswith("H1"):
            record["evaluation_contract"] = read_json(archive / "evaluation_contract.json")
        evidence["points"][name] = record
    if trial_dir is not None:
        directory = Path(trial_dir)
        rows, qsrc = selected_jsonl(directory / "H1.jsonl", selected)
        detections, psrc = selected_jsonl(directory / "H1.detections.jsonl", selected)
        bundle["points"]["H1_low_lr_200"] = {"quality": rows, "pred": detections}
        bundle["provenance"]["H1_low_lr_200_quality"] = qsrc
        bundle["provenance"]["H1_low_lr_200_pred"] = psrc
        # Trial helper explicitly stores the archive path in this small manifest.
        source = read_json(directory / "review_source.json")
        archive = Path(source["archive"])
        evidence["points"]["H1_low_lr_200"] = {
            "run_info": read_json(archive / "run_info.json"),
            "evaluation_contract": read_json(archive / "evaluation_contract.json"),
            "source": {"path": str(archive)}}
    return attach_protocol_evidence(bundle, evidence)


def index_exact(records, ids, label):
    if not isinstance(records, list) or len(records) != len(ids):
        raise ValueError(f"{label}: exact dev30 records required")
    lookup = {}
    for row in records:
        if not isinstance(row, dict) or row.get("image_id") in lookup:
            raise ValueError(f"{label}: duplicate/invalid record")
        lookup[row.get("image_id")] = row
    if set(lookup) != set(ids):
        raise ValueError(f"{label}: image ID membership mismatch")
    return lookup


def safe_relative(value):
    if not isinstance(value, str) or "\\" in value or re.match(r"^[A-Za-z]:", value):
        raise ValueError(f"Invalid relative_path: {value}")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(p in {"..", "."} for p in path.parts):
        raise ValueError(f"Invalid relative_path: {value}")
    return path


def validate_detection(record, dimensions, label):
    if (record.get("width"), record.get("height")) != dimensions:
        raise ValueError(f"{label}: detection/original dimensions mismatch")
    if not isinstance(record.get("detections"), list):
        raise ValueError(f"{label}: missing detections list")
    for detection in record["detections"]:
        bbox = detection.get("bbox", [])
        confidence = detection.get("conf")
        if (len(bbox) != 4 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in bbox)
                or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1
                or min(bbox) < 0 or max(bbox) > 1):
            raise ValueError(f"{label}: invalid normalized bbox/confidence")


def validate_bundle(bundle):
    ids = bundle.get("ids")
    if (not isinstance(ids, list) or len(ids) != 30 or len(set(ids)) != 30
            or not all(isinstance(i, str) and i for i in ids)):
        raise ValueError("Need exactly 30 unique frozen validation IDs")
    points = bundle.get("points", {})
    if not set(REQUIRED_POINTS) <= set(points):
        raise ValueError("Missing B0 lambda 2/8/32 or H1 step 500/1000")
    if set(points) - set(REQUIRED_POINTS) - {"H1_low_lr_200"}:
        raise ValueError("Unrecognized comparison point")
    if not bundle.get("provenance"):
        raise ValueError("Missing source provenance")
    originals = index_exact(bundle.get("originals"), ids, "originals")
    reference, indexed, missing = None, {}, []
    for name, point in points.items():
        quality = index_exact(point.get("quality"), ids, f"{name}.quality")
        pred = index_exact(point.get("pred"), ids, f"{name}.pred")
        protocol = point.get("protocol")
        source = point.get("protocol_source", {})
        if protocol is None or not source.get("sha256"):
            missing.append(f"{name}: actual source protocol unavailable")
        elif protocol != EXPECTED_PROTOCOL:
            raise ValueError(f"{name}: mismatched evaluation protocol: {protocol}")
        if reference is None:
            reference = quality
        for image_id in ids:
            row = quality[image_id]
            original = originals[image_id]
            dimensions = (row.get("width"), row.get("height"))
            if (row.get("split") != "val" or not all(isinstance(v, int) and v > 0 for v in dimensions)
                    or dimensions != (reference[image_id].get("width"), reference[image_id].get("height"))):
                raise ValueError(f"{name}/{image_id}: validation split/dimensions mismatch")
            metadata = ("site_id", "is_empty", "illumination", "relative_path")
            safe_relative(row.get("relative_path"))
            if any(row.get(k) != reference[image_id].get(k) for k in metadata):
                raise ValueError(f"{name}/{image_id}: selection metadata mismatch")
            if not isinstance(row.get("is_empty"), bool) or row.get("illumination") not in {"day", "night"}:
                raise ValueError(f"{name}/{image_id}: missing human-empty/illumination metadata")
            coded = row.get("coded_size")
            if (not isinstance(coded, list) or len(coded) != 2 or max(coded) != 512
                    or not all(isinstance(v, int) and v > 0 for v in coded)
                    or abs(coded[0] / coded[1] - dimensions[0] / dimensions[1]) > .01
                    or coded != reference[image_id].get("coded_size")):
                raise ValueError(f"{name}/{image_id}: coded geometry mismatch")
            nbytes = row.get("bitstream_bytes")
            if not isinstance(nbytes, int) or nbytes <= 0 or row.get("rate_source") != "coded":
                raise ValueError(f"{name}/{image_id}: actual coded bytes required")
            expected = nbytes * 8 / (dimensions[0] * dimensions[1])
            if not math.isclose(float(row.get("bpp", -1)), expected, rel_tol=1e-8, abs_tol=1e-12):
                raise ValueError(f"{name}/{image_id}: bpp does not match actual bytes/original pixels")
            for metric in QUALITY_METRICS:
                value = row.get(metric)
                if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value)):
                    raise ValueError(f"{name}/{image_id}: invalid {metric}")
            validate_detection(original, dimensions, "originals")
            validate_detection(pred[image_id], dimensions, name)
        indexed[name] = (quality, pred)
    return ids, originals, indexed, missing


def rate_comparison(points):
    b0 = sorted((points[n]["quality"]["bpp"], n) for n in REQUIRED_POINTS if n.startswith("B0"))
    comparisons = {}
    for name in points:
        if not name.startswith("H1"):
            continue
        rate = points[name]["quality"]["bpp"]
        lower = [(b, n) for b, n in b0 if b <= rate]
        upper = [(b, n) for b, n in b0 if b >= rate]
        bracket = [lower[-1][1], upper[0][1]] if lower and upper else None
        metrics = {}
        for metric, direction in DIRECTIONS.items():
            section = "quality" if metric in QUALITY_METRICS else "machine"
            value = points[name][section].get(metric)
            dominated, interpolation = [], None
            for baseline_rate, baseline in b0:
                other = points[baseline][section].get(metric)
                if (value is not None and other is not None and baseline_rate <= rate
                        and direction * other >= direction * value
                        and (baseline_rate < rate or direction * other > direction * value)):
                    dominated.append(baseline)
            if bracket and bracket[0] != bracket[1]:
                left, right = bracket
                a, b = points[left][section].get(metric), points[right][section].get(metric)
                if a is not None and b is not None:
                    lr, rr = points[left]["quality"]["bpp"], points[right]["quality"]["bpp"]
                    t = (math.log(rate) - math.log(lr)) / (math.log(rr) - math.log(lr))
                    interpolation = a + t * (b - a)
            metrics[metric] = {"h1_value": value, "dominated_by": dominated,
                               "descriptive_b0_log_rate_interpolation": interpolation,
                               "interpolation_is_measured": False}
        comparisons[name] = {"bpp": rate, "bracket": bracket, "metrics": metrics,
                             "measured_matched_rate": False,
                             "warning": "Descriptive bracket/Pareto only; not BD-rate or significance."}
    return comparisons


def review_loss_logs(directory=None):
    """Optional CPU scalar read; no log files are modified or models imported."""
    result = {"status": "unavailable", "sources": [], "scalars": {}, "missing": [],
              "conclusion": "Lower LR is a hypothesis, not a diagnosed cause."}
    if not directory or not Path(directory).is_dir():
        result["missing"] = ["training/validation component losses", "actual optimizer learning rates"]
        return result
    directory = Path(directory)
    files = sorted(directory.rglob("*.csv")) + sorted(directory.rglob("events.out.tfevents.*"))
    if len(files) > 16:
        result["missing"].append("Only the newest 16 scalar files inspected (bounded CPU review)")
    values = {}
    for path in files[-16:]:
        if path.stat().st_size > MAX_FILE_BYTES:
            result["missing"].append(f"Oversized scalar file not loaded: {path.name}")
            continue
        try:
            if path.suffix == ".csv":
                with path.open(encoding="utf-8", newline="") as stream:
                    for n, row in enumerate(csv.DictReader(stream)):
                        if n >= MAX_RECORDS:
                            raise ValueError("CSV scalar record bound exceeded")
                        step = row.get("global_step", row.get("step"))
                        for key, value in row.items():
                            if value and any(token in key.lower() for token in SCALAR_TOKENS):
                                try:
                                    number = float(value)
                                except ValueError:
                                    continue
                                if math.isfinite(number):
                                    values.setdefault(key, []).append([step, number])
            else:
                try:
                    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
                except ImportError:
                    result["missing"].append("TensorBoard scalar reader not installed; no package installed automatically")
                    continue
                accumulator = EventAccumulator(str(path), size_guidance={"scalars": 2000}).Reload()
                for key in accumulator.Tags().get("scalars", []):
                    if any(token in key.lower() for token in SCALAR_TOKENS):
                        values.setdefault(key, []).extend([[event.step, event.value] for event in accumulator.Scalars(key)])
            result["sources"].append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        except (ValueError, OSError) as error:
            result["missing"].append(f"Scalar read unavailable for {path.name}: {error}")
    for key, entries in values.items():
        result["scalars"][key] = {"n": len(entries), "first": entries[0], "last": entries[-1],
                                   "min": min(v for _, v in entries), "max": max(v for _, v in entries)}
    result["status"] = "available" if values else "unavailable"
    if not any("lr" in key.lower() or "learning_rate" in key.lower() for key in values):
        result["missing"].append("actual optimizer learning rates")
    if not any("bpp" in key.lower() for key in values):
        result["missing"].append("rate loss component")
    return result


def review_bundle(bundle, *, bootstrap_resamples=0, image_root=None, logs_dir=None):
    from tools.eval_machine import image_flags, summarize
    from utils.eval_stats import aggregate, bootstrap_ci
    if not isinstance(bootstrap_resamples, int) or not 0 <= bootstrap_resamples <= 100:
        raise ValueError("CPU bootstrap-resamples must be in 0..100")
    ids, originals, indexed, missing = validate_bundle(bundle)
    points = {}
    for name, (quality, pred) in indexed.items():
        rows = [quality[i] for i in ids]
        items = [{"row": quality[i], "gt": originals[i], "pred": pred[i],
                  "site_id": quality[i]["site_id"]} for i in ids]
        machine = summarize(items, threshold=.2, gt_threshold=.2)
        flags = {item["row"]["image_id"]: image_flags(item, .2) for item in items}
        machine["denominators"] = {k: sum(f[k] is not None for f in flags.values())
                                   for k in ("empty_fp", "hallucination", "missed_animal")}
        if bootstrap_resamples:
            machine["ci"] = bootstrap_ci(items, ("map", "missed_animal_rate"),
                n_resamples=bootstrap_resamples, seed=0,
                summarize=lambda sample: summarize(sample, threshold=.2, gt_threshold=.2))
        points[name] = {"quality": aggregate(rows, QUALITY_METRICS), "machine": machine,
                        "missed_case_ids": [i for i in ids if flags[i]["missed_animal"]],
                        "hallucination_candidate_ids": [i for i in ids if flags[i]["hallucination"]],
                        "quality_records_sha256": canonical_hash(rows),
                        "prediction_records_sha256": canonical_hash([pred[i] for i in ids]),
                        "protocol_source": bundle["points"][name].get("protocol_source")}
    baseline = points["B0_2"]
    tradeoff = any(points[name]["quality"]["bpp"] < baseline["quality"]["bpp"]
                  and (points[name]["machine"]["map"] or 0) < (baseline["machine"]["map"] or 0)
                  for name in ("H1_500", "H1_1000"))
    verified = not missing
    recommendation = "conditional_low_lr_trial" if verified and tradeoff else "stop_review"
    cases = sorted(set(points["H1_1000"]["missed_case_ids"] + points["H1_500"]["missed_case_ids"]))[:10]
    previews = []
    for i in cases:
        relative = safe_relative(indexed["B0_2"][0][i]["relative_path"])
        path = Path(image_root) / relative if image_root else None
        if path is not None:
            path = path.resolve()
            if not path.is_relative_to(Path(image_root).resolve()):
                raise ValueError("Image preview path escapes image root")
        previews.append({"image_id": i, "relative_path": str(relative),
                         "original_available": bool(path and path.is_file()),
                         "original_path": str(path) if path and path.is_file() else None})
    loss = review_loss_logs(logs_dir or bundle.get("loss_logs_dir"))
    dimensions = Counter((indexed["B0_2"][0][i]["width"], indexed["B0_2"][0][i]["height"]) for i in ids)
    bpp_change = 100 * (points["H1_1000"]["quality"]["bpp"] / baseline["quality"]["bpp"] - 1)
    return {"schema_version": 1, "evidence_sha256": canonical_hash(bundle),
            "input_hash_kind": "canonical_selected_evidence_json", "protocol_verified": verified,
            "training_allowed": recommendation == "conditional_low_lr_trial", "recommendation": recommendation,
            "protocol": EXPECTED_PROTOCOL, "missing_evidence": missing,
            "selection": {"n": 30, "split": "val", "image_ids": ids, "ids_sha256": ids_hash(ids),
                          "sampling": "frozen stratified dev30; selection signal, not population estimate",
                          "original_dimensions": [{"width": w, "height": h, "n": n} for (w, h), n in dimensions.items()]},
            "sources": bundle["provenance"], "source_protocols": bundle.get("protocol_evidence", {}),
            "points": points, "rate_comparison": rate_comparison(points), "loss_review": loss,
            "case_review": {"status": "inventory_only", "cases": previews,
                            "warning": "No visual conclusion: inspect original plus existing B0/H1 reconstructions; no images generated."},
            "trial_rationale": {"bpp_change_H1_1000_vs_B0_2_percent": bpp_change,
                "reason": "H1 trades lower bitrate for lower detector agreement; unequal-rate scores do not establish global failure.",
                "hypothesis": "A single smaller main LR may limit early drift; cause unconfirmed by scalar/visual evidence.",
                "limits": {"total_optimizer_steps": 200, "main_lr": 1e-5, "aux_lr": 1e-3,
                           "lambda_rate": 2, "crop": 256, "source": "author lambda-2 weights; fresh optimizer",
                           "automatic_gpu_allowed": False}},
            "limitations": [
                "Animal boxes from MegaDetector on original images are pseudo-reference, not human box ground truth.",
                "Missed rate counts original-positive frames with no decoded detection; it is not missed individual animals.",
                "Hallucination candidates are detector disagreement on eligible empty frames, not human-confirmed invented animals.",
                "Bpp uses actual bitstream bytes divided by original-frame pixels; 512 processing cannot preserve all fine original detail.",
                "30 stratified validation images do not prove population improvement, significance, or publication readiness.",
                "B0 brackets/log-rate interpolation are descriptive, not measured matched-rate results or BD-rate.",
                "No images, checkpoint bytes, model inference, or full-dev/test evaluation were generated by this review.",
                "Unavailable scalar logs or images cannot establish that learning rate or cropping caused the observed tradeoff."],
            "operations": ["cached_json_read", "cpu_aggregation", "cpu_coco_scoring"]}


def render_markdown(report):
    def number(value, places=4):
        return "—" if value is None else f"{value:.{places}f}"
    lines = ["# Kiểm tra H1 bằng kết quả đã lưu", "",
             "Đã tính lại đúng cùng 30 ảnh kiểm chứng; không huấn luyện, giải mã hay chạy bộ phát hiện mới.", "",
             "| Điểm | Bit/pixel ảnh gốc ↓ | Chất lượng điểm ảnh ↑ | Sai khác cảm nhận ↓ | Độ khớp phát hiện ↑ | Ảnh mất phát hiện ↓ |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, point in report["points"].items():
        q, m = point["quality"], point["machine"]
        lines.append(f"| {name} | {number(q['bpp'], 6)} | {number(q['psnr'])} | {number(q['lpips'])} | {number(m['map'])} | {number(m['missed_animal_rate'])} |")
    lines += ["", "Tên cột tương ứng: số bit trên mỗi pixel; điểm chất lượng tái tạo (PSNR); sai khác cảm nhận (LPIPS); độ chính xác phát hiện trung bình (mAP). Mũi tên cho biết chiều tốt hơn.",
              "", "## Kết luận dễ hiểu", "",
              f"H1 ở 1.000 bước dùng {abs(report['trial_rationale']['bpp_change_H1_1000_vs_B0_2_percent']):.1f}% {'ít' if report['trial_rationale']['bpp_change_H1_1000_vs_B0_2_percent'] < 0 else 'nhiều'} bit hơn B0 λ=2. Chưa thể gọi đây là tiến bộ toàn diện: phải xem khả năng giữ con vật cùng với dung lượng.",
              "Không so điểm số ở hai dung lượng khác nhau rồi kết luận toàn bộ phương pháp tốt/xấu hơn. Ba mức B0 đã được tính trên đúng cùng 30 ảnh để đối chiếu."]
    lines.append("")
    for name, comparison in report["rate_comparison"].items():
        bracket = " ↔ ".join(comparison["bracket"]) if comparison["bracket"] else "nằm ngoài khoảng B0 đã đo"
        dominates = comparison["metrics"]["map"]["dominated_by"]
        lines.append(f"- {name}: khoảng dung lượng {bracket}; B0 dùng không nhiều bit hơn mà có điểm phát hiện không thấp hơn: {', '.join(dominates) or 'không có điểm đã đo'}. Không có điểm lấn át không có nghĩa H1 thắng cả đường cong.")
    lines += ["", f"Thông số giải mã thực tế: {'đã xác minh' if report['protocol_verified'] else 'CHƯA ĐỦ BẰNG CHỨNG'}. Quyết định: `{report['recommendation']}`.",
              "Nếu được phép thử: một run mới từ trọng số tác giả λ=2, optimizer mới, tốc độ học chính 0,00001; tốc độ học phụ vẫn 0,001. Dừng ở 200 bước rồi chấm đúng 30 ảnh. Không tự kéo lên 3.000 bước hoặc làm H2.",
              "", "## Chưa thể khẳng định", "",
              "Đây là đối chiếu với nhãn tự động từ ảnh gốc, không phải hộp con vật do người gán. Tỉ lệ mất phát hiện tính trên số ảnh gốc có phát hiện con vật; không phải số con vật bị mất. Các ứng viên ‘con vật ảo’ phải kiểm tra ảnh thủ công.",
              f"Bằng chứng thành phần hàm lỗi/tốc độ học: `{report['loss_review']['status']}`. Chưa xác định tốc độ học là nguyên nhân; giảm tốc độ học chỉ là giả thuyết thử có kiểm soát.",
              "Ảnh minh họa: chỉ liệt kê các trường hợp cần xem, không tự tải/sinh ảnh và chưa đưa ra kết luận bằng mắt.",
              "", "## Khoảng 10 trường hợp cần xem", ""]
    for case in report["case_review"]["cases"]:
        lines.append(f"- `{case['image_id']}` — `{case['relative_path']}`; ảnh gốc {'có sẵn' if case['original_available'] else 'chưa có trong phiên kiểm tra này'}. Xem cùng ảnh B0 và H1 đã lưu.")
    lines += ["", "## Dấu vết kiểm chứng", "",
              f"Mã băm dữ liệu đã chọn: `{report['evidence_sha256']}`. Mã băm danh sách 30 ảnh: `{report['selection']['ids_sha256']}`.",
              "JSON đi kèm giữ nguồn từng điểm, thông số giải mã, mẫu số và mã băm bản ghi. Chưa đánh dấu Phase 3 hoàn thành.", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--bundle")
    source.add_argument("--drive-root")
    parser.add_argument("--protocol-sidecar", help="Observed per-point run_info and contracts, not assumed defaults")
    parser.add_argument("--trial-dir")
    parser.add_argument("--logs-dir")
    parser.add_argument("--image-root", help="Optional existing originals only; never downloads or generates images")
    parser.add_argument("--bootstrap-resamples", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.trial_dir and not args.drive_root:
        parser.error("trial-dir is available only with drive-root")
    bundle = read_json(args.bundle) if args.bundle else load_drive_bundle(args.drive_root, args.trial_dir)
    if args.protocol_sidecar:
        attach_protocol_evidence(bundle, read_json(args.protocol_sidecar))
    report = review_bundle(bundle, bootstrap_resamples=args.bootstrap_resamples,
                           image_root=args.image_root, logs_dir=args.logs_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(render_markdown(report), encoding="utf-8")
    print(f"CPU cached review: {output}\nProtocol verified: {report['protocol_verified']}\nDecision: {report['recommendation']}\nEvidence SHA256: {report['evidence_sha256']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
