"""Stable Phase 3 notebook operations. Pulling main updates these commands.

No baseline generation, test evaluation, ROI training or Xie reproduction.
Torch is imported only when inspecting a trusted project checkpoint.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
MANIFEST = REPO / "data/manifests/kgalagadi_site_split.jsonl"
CONFIG = "configs/train_kgalagadi_h1_pilot.yaml"
PROTOCOL = {"lambda": 2, "processing_long_side": 512, "crop": 256,
            "seed": 20260916, "decode_seed": 231, "ddim_steps": 50,
            "cfg_scale": 3.0, "control_ratio": 1.0, "pooled": True}


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def ids(path):
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")]


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def freeze(path, text):
    path = Path(path)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise RuntimeError(f"Frozen artifact changed: {path}. Do not overwrite an experiment.")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".part")
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)


def select_quick(rows, dev_ids, illumination, detections, count=30):
    """Round-robin deterministic coverage, not an unbiased population estimate."""
    lookup = {row["image_id"]: row for row in rows}
    if len(dev_ids) != len(set(dev_ids)) or not set(dev_ids) <= lookup.keys():
        raise ValueError("Duplicate or unknown dev IDs")
    groups = defaultdict(list)
    for image_id in dev_ids:
        row = lookup[image_id]
        if row["split"] != "val":
            raise ValueError(f"Dev selection must be validation only: {image_id}")
        if image_id not in illumination or image_id not in detections:
            raise ValueError(f"Missing Phase 2 sidecar: {image_id}")
        det = detections[image_id]
        areas = [float(d["bbox"][2]) * float(d["bbox"][3]) for d in det.get("detections", [])
                 if str(d.get("category")) == "1" and float(d["conf"]) >= .2]
        # Area fraction at 512: small/medium/large by relative size, recorded
        # as sampling bins only (not the COCO original-pixel AP buckets).
        size = "no_box" if not areas else ("small" if min(areas) < .01 else "larger")
        key = (illumination[image_id]["illumination"], bool(row["is_empty"]), size)
        groups[key].append(image_id)
    for bucket in groups.values():
        bucket.sort(key=lambda value: hashlib.sha256(f"231:{value}".encode()).hexdigest())
    selected = []
    keys = sorted(groups)
    while len(selected) < min(count, len(dev_ids)):
        for key in keys:
            if groups[key] and len(selected) < count:
                selected.append(groups[key].pop(0))
    return sorted(selected)


def paths(root):
    root = Path(root)
    phase = root / "phase3"
    return {"root": root, "phase": phase, "p2": root / "phase2",
            "run": root / "runs/h1_pooled_pilot_ls512/lambda_2",
            "images": Path("/content/data/wild_diff_icmh/images"),
            "checkpoints": Path("/content/data/wild_diff_icmh/checkpoints"),
            "tags": root / "tags/KGA_all.jsonl",
            "quick": phase / "dev30.txt", "smoke": phase / "smoke4.txt",
            "full": root / "phase2/kgalagadi_dev_b0eval.txt",
            "illumination": root / "phase2/kgalagadi_illumination.jsonl",
            "detections": root / "phase2/detections/originals.jsonl",
            "b0": root / "phase2/archive/B0_ls512_ddim50/lambda_2"}


def command_env(p):
    env = os.environ.copy()
    env.update(WILD_DATA_ROOT=str(p["images"]), KGA_TAGS=str(p["tags"]),
               KGA_DETECTIONS=str(p["detections"]), BPP_WEIGHT="2", WILD_RUN_DIR=str(p["run"]),
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    env.pop("KGA_SITE_ID", None)
    return env


def run_logged(command, p, name, rate):
    log = p["root"] / "logs" / f"p3_{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [str(value) for value in command]
    started = time.monotonic()
    print("Lệnh:", " ".join(command), "\nLog:", log, flush=True)
    with log.open("a", encoding="utf-8") as stream:
        stream.write(f"\n=== {datetime.now(timezone.utc).isoformat()} ===\n{' '.join(command)}\n")
        process = subprocess.Popen(command, cwd=REPO, env=command_env(p), stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            print(line, end="", flush=True)
            stream.write(line)
            stream.flush()
        code = process.wait()
    elapsed = time.monotonic() - started
    ledger = p["phase"] / "resource_usage.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a", encoding="utf-8") as stream:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
        stream.write(json.dumps({"name": name, "elapsed_seconds": elapsed, "exit_code": code,
                                 "cu_estimate": elapsed / 3600 * rate, "cu_rate_assumed": rate,
                                 "git_commit": commit,
                                 "command": command, "time": datetime.now(timezone.utc).isoformat()}) + "\n")
    if code:
        raise RuntimeError(f"{name} dừng với mã {code}. Gửi log đầy đủ: {log}")


def prepare(p):
    rows = read_jsonl(MANIFEST)
    for key in ("tags", "full", "illumination", "detections"):
        if not p[key].is_file():
            raise FileNotFoundError(f"Missing Phase 2 artifact (will not regenerate): {p[key]}")
    trainval = [row for row in rows if row["split"] in {"train", "val"}]
    cache = {item["image_id"]: item for item in read_jsonl(p["tags"])}
    missing = [row["image_id"] for row in trainval if row["image_id"] not in cache
               or not isinstance(cache[row["image_id"]].get("tags"), str)]
    if missing:
        raise ValueError(f"Missing cached RAM tags for {len(missing)} train/val images")
    missing_images = [row["relative_path"] for row in trainval if not (p["images"] / row["relative_path"]).is_file()]
    if missing_images:
        raise FileNotFoundError(f"{len(missing_images)} local images missing. Rerun step 4.")
    dev = ids(p["full"])
    illum = {item["image_id"]: item for item in read_jsonl(p["illumination"])}
    det = {item["image_id"]: item for item in read_jsonl(p["detections"])}
    chosen = select_quick(rows, dev, illum, det)
    if len(chosen) != 30:
        raise ValueError("Need at least 30 cached validation images")
    freeze(p["quick"], "\n".join(chosen) + "\n")
    # Spread the smoke sample across the deterministic selection.
    freeze(p["smoke"], "\n".join(chosen[index] for index in (0, 7, 15, 23)) + "\n")
    contract = {**PROTOCOL, "manifest_sha256": digest(MANIFEST), "tags_sha256": digest(p["tags"]),
                "quick_sha256": digest(p["quick"]), "full_sha256": digest(p["full"]),
                "illumination_sha256": digest(p["illumination"]), "detections_sha256": digest(p["detections"])}
    freeze(p["phase"] / "protocol.json", json.dumps(contract, sort_keys=True, indent=2) + "\n")
    stats = Counter((illum[i]["illumination"], bool(next(r for r in rows if r["image_id"] == i)["is_empty"])) for i in chosen)
    print("Train/val:", len(trainval), "sites:", len({r['site_id'] for r in trainval}))
    print("Frozen dev30 (day/night, human-labelled empty):", dict(stats))
    print("PREPARE THÀNH CÔNG — không tạo lại baseline/tags.")


def checkpoint_step(path):
    # Only user-trusted project files; checkpoint includes optimizer state.
    import torch
    from utils.checkpoint_contract import validate_project_resume_checkpoint
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    validate_project_resume_checkpoint(checkpoint)
    step = int(checkpoint["global_step"])
    del checkpoint
    gc.collect()
    return step


def train(p, target, hours, rate):
    if target not in (20, 500, 1000, 3000):
        raise ValueError("Targets: 20 smoke, 500 pilot, 1000/3000 explicit extension")
    if not 0 < hours <= 5:
        raise ValueError("Session hours must be >0 and <=5")
    prepare(p)
    # Smoke has its own run; never resume the old site A01 or seed a pilot
    # from a smoke run's optimizer. The pilot starts from the author weights.
    if target == 20:
        p = {**p, "run": p["root"] / "runs/h1_pooled_smoke_ls512/lambda_2"}
    p["run"].mkdir(parents=True, exist_ok=True)
    contract = {"protocol": json.loads((p["phase"] / "protocol.json").read_text()),
                "config_chain_sha256": [digest(REPO / f) for f in
                 (CONFIG, "configs/train_kgalagadi_pooled.yaml", "configs/train_kgalagadi_colab.yaml",
                  "configs/model/diffeic.yaml", "configs/dataset/kgalagadi_train.yaml", "configs/dataset/kgalagadi_val.yaml")]}
    freeze(p["run"] / "experiment_contract.json", json.dumps(contract, sort_keys=True, indent=2) + "\n")
    last = p["run"] / "checkpoints/last.ckpt"
    if last.is_file():
        step = checkpoint_step(last)  # fail closed: no silent restart on corruption
        if step >= target:
            print(f"Đã đạt {step}/{target} optimizer steps — không train lại.")
            return
    elif list((p["run"] / "checkpoints").glob("*.ckpt")):
        raise RuntimeError("Missing last.ckpt but other checkpoints exist; inspect before restarting.")
    author = p["checkpoints"] / "difficmh_models/CNscale1.0_1_1_2_2_WTagGCM_bs16x1_lr0.00005_cfg7.0/model.ckpt"
    if not author.is_file():
        raise FileNotFoundError(author)
    seconds = int(hours * 3600)
    max_time = f"00:{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"
    command = [sys.executable, "-u", "train.py", "--config", CONFIG, "--init-checkpoint", author,
               "--resume", str(last) if last.is_file() else "auto",
               f"lightning.trainer.max_steps={target}", f"lightning.trainer.max_time={max_time}"]
    run_logged(command, p, f"train_{target}", rate)
    actual = checkpoint_step(last)
    print(f"Checkpoint hợp lệ: {actual}/{target} steps. " +
          ("Đã đạt mục tiêu." if actual >= target else "Hết thời gian phiên; chạy lại cell này để resume."))


def snapshot(p):
    source = p["run"] / "checkpoints/last.ckpt"
    step = checkpoint_step(source)
    target = p["phase"] / "snapshots" / p["run"].parent.name / f"step_{step:06d}" / "checkpoints/last.ckpt"
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".part")
        shutil.copyfile(source, temporary)
        if checkpoint_step(temporary) != step:
            raise RuntimeError("Checkpoint changed while copying")
        os.replace(temporary, target)
    elif digest(target) != digest(source):
        raise RuntimeError("Same global step has different weights; preserve old snapshot and inspect run.")
    freeze(target.parent.parent / "config_model.yaml", (p["run"] / "config_model.yaml").read_text(encoding="utf-8"))
    return target, step


def assert_archive_complete(root, selected):
    root = Path(root)
    logged = {item["image_id"]: item for item in read_jsonl(root / "decode_log.jsonl")}
    rows = {row["image_id"]: row for row in read_jsonl(MANIFEST)}
    missing = []
    for image_id in selected:
        png = (root / rows[image_id]["relative_path"]).with_suffix(".png")
        stream = png.parent / "data" / png.stem
        if image_id not in logged or not png.is_file() or not stream.is_file() or stream.stat().st_size == 0:
            missing.append(image_id)
    if missing:
        raise RuntimeError(f"Archive incomplete ({len(missing)} images): {root}")


def evaluate(p, scope, rate, detect_python):
    prepare(p)
    if scope == "smoke":
        p = {**p, "run": p["root"] / "runs/h1_pooled_smoke_ls512/lambda_2"}
    checkpoint, step = snapshot(p)
    minimum = 20 if scope == "smoke" else 500
    if step < minimum:
        raise ValueError(f"Checkpoint has {step}/{minimum} steps. Resume the training cell before evaluation.")
    selection = p[{"smoke": "smoke", "quick": "quick", "full": "full"}[scope]]
    selected = ids(selection)
    if scope != "smoke":
        b0info = json.loads((p["b0"] / "run_info.json").read_text(encoding="utf-8"))
        if (b0info.get("processing_long_side") != 512 or b0info.get("steps") != 50
                or b0info.get("sampler") != "ddim" or b0info.get("seed") != 231
                or b0info.get("c_cfg_scale") != 3.0):
            raise ValueError("Cached B0 does not match the H1 evaluation protocol")
        assert_archive_complete(p["b0"], selected)
    steps = 5 if scope == "smoke" else 50
    folder = p["phase"] / "archive" / f"step_{step:06d}_{scope}_ddim{steps}"
    info = {"protocol": PROTOCOL, "scope": scope, "steps": steps, "checkpoint_sha256": digest(checkpoint),
            "dev_sha256": digest(selection), "checkpoint_step": step}
    freeze(folder / "evaluation_contract.json", json.dumps(info, sort_keys=True, indent=2) + "\n")
    command = [sys.executable, "-u", "inference_partition.py", "--ckpt_sd", p["checkpoints"] / "sd2p1/v2-1_512-ema-pruned.ckpt",
               "--ckpt_lc", checkpoint, "--config", "configs/model/diffeic.yaml", "--input", p["images"],
               "--output", folder, "--manifest", MANIFEST, "--dev-list", selection, "--tag-cache", p["tags"],
               "--split", "val", "--sampler", "ddim", "--steps", str(steps), "--seed", "231", "--device", "cuda",
               "--processing-long-side", "512", "--skip-existing",
               "params.control_stage_config.params.control_model_ratio=1.0", "params.c_cfg_scale=3.0"]
    run_logged(command, p, f"decode_{step}_{scope}", rate)
    assert_archive_complete(folder, selected)
    if scope == "smoke":
        print("SMOKE THÀNH CÔNG — kiểm tra đường chạy, không so sánh chất lượng DDIM5 với B0 DDIM50.")
        return
    # Equal images/geometry/sampler settings; B0 comes from Phase 2, no decode.
    output = p["phase"] / "eval" / f"step_{step:06d}_{scope}"
    for method, archive in (("B0", p["b0"]), ("H1", folder)):
        # B0 metrics are reusable across checkpoints: no scoring them anew
        # after every extension. H1 output remains bound to one snapshot.
        method_output = p["phase"] / "eval" / f"B0_{scope}" if method == "B0" else output
        image_eval = method_output / f"{method}.jsonl"
        marker = method_output / f"{method}.done.json"
        marker_info = {"dev_sha256": digest(selection), "b0_info_sha256": digest(p["b0"] / "run_info.json")} if method == "B0" else info
        if marker.exists():
            if json.loads(marker.read_text()) != marker_info:
                raise RuntimeError("Evaluation marker contract mismatch")
            print("Đã chấm đủ:", method, scope)
            continue
        common = ["--manifest", MANIFEST, "--split", "val", "--dev-list", selection,
                  "--illumination-sidecar", p["illumination"]]
        registry_args = ["--results-registry", p["root"] / "results/results.jsonl", "--lambda-rate", "2",
                         "--git-commit", subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                         "--ddim-steps", "50", "--exp-id",
                         f"p3_B0_{scope}" if method == "B0" else f"p3_H1_step_{step}_{scope}"]
        run_logged([sys.executable, "-u", "tools/evaluate_kgalagadi.py", *common, "--data-root", p["images"],
                    "--reconstruction-root", archive, "--detections", p["detections"], "--method", method,
                    "--lpips", "--bootstrap-resamples", "100", "--output", image_eval, *registry_args], p, f"quality_{step}_{scope}_{method}", rate)
        # Reuse Phase 2 B0 detection cache if available, but run only this subset
        # when absent; never rerun MegaDetector on originals.
        cached = p["p2"] / "detections/B0_ls512_ddim50__lambda_2.jsonl"
        pred = cached if method == "B0" and cached.is_file() else method_output / f"{method}.detections.jsonl"
        run_logged([detect_python, "-u", "tools/detect/run_megadetector.py", "--manifest", MANIFEST,
                    "--split", "val", "--dev-list", selection, "--image-root", archive, "--suffix", ".png",
                    "--output", pred], p, f"detect_{step}_{scope}_{method}", rate)
        run_logged([sys.executable, "-u", "tools/eval_machine.py", *common, "--gt", p["detections"], "--pred", pred,
                    "--method", method, "--bootstrap-resamples", "100", "--output", method_output / f"{method}.machine.json",
                    *registry_args],
                   p, f"machine_{step}_{scope}_{method}", rate)
        write_json(marker, marker_info)
    comparison = {"scope": scope, "checkpoint_step": step, "n": len(selected), "methods": {}}
    for method in ("B0", "H1"):
        directory = p["phase"] / "eval" / f"B0_{scope}" if method == "B0" else output
        quality = json.loads((directory / f"{method}.summary.json").read_text())["groups"]
        machine = json.loads((directory / f"{method}.machine.json").read_text())["groups"]
        comparison["methods"][method] = {"quality": quality, "machine": machine}
    write_json(output / "comparison.json", comparison)
    print("ĐÁNH GIÁ THÀNH CÔNG:", output)
    if scope == "quick":
        print("30 ảnh là tín hiệu chọn pilot, không phải kết luận cho toàn bộ corpus/bài báo.")


def status(p, rate):
    print("Run:", p["run"], "\nArtifacts:", p["phase"])
    for name in ("throughput.json", "crop_stats.json"):
        path = p["run"] / name
        if path.is_file():
            report = json.loads(path.read_text())
            print(name, json.dumps(report, ensure_ascii=False, indent=2))
            if name == "throughput.json":
                speed = report["seconds_per_optimizer_step"]
                for target in (500, 1000, 3000):
                    remaining = max(0, target - report["global_step_end"]) * speed / 3600
                    print(f"Còn tới {target}: ~{remaining:.2f} giờ, ~{remaining * rate:.2f} CU (chưa gồm setup/eval)")
    ledger = p["phase"] / "resource_usage.jsonl"
    if ledger.is_file():
        cost = sum(item["cu_estimate"] for item in read_jsonl(ledger))
        print(f"Chi phí các job đã ghi: ~{cost:.2f} CU; ước tính, không phải hóa đơn Colab.")
    print("Phase 3 chưa tự đánh dấu hoàn thành; cần review kết quả GPU và quyết định mở rộng.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "train", "evaluate", "status"])
    parser.add_argument("--drive-root", default="/content/drive/MyDrive/wild_diff_icmh")
    parser.add_argument("--target-steps", type=int, default=500)
    parser.add_argument("--session-hours", type=float, default=1.5)
    parser.add_argument("--cu-rate", type=float, default=1.54)
    parser.add_argument("--scope", choices=["smoke", "quick", "full"], default="quick")
    parser.add_argument("--allow-full", action="store_true")
    parser.add_argument("--allow-extension", action="store_true")
    parser.add_argument("--detect-python", default="/content/envs/detect/bin/python")
    args = parser.parse_args(argv)
    if args.cu_rate <= 0:
        parser.error("cu-rate must be positive")
    if args.action == "evaluate" and args.scope == "full" and not args.allow_full:
        parser.error("Full dev evaluation requires explicit --allow-full")
    if args.action == "train" and args.target_steps > 500 and not args.allow_extension:
        parser.error("Training beyond pilot requires explicit --allow-extension")
    p = paths(args.drive_root)
    if args.action == "prepare":
        prepare(p)
    elif args.action == "train":
        train(p, args.target_steps, args.session_hours, args.cu_rate)
    elif args.action == "evaluate":
        evaluate(p, args.scope, args.cu_rate, args.detect_python)
    else:
        status(p, args.cu_rate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
