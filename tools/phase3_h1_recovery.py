"""CPU-first review and explicitly approved, isolated 200-step H1 LR test.

The CLI has no extension, full-dev, test, arbitrary checkpoint or baseline
inference options. The old pilot's code/config/contracts remain unchanged.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import re
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tools import phase3_h1 as old

CONFIG = "configs/train_kgalagadi_h1_low_lr.yaml"
PROTOCOL = old.PROTOCOL
TARGET_STEPS = 200
EXPECTED_LRS = {"main": 1e-5, "entropy_aux": 1e-3}
CHECKPOINT_IDENTITY_KEY = "wild_diff_h1_recovery_contract_sha256"
AUTHOR_RELATIVE = "difficmh_models/CNscale1.0_1_1_2_2_WTagGCM_bs16x1_lr0.00005_cfg7.0/model.ckpt"
digest = old.digest
run_logged = old.run_logged
CONFIG_CHAIN = (CONFIG, old.CONFIG, "configs/train_kgalagadi_pooled.yaml",
                "configs/train_kgalagadi_colab.yaml", "configs/model/diffeic.yaml",
                "configs/dataset/kgalagadi_train.yaml", "configs/dataset/kgalagadi_val.yaml")


def paths(root):
    p = old.paths(root)
    p["phase"] = p["root"] / "phase3/h1_low_lr_recovery"
    p["run"] = p["root"] / "runs/h1_low_lr_recovery_ls512/lambda_2"
    p["review"] = p["phase"] / "review.json"
    return p


def validate_acknowledgement(path, acknowledgement, *, enabled):
    if not enabled:
        raise RuntimeError("Recovery requires explicit user enablement; no job was launched")
    path = Path(path)
    if not path.is_file():
        raise RuntimeError("Run the CPU review first; review.json is missing")
    if not re.fullmatch(r"[0-9a-f]{64}", str(acknowledgement)) or digest(path) != acknowledgement:
        raise RuntimeError("Acknowledged review hash does not match the report you read")
    value = json.loads(path.read_text(encoding="utf-8"))
    if (value.get("schema_version") != 1 or value.get("protocol_verified") is not True
            or not re.fullmatch(r"[0-9a-f]{64}", str(value.get("evidence_sha256", "")))):
        raise RuntimeError("A verified source-bound cached review is required")
    if value.get("training_allowed") is not True or value.get("recommendation") != "conditional_low_lr_trial":
        raise RuntimeError("Review does not support a trial; inspect the report before deciding")
    return value


def validate_recovery_checkpoint(checkpoint, identity):
    from utils.checkpoint_contract import validate_project_resume_checkpoint
    validate_project_resume_checkpoint(checkpoint)
    if checkpoint.get(CHECKPOINT_IDENTITY_KEY) != identity:
        raise RuntimeError("Recovery checkpoint identity mismatch; cannot resume a foreign run")
    optimizers = checkpoint["optimizer_states"]
    if len(optimizers) != 1:
        raise RuntimeError("Recovery requires one optimizer")
    groups = optimizers[0].get("param_groups", [])
    names = [item.get("name") for item in groups]
    if len(names) != len(set(names)) or set(names) != set(EXPECTED_LRS):
        raise RuntimeError("Recovery optimizer group contract mismatch")
    for group in groups:
        value = float(group.get("lr", float("nan")))
        if not math.isfinite(value) or not math.isclose(value, EXPECTED_LRS[group["name"]], rel_tol=1e-9, abs_tol=0):
            raise RuntimeError(f"Recovery optimizer learning rate mismatch: {group['name']}={value}")
    step = int(checkpoint["global_step"])
    if not 0 <= step <= TARGET_STEPS:
        raise RuntimeError("Recovery checkpoint exceeds the fixed 200-step experiment")
    return step


def checkpoint_step(path, identity):
    """Only call for this run's explicitly approved, user-trusted checkpoint."""
    print("Kiểm tra checkpoint riêng của recovery (CPU):", path, flush=True)
    import torch
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    try:
        return validate_recovery_checkpoint(checkpoint, identity)
    finally:
        del checkpoint
        gc.collect()


def _session_seconds(hours):
    if not math.isfinite(hours) or not 0 < hours <= 2 or int(hours * 3600) < 1:
        raise ValueError("Session hours must be finite, >0, <=2 and allow at least one second")
    return int(hours * 3600)


def _bind_review_selection(p, report):
    chosen = old.ids(p["quick"])
    reviewed = report.get("selection", {}).get("image_ids")
    if len(chosen) != 30 or len(set(chosen)) != 30 or reviewed != chosen:
        raise RuntimeError("Recovery must use exactly the same ordered frozen dev30 as the review")
    actual = json.loads((p["quick"].parent / "protocol.json").read_text(encoding="utf-8"))
    if any(actual.get(key) != value for key, value in PROTOCOL.items()):
        raise RuntimeError("Frozen pilot protocol changed; do not alter its evidence")


def prepare_assets(p):
    # Proven preflight; writes only identical existing frozen pilot sidecars.
    old.prepare({**p, "phase": p["quick"].parent})


def _contract(p, report, acknowledgement, author):
    return {"schema_version": 1, "experiment": "h1_low_lr_recovery_ls512_lambda2",
            "review_sha256": acknowledgement, "evidence_sha256": report["evidence_sha256"],
            "approved_decision": report["recommendation"], "target_steps": TARGET_STEPS,
            "protocol": json.loads((p["quick"].parent / "protocol.json").read_text()),
            "quick_sha256": digest(p["quick"]), "optimizer_lrs": EXPECTED_LRS,
            "initialization": "trusted_author_weights_only_fresh_optimizer",
            "author_relative": AUTHOR_RELATIVE, "author_sha256": digest(author),
            "config_chain_sha256": {name: digest(REPO / name) for name in CONFIG_CHAIN}}


def train(p, hours, rate, *, enabled=False, acknowledgement=""):
    seconds = _session_seconds(hours)
    report = validate_acknowledgement(p["review"], acknowledgement, enabled=enabled)
    _bind_review_selection(p, report)
    prepare_assets(p)
    author = p["checkpoints"] / AUTHOR_RELATIVE
    if not author.is_file():
        raise FileNotFoundError(author)
    contract_path = p["run"] / "experiment_contract.json"
    # An orphan pre-existing run cannot be adopted as a fresh experiment.
    if p["run"].exists() and not contract_path.exists() and any(p["run"].iterdir()):
        raise RuntimeError("Recovery run already has unbound files; preserve them and inspect first")
    value = _contract(p, report, acknowledgement, author)
    old.freeze(contract_path, json.dumps(value, sort_keys=True, indent=2) + "\n")
    old.freeze(p["run"] / "acknowledged_review.json", p["review"].read_text(encoding="utf-8"))
    identity = digest(contract_path)
    last = p["run"] / "checkpoints/last.ckpt"
    step = 0
    if last.is_file():
        if not last.resolve().is_relative_to(p["run"].resolve()):
            raise RuntimeError("Recovery checkpoint escapes its own run")
        step = checkpoint_step(last, identity)
        if step == TARGET_STEPS:
            print("Đã đủ 200 bước của recovery — không train lại.", flush=True)
            return
    elif list((p["run"] / "checkpoints").glob("*.ckpt")):
        raise RuntimeError("Missing own last.ckpt; never restart silently from another checkpoint")
    duration = f"00:{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"
    command = [sys.executable, "-u", "train.py", "--config", CONFIG, "--session-time-limit",
               f"lightning.trainer.max_steps={TARGET_STEPS}", f"lightning.trainer.max_time={duration}"]
    if last.is_file():
        command.extend(["--resume", str(last)])
    else:
        command.extend(["--init-checkpoint", author])
    print(f"Recovery: {step} → 200 bước, main LR=1e-5, aux LR=1e-3; "
          "optimizer mới chỉ ở lần đầu. Không kéo dài tự động.", flush=True)
    run_logged(command, p, "h1_low_lr_recovery_train_200", rate)
    actual = checkpoint_step(last, identity)
    print(f"Checkpoint recovery: {actual}/200. " +
          ("Chạy đánh giá riêng khi được bật." if actual == 200 else "Hết phiên; chạy lại cell train để resume đúng run."), flush=True)


def review(p, bundle=None):
    from tools import review_h1_cached as audit
    contract = p["run"] / "experiment_contract.json"
    if contract.exists():
        value = json.loads(contract.read_text(encoding="utf-8"))
        if not p["review"].is_file() or digest(p["review"]) != value["review_sha256"]:
            raise RuntimeError("Acknowledged review changed; preserve the existing recovery contract")
        print("Đang giữ nguyên báo cáo đã duyệt của run recovery:", p["review"], flush=True)
    else:
        args = ["--output", str(p["review"])]
        args.extend(["--bundle", str(bundle)] if bundle else ["--drive-root", str(p["root"])])
        audit.main(args)
    print("Mã băm báo cáo cần đọc/duyệt:", digest(p["review"]), flush=True)
    print("Báo cáo:", p["review"], "\nGiải thích:", p["review"].with_suffix(".md"), flush=True)


def _evaluation_contract(p, acknowledgement):
    contract = p["run"] / "experiment_contract.json"
    if not contract.is_file():
        raise RuntimeError("Train the isolated trial before evaluation")
    value = json.loads(contract.read_text(encoding="utf-8"))
    if value.get("review_sha256") != acknowledgement or value.get("target_steps") != TARGET_STEPS:
        raise RuntimeError("Recovery experiment is bound to a different review/target")
    if value.get("config_chain_sha256") != {name: digest(REPO / name) for name in CONFIG_CHAIN}:
        raise RuntimeError("Recovery configuration changed after training")
    return digest(contract)


def _archive_ready(folder, selected):
    try:
        old.assert_archive_complete(folder, selected)
    except FileNotFoundError:
        return False
    except RuntimeError as error:
        if str(error).startswith("Archive incomplete"):
            return False
        raise
    return True


def evaluate(p, rate, detect_python, *, enabled=False, acknowledgement=""):
    report = validate_acknowledgement(p["review"], acknowledgement, enabled=enabled)
    _bind_review_selection(p, report)
    identity = _evaluation_contract(p, acknowledgement)
    last = p["run"] / "checkpoints/last.ckpt"
    if not last.is_file() or not last.resolve().is_relative_to(p["run"].resolve()):
        raise RuntimeError("Own recovery checkpoint is missing/foreign")
    if checkpoint_step(last, identity) != TARGET_STEPS:
        raise RuntimeError("Complete exactly 200 recovery steps before quick evaluation")
    prepare_assets(p)
    checkpoint, step = old.snapshot(p)  # Isolated nested phase and run-name namespace.
    folder = p["phase"] / "archive/step_000200_quick_ddim50"
    output = p["phase"] / "eval/step_000200_quick"
    selected = old.ids(p["quick"])
    info = {"protocol": PROTOCOL, "scope": "quick", "steps": 50,
            "checkpoint_sha256": digest(checkpoint), "checkpoint_step": step,
            "dev_sha256": digest(p["quick"]), "recovery_contract_sha256": identity,
            "review_sha256": acknowledgement}
    old.freeze(folder / "evaluation_contract.json", json.dumps(info, sort_keys=True, indent=2) + "\n")
    old.freeze(output / "evaluation_contract.json", json.dumps(info, sort_keys=True, indent=2) + "\n")
    # The CPU auditor follows this source rather than guessing an old-pilot
    # archive from the same step number. Bind it before any expensive action.
    old.freeze(output / "review_source.json", json.dumps({"archive": str(folder)},
               sort_keys=True, indent=2) + "\n")
    complete = output / "done.json"
    if complete.exists():
        if json.loads(complete.read_text()) != info:
            raise RuntimeError("Recovery evaluation marker changed")
        if not all((output / name).is_file() for name in ("H1.jsonl", "H1.detections.jsonl", "comparison.json")):
            raise RuntimeError("Recovery marker exists but cached evaluation files are missing")
        print("Đánh giá recovery đã hoàn thành:", output / "comparison.json", flush=True)
        return
    if not _archive_ready(folder, selected):
        run_logged([sys.executable, "-u", "inference_partition.py", "--ckpt_sd",
                    p["checkpoints"] / "sd2p1/v2-1_512-ema-pruned.ckpt", "--ckpt_lc", checkpoint,
                    "--config", "configs/model/diffeic.yaml", "--input", p["images"], "--output", folder,
                    "--manifest", old.MANIFEST, "--dev-list", p["quick"], "--tag-cache", p["tags"],
                    "--split", "val", "--sampler", "ddim", "--steps", "50", "--seed", "231",
                    "--device", "cuda", "--processing-long-side", "512", "--skip-existing",
                    "params.control_stage_config.params.control_model_ratio=1.0", "params.c_cfg_scale=3.0"],
                   p, "h1_low_lr_recovery_decode_200", rate)
    else:
        print("Đủ 30 ảnh/bitstream đã lưu — bỏ qua decode, giữ metadata gốc.", flush=True)
    old.assert_archive_complete(folder, selected)
    # Record actual decoder metadata, not values fabricated from intended flags.
    old.freeze(output / "run_info.json", (folder / "run_info.json").read_text(encoding="utf-8"))
    common = ["--manifest", old.MANIFEST, "--split", "val", "--dev-list", p["quick"],
              "--illumination-sidecar", p["illumination"]]
    registry = ["--results-registry", p["phase"] / "results.jsonl", "--lambda-rate", "2",
                "--ddim-steps", "50", "--exp-id", "p3_H1_low_lr_recovery_200_quick"]
    run_logged([sys.executable, "-u", "tools/evaluate_kgalagadi.py", *common,
                "--data-root", p["images"], "--reconstruction-root", folder, "--detections", p["detections"],
                "--method", "H1_low_lr_200", "--lpips", "--bootstrap-resamples", "100",
                "--output", output / "H1.jsonl", *registry], p, "h1_low_lr_recovery_quality_200", rate)
    run_logged([detect_python, "-u", "tools/detect/run_megadetector.py", "--manifest", old.MANIFEST,
                "--split", "val", "--dev-list", p["quick"], "--image-root", folder,
                "--suffix", ".png", "--output", output / "H1.detections.jsonl"],
               p, "h1_low_lr_recovery_detect_200", rate)
    run_logged([sys.executable, "-u", "tools/eval_machine.py", *common, "--gt", p["detections"],
                "--pred", output / "H1.detections.jsonl", "--method", "H1_low_lr_200",
                "--bootstrap-resamples", "100", "--output", output / "H1.machine.json", *registry],
               p, "h1_low_lr_recovery_machine_200", rate)
    from tools import review_h1_cached as audit
    comparison = audit.review_bundle(audit.load_drive_bundle(p["root"], trial_dir=output),
                                     logs_dir=p["run"])
    old.write_json(output / "comparison.json", comparison)
    old.write_json(complete, info)
    print("ĐÁNH GIÁ RECOVERY XONG:", output / "comparison.json", flush=True)
    print("B0 chỉ đọc lại số đo đã lưu; 30 ảnh chưa chứng minh thắng baseline/bài báo.", flush=True)


def status(p, rate):
    print("Run riêng:", p["run"], "\nBáo cáo CPU:", p["review"], flush=True)
    checkpoint = p["run"] / "checkpoints/last.ckpt"
    print("Checkpoint:", checkpoint, "— có" if checkpoint.is_file() else "— chưa có", flush=True)
    print("CSV loss/LR:", p["run"] / "loss_components.csv", flush=True)
    print("Đánh giá:", p["phase"] / "eval/step_000200_quick/comparison.json", flush=True)
    # Status does not load a multi-GB checkpoint or initialize torch.
    throughput = p["run"] / "throughput.json"
    if throughput.is_file():
        value = json.loads(throughput.read_text())
        remaining = max(0, TARGET_STEPS - int(value["global_step_end"]))
        hours = remaining * float(value["seconds_per_optimizer_step"]) / 3600
        print(f"Training còn ~{hours:.2f} giờ / ~{hours * rate:.2f} CU; chưa gồm setup/eval.", flush=True)
    print("Không tự kéo dài, không chạy H2; Phase 3 chưa hoàn tất.", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("review", "status", "train", "evaluate"))
    parser.add_argument("--drive-root", default="/content/drive/MyDrive/wild_diff_icmh")
    parser.add_argument("--bundle", help="Bounded local evidence; review action only")
    parser.add_argument("--enable", action="store_true")
    parser.add_argument("--acknowledge-review", default="")
    parser.add_argument("--session-hours", type=float, default=0.75)
    parser.add_argument("--cu-rate", type=float, default=1.54)
    parser.add_argument("--detect-python", default="/content/envs/detect/bin/python")
    args = parser.parse_args(argv)
    if not math.isfinite(args.cu_rate) or args.cu_rate <= 0:
        parser.error("cu-rate must be positive and finite")
    if args.bundle and args.action != "review":
        parser.error("--bundle is only supported for CPU review")
    p = paths(args.drive_root)
    if args.action == "review":
        review(p, args.bundle)
    elif args.action == "status":
        status(p, args.cu_rate)
    elif args.action == "train":
        train(p, args.session_hours, args.cu_rate, enabled=args.enable, acknowledgement=args.acknowledge_review)
    else:
        evaluate(p, args.cu_rate, args.detect_python, enabled=args.enable, acknowledgement=args.acknowledge_review)


if __name__ == "__main__":
    main()
