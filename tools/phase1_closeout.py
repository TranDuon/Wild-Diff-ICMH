"""Build the machine-readable Phase-1 evidence and calibration decision."""
from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_DATA_TOOLS = _REPO_ROOT / "tools" / "data"
if str(_DATA_TOOLS) not in sys.path:
    sys.path.insert(0, str(_DATA_TOOLS))

import split_check  # noqa: E402

from utils.checkpoint_contract import validate_project_resume_checkpoint
from utils.results_registry import REQUIRED_FIELDS


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _negative_leakage_probe(rows: list[dict]) -> str:
    if not rows:
        raise RuntimeError("cannot run leakage probe on an empty manifest")
    leaking = copy.deepcopy(rows)
    duplicate = copy.deepcopy(rows[0])
    duplicate["image_id"] = f"{duplicate['image_id']}::deliberate-leak"
    current = duplicate["split"]
    duplicate["split"] = next(split for split in ("train", "val", "test") if split != current)
    leaking.append(duplicate)
    violations = split_check.find_violations(leaking)
    sequence_violations = [item for item in violations if "sequence '" in item]
    if not sequence_violations:
        raise RuntimeError("deliberate sequence leakage was not rejected")
    return sequence_violations[0]


def _validate_registry(rows: list[dict], exp_id: str) -> dict:
    experiment_rows = [row for row in rows if row.get("exp_id") == exp_id]
    if not experiment_rows:
        raise RuntimeError(f"results registry has no rows for exp_id={exp_id!r}")
    for row in experiment_rows:
        missing = [field for field in REQUIRED_FIELDS if field not in row]
        if missing:
            raise RuntimeError(f"result row is missing required fields: {missing}")
    metrics = sorted({str(row["metric"]) for row in experiment_rows})
    required_metrics = {"bpp", "psnr", "lpips"}
    missing_metrics = sorted(required_metrics - set(metrics))
    if missing_metrics:
        raise RuntimeError(f"smoke result is missing metrics: {missing_metrics}")
    identities = [
        tuple(row.get(field) for field in (
            "exp_id", "dataset", "lambda_rate", "illumination", "ddim_steps", "metric"
        ))
        for row in experiment_rows
    ]
    if len(identities) != len(set(identities)):
        raise RuntimeError("results registry contains duplicate canonical identities")
    return {"rows": len(experiment_rows), "metrics": metrics, "idempotent": True}


def build_closeout_report(
    *,
    manifest_rows: list[dict],
    metadata_report: dict,
    result_rows: list[dict],
    exp_id: str,
    checkpoint: dict,
    checkpoint_path: str,
    resume_before: int,
    resume_after: int,
    smoke_steps: int,
    smoke_elapsed_seconds: float,
    consumed_cu_estimate: float,
    cu_per_hour: float,
    phase_budget_cu: float,
    calibration_target_steps: int,
    gpu_name: str,
    git_commit: str,
    steady_seconds_per_step: float | None = None,
    measured_cu_consumed: float | None = None,
) -> dict:
    if smoke_steps <= 0 or smoke_elapsed_seconds <= 0 or cu_per_hour <= 0:
        raise ValueError("smoke steps, elapsed seconds and CU/hour must be positive")
    if steady_seconds_per_step is not None and steady_seconds_per_step <= 0:
        raise ValueError("steady seconds per step must be positive")
    if measured_cu_consumed is not None and measured_cu_consumed < 0:
        raise ValueError("measured CU consumption cannot be negative")
    split_violations = split_check.find_violations(manifest_rows)
    if split_violations:
        raise split_check.SplitLeakageError("; ".join(split_violations[:5]))
    negative_probe = _negative_leakage_probe(manifest_rows)

    if metadata_report.get("sample_size", 0) < 100:
        raise RuntimeError("metadata evidence must cover at least 100 images")
    if metadata_report.get("datetime_usable_ratio", 0) <= 0:
        raise RuntimeError("metadata evidence has no usable datetime source")
    if metadata_report.get("location_usable_ratio", 0) <= 0:
        raise RuntimeError("metadata evidence has no usable location source")

    validate_project_resume_checkpoint(checkpoint)
    if checkpoint.get("wild_diff_compact") is not True:
        raise RuntimeError("project checkpoint is not marked compact")
    if resume_after <= resume_before:
        raise RuntimeError(
            f"resume did not advance global_step: before={resume_before}, after={resume_after}"
        )
    checkpoint_step = int(checkpoint["global_step"])
    if checkpoint_step < resume_after:
        raise RuntimeError(
            f"checkpoint global_step={checkpoint_step} is behind resume evidence {resume_after}"
        )

    registry = _validate_registry(result_rows, exp_id)
    wall_clock_seconds_per_step = smoke_elapsed_seconds / smoke_steps
    # Wall clock includes model construction and validation; prefer the steady-state
    # training speed from ThroughputMonitor when available.
    if steady_seconds_per_step is not None:
        seconds_per_step = steady_seconds_per_step
        projection_source = "measured_steady_throughput"
    else:
        seconds_per_step = wall_clock_seconds_per_step
        projection_source = "measured_smoke_extrapolation"
    projected_seconds = seconds_per_step * calibration_target_steps
    projected_cu = projected_seconds / 3600 * cu_per_hour
    consumed_cu = measured_cu_consumed if measured_cu_consumed is not None else consumed_cu_estimate
    remaining_cu = max(phase_budget_cu - consumed_cu, 0.0)
    within_budget = projected_cu <= remaining_cu
    recommendation = (
        "run_2k_allowed_not_started" if within_budget else "do_not_run_2k"
    )

    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "phase": 1,
        "status": "evidence_complete_calibration_pending" if within_budget else "evidence_complete_budget_reforecast_required",
        "git_commit": git_commit,
        "gpu": {"name": gpu_name, "cu_per_hour_estimate": cu_per_hour},
        "gates": {
            "split_positive": {"passed": True, "rows": len(manifest_rows)},
            "split_negative": {"passed": True, "evidence": negative_probe},
            "metadata_100": {
                "passed": True,
                "sample_size": metadata_report["sample_size"],
                "datetime_usable_ratio": metadata_report["datetime_usable_ratio"],
                "location_usable_ratio": metadata_report["location_usable_ratio"],
            },
            "checkpoint": {
                "passed": True,
                "path": checkpoint_path,
                "contract_version": checkpoint["wild_diff_checkpoint_contract_version"],
                "compact": True,
                "optimizer_states": len(checkpoint["optimizer_states"]),
                "global_step": checkpoint_step,
            },
            "resume": {
                "passed": True,
                "before_global_step": resume_before,
                "after_global_step": resume_after,
            },
            "results_registry": {"passed": True, "exp_id": exp_id, **registry},
        },
        "calibration": {
            "source": projection_source,
            "smoke_steps": smoke_steps,
            "smoke_elapsed_seconds": smoke_elapsed_seconds,
            "wall_clock_seconds_per_step": wall_clock_seconds_per_step,
            "steady_seconds_per_step": steady_seconds_per_step,
            "seconds_per_optimizer_step": seconds_per_step,
            "target_steps": calibration_target_steps,
            "projected_elapsed_seconds": projected_seconds,
            "projected_cu_estimate": projected_cu,
            "consumed_cu_estimate": consumed_cu_estimate,
            "measured_cu_consumed": measured_cu_consumed,
            "consumed_cu_source": (
                "colab_available_delta" if measured_cu_consumed is not None else "elapsed_time_estimate"
            ),
            "phase_budget_cu": phase_budget_cu,
            "remaining_budget_cu": remaining_cu,
            "within_remaining_budget": within_budget,
            "recommendation": recommendation,
            "note": (
                "Projected CU = projected hours x cu_per_hour (an estimate). measured_cu_consumed, "
                "when present, is the drop in Colab Resources 'Available' over the whole session."
            ),
        },
        "classification": "smoke/non-report",
    }


def _environment_versions() -> dict:
    versions = {"python": sys.version.split()[0]}
    for package in ("numpy", "scipy", "torch", "torchvision", "lightning", "pyiqa"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--metadata-report", required=True)
    parser.add_argument("--results-registry", required=True)
    parser.add_argument("--exp-id", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--resume-before", type=int, required=True)
    parser.add_argument("--resume-after", type=int, required=True)
    parser.add_argument("--smoke-steps", type=int, required=True)
    parser.add_argument("--smoke-elapsed-seconds", type=float, required=True)
    parser.add_argument("--consumed-cu-estimate", type=float, required=True)
    parser.add_argument("--cu-per-hour", type=float, required=True)
    parser.add_argument("--phase-budget-cu", type=float, default=8.0)
    parser.add_argument("--calibration-target-steps", type=int, default=2000)
    parser.add_argument("--steady-seconds-per-step", type=float, default=None,
                        help="steady-state s/optimizer step from ThroughputMonitor")
    parser.add_argument("--measured-cu-consumed", type=float, default=None,
                        help="drop in Colab Resources 'Available' since session start")
    parser.add_argument("--gpu-name", required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    import torch

    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", mmap=True, weights_only=False
    )
    report = build_closeout_report(
        manifest_rows=split_check.assert_no_leakage([Path(args.manifest)]),
        metadata_report=json.loads(Path(args.metadata_report).read_text(encoding="utf-8")),
        result_rows=_read_jsonl(args.results_registry),
        exp_id=args.exp_id,
        checkpoint=checkpoint,
        checkpoint_path=args.checkpoint,
        resume_before=args.resume_before,
        resume_after=args.resume_after,
        smoke_steps=args.smoke_steps,
        smoke_elapsed_seconds=args.smoke_elapsed_seconds,
        consumed_cu_estimate=args.consumed_cu_estimate,
        cu_per_hour=args.cu_per_hour,
        phase_budget_cu=args.phase_budget_cu,
        calibration_target_steps=args.calibration_target_steps,
        gpu_name=args.gpu_name,
        git_commit=args.git_commit,
        steady_seconds_per_step=args.steady_seconds_per_step,
        measured_cu_consumed=args.measured_cu_consumed,
    )
    report["environment"] = _environment_versions()
    output = Path(args.output)
    _atomic_write_json(output, report)
    calibration = report["calibration"]
    print("PHASE 1 EVIDENCE: OK")
    print(
        f"Dự báo {calibration['target_steps']} step: "
        f"{calibration['projected_elapsed_seconds'] / 3600:.2f} giờ, "
        f"~{calibration['projected_cu_estimate']:.2f} CU"
    )
    print("Quyết định:", calibration["recommendation"])
    print("Closeout artifact:", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

