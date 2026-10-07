"""Small, append-safe registry for experiment-level metric rows."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from utils.eval_stats import parse_group_key


REQUIRED_FIELDS = (
    "exp_id",
    "dataset",
    "lambda_rate",
    "illumination",
    "ddim_steps",
    "metric",
    "value",
    "n_images",
    "cu_estimate",
    "git_commit",
    "date",
)

# Metrics copied from evaluator summaries into the registry (EVAL-13).
REGISTRY_METRICS = (
    "bpp", "compression_ratio_rgb24", "bytes_per_image",
    "psnr", "ssim", "ssim_fullres", "ms_ssim", "foreground_ssim", "lpips", "dists",
    "encode_seconds", "decode_seconds",
)


def metric_rows(
    summary: Mapping[str, Mapping[str, object]],
    *,
    exp_id: str,
    dataset: str,
    lambda_rate: object,
    ddim_steps: int,
    cu_estimate: float | None,
    git_commit: str,
    method: str,
    split: str,
    site_id: str | None,
    protocol: Mapping[str, object] | None = None,
    metric_names: Sequence[str] = REGISTRY_METRICS,
) -> list[dict[str, object]]:
    """Convert evaluator summaries into the project's canonical JSONL rows.

    Accepts the stratified ``illumination=<x>|content=<y>`` groups and the
    legacy ``all`` / ``illumination=<x>`` keys.  ``protocol`` (EVAL-15) and the
    bootstrap interval, when the summary has one, are attached to every row.
    """
    created = datetime.now(timezone.utc).isoformat()
    rows: list[dict[str, object]] = []
    for group, values in summary.items():
        parsed = parse_group_key(group)
        if parsed is not None:
            illumination, subset = parsed["illumination"], parsed["content"]
        elif group == "all":
            illumination, subset = "all", "all"
        elif group.startswith("illumination="):
            illumination, subset = group.split("=", 1)[1], "all"
        else:
            continue
        n_images = int(values["n"])
        intervals = (values.get("ci") or {}).get("intervals", {})
        for name in metric_names:
            value = values.get(name)
            if value is None:
                continue
            row = {
                "exp_id": exp_id,
                "dataset": dataset,
                "lambda_rate": lambda_rate,
                "illumination": illumination,
                "ddim_steps": int(ddim_steps),
                "metric": name,
                "value": float(value),
                "n_images": n_images,
                "cu_estimate": cu_estimate,
                "git_commit": git_commit,
                "date": created,
                "method": method,
                "split": split,
                "site_id": site_id,
                "subset": subset,
            }
            if name in intervals:
                row["ci95"] = intervals[name]
                row["ci_method"] = values["ci"].get("method")
            if protocol:
                row["protocol"] = dict(protocol)
            rows.append(row)
    return rows


def upsert_jsonl(path: str | Path, new_rows: Iterable[Mapping[str, object]]) -> int:
    """Write metric rows idempotently, replacing an earlier row for the same run."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = [dict(row) for row in new_rows]
    for row in rows:
        missing = [field for field in REQUIRED_FIELDS if field not in row]
        if missing:
            raise ValueError(f"result row is missing required fields: {', '.join(missing)}")

    identity = ("exp_id", "dataset", "lambda_rate", "illumination", "subset", "ddim_steps", "metric")
    # Rows written before ``subset`` existed describe the whole split.
    defaults = {"subset": "all"}
    by_key: dict[tuple[object, ...], dict[str, object]] = {}
    if destination.is_file():
        with destination.open("r", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    row = json.loads(line)
                    by_key[tuple(row.get(field, defaults.get(field)) for field in identity)] = row
    for row in rows:
        by_key[tuple(row.get(field, defaults.get(field)) for field in identity)] = row

    ordered = sorted(
        by_key.values(),
        key=lambda row: tuple(str(row.get(field, defaults.get(field, ""))) for field in identity),
    )
    with destination.open("w", encoding="utf-8", newline="\n") as stream:
        for row in ordered:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    return len(rows)
