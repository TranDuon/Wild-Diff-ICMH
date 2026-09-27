"""Small, append-safe registry for experiment-level metric rows."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping


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
) -> list[dict[str, object]]:
    """Convert evaluator summaries into the project's canonical JSONL rows."""
    created = datetime.now(timezone.utc).isoformat()
    rows: list[dict[str, object]] = []
    for group, values in summary.items():
        if group == "all":
            illumination = "all"
        elif group.startswith("illumination="):
            illumination = group.split("=", 1)[1]
        else:
            continue
        n_images = int(values["n"])
        for name in ("bpp", "compression_ratio_rgb24", "psnr", "ssim", "foreground_ssim", "lpips"):
            value = values.get(name)
            if value is None:
                continue
            rows.append({
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
            })
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

    identity = ("exp_id", "dataset", "lambda_rate", "illumination", "ddim_steps", "metric")
    by_key: dict[tuple[object, ...], dict[str, object]] = {}
    if destination.is_file():
        with destination.open("r", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    row = json.loads(line)
                    by_key[tuple(row.get(field) for field in identity)] = row
    for row in rows:
        by_key[tuple(row.get(field) for field in identity)] = row

    ordered = sorted(
        by_key.values(),
        key=lambda row: tuple(str(row.get(field, "")) for field in identity),
    )
    with destination.open("w", encoding="utf-8", newline="\n") as stream:
        for row in ordered:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    return len(rows)
