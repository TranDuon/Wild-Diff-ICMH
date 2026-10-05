"""Aggregation and site-clustered bootstrap for per-image evaluation rows.

Pure Python so the statistics are testable without torch.  Rows are the
per-image dictionaries written by ``tools/evaluate_kgalagadi.py``.
"""
from __future__ import annotations

import math
import random
from typing import Callable, Iterable, Mapping, Sequence

# bpp and compression ratio aggregate as total bits over total pixels; every
# other metric is a plain mean over images.
RATE_METRICS = ("bpp", "compression_ratio_rgb24")
CI_LEVEL = 0.95


def _mean(values: Iterable[object]) -> float | None:
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return sum(finite) / len(finite) if finite else None


def aggregate(rows: Sequence[Mapping[str, object]], metric_names: Sequence[str]) -> dict:
    total_pixels = sum(int(row["width"]) * int(row["height"]) for row in rows)
    total_bits = sum(int(row["bitstream_bytes"]) * 8 for row in rows)
    bpp = total_bits / total_pixels if total_pixels else None
    summary: dict[str, object] = {
        "n": len(rows),
        "pixels": total_pixels,
        "bpp": bpp,
        "compression_ratio_rgb24": 24.0 / bpp if bpp else None,
        "bytes_per_image": total_bits / 8 / len(rows) if rows else None,
    }
    for name in metric_names:
        if name not in RATE_METRICS:
            summary[name] = _mean(row.get(name) for row in rows)
    return summary


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    position = q * (len(sorted_values) - 1)
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return sorted_values[low]
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (position - low)


def bootstrap_ci(
    rows: Sequence[Mapping[str, object]],
    metric_names: Sequence[str],
    *,
    n_resamples: int = 1000,
    seed: int = 0,
    level: float = CI_LEVEL,
) -> dict:
    """Percentile confidence intervals for every aggregated metric (EVAL-09).

    Resamples whole sites when the rows span several sites, because images of
    one camera are correlated; falls back to resampling images otherwise.
    """
    if not rows:
        return {"method": None, "n_resamples": 0, "intervals": {}}
    by_site: dict[str, list] = {}
    for row in rows:
        by_site.setdefault(str(row.get("site_id")), []).append(row)
    if len(by_site) > 1:
        method = "site_cluster"
        clusters = list(by_site.values())
    else:
        method = "image"
        clusters = [[row] for row in rows]

    rng = random.Random(seed)
    draws: dict[str, list[float]] = {}
    names = ["bpp", "compression_ratio_rgb24", "bytes_per_image", *[
        name for name in metric_names if name not in RATE_METRICS
    ]]
    for _ in range(n_resamples):
        sample = [row for cluster in rng.choices(clusters, k=len(clusters)) for row in cluster]
        summary = aggregate(sample, metric_names)
        for name in names:
            value = summary.get(name)
            if value is not None and math.isfinite(value):
                draws.setdefault(name, []).append(float(value))

    alpha = (1.0 - level) / 2.0
    intervals = {}
    for name, values in draws.items():
        values.sort()
        intervals[name] = [_percentile(values, alpha), _percentile(values, 1.0 - alpha)]
    return {"method": method, "n_resamples": n_resamples, "level": level, "intervals": intervals}


def group_key(illumination: str = "all", content: str = "all") -> str:
    return f"illumination={illumination}|content={content}"


def parse_group_key(key: str) -> dict[str, str] | None:
    parts = dict(part.split("=", 1) for part in key.split("|") if "=" in part)
    if set(parts) != {"illumination", "content"}:
        return None
    return parts


def stratified_groups(
    rows: Sequence[Mapping[str, object]],
    illumination_of: Callable[[Mapping[str, object]], str],
) -> dict[str, list]:
    """Groups for EVAL-05/06/07: day/night x empty/animal, plus marginals."""
    def content_of(row):
        return "empty" if row.get("is_empty") else "animal"

    groups: dict[str, list] = {}
    illuminations = ["all", *sorted({illumination_of(row) for row in rows})]
    for illumination in illuminations:
        for content in ("all", "empty", "animal"):
            members = [
                row for row in rows
                if (illumination == "all" or illumination_of(row) == illumination)
                and (content == "all" or content_of(row) == content)
            ]
            if members:
                groups[group_key(illumination, content)] = members
    return groups
