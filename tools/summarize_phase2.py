"""Summarise the Phase 2 dev results from ``results.jsonl`` for the repo.

Reads the experiment registry written by the Phase 2 Eval notebook (rows with
``exp_id = p2dev_<curve>__<point>``) and writes three files that teammates can
read without Drive access:

    <out>/phase2_points.csv   one row per (curve, point, illumination), with 95% CIs
    <out>/phase2_table.md     the illumination=all rows as a Markdown table
    <out>/rd_dev.png          the same six RD panels as Eval step 9

Usage:
    python tools/summarize_phase2.py --registry tmp/results.jsonl \
        --out .planning/phases/02-mo-rong-corpus-eval-baseline/results
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

PREFIX = "p2dev_"
IMAGE_METRICS = ("bpp", "compression_ratio_rgb24", "psnr", "ms_ssim", "lpips", "dists", "foreground_ssim")
MACHINE_METRICS = (
    "map", "ap50", "ap_medium", "ap_large", "missed_animal_rate", "hallucination_rate", "empty_fp_rate",
)
METRICS = IMAGE_METRICS + MACHINE_METRICS
CI_METRICS = ("psnr", "lpips", "map", "missed_animal_rate", "hallucination_rate", "empty_fp_rate")
ILLUMINATIONS = ("all", "day", "night")
TABLE_COLUMNS = (
    ("bpp", 4), ("psnr", 2), ("ms_ssim", 3), ("lpips", 3), ("map", 3), ("ap50", 3),
    ("missed_animal_rate", 3), ("hallucination_rate", 3), ("empty_fp_rate", 3),
)
RD_PANELS = ("psnr", "ms_ssim", "lpips", "map", "missed_animal_rate", "hallucination_rate")


def split_exp_id(exp_id: str) -> tuple[str, str]:
    curve, _, point = exp_id[len(PREFIX):].partition("__")
    return curve, point


def load_points(registry: Path) -> dict[tuple[str, str, str], dict[str, object]]:
    """Return {(curve, point, illumination): {metric: value, metric_ci: [lo, hi], n, rate}}.

    Only whole-content groups (``subset == all``) are kept. When a metric was written
    more than once, the newest row wins.
    """
    latest: dict[tuple[str, str, str], dict] = {}
    for line in registry.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        exp_id = str(row.get("exp_id", ""))
        if not exp_id.startswith(PREFIX) or row.get("subset", "all") != "all":
            continue
        if row.get("illumination") not in ILLUMINATIONS or row.get("metric") not in METRICS:
            continue
        key = (exp_id, row["illumination"], row["metric"])
        if key not in latest or str(row.get("date", "")) >= str(latest[key].get("date", "")):
            latest[key] = row

    points: dict[tuple[str, str, str], dict[str, object]] = {}
    for (exp_id, illumination, metric), row in latest.items():
        curve, point = split_exp_id(exp_id)
        entry = points.setdefault((curve, point, illumination), {})
        entry[metric] = row["value"]
        if row.get("ci95"):
            entry[f"{metric}_ci"] = list(row["ci95"])
        if metric in IMAGE_METRICS:
            entry["n_images"] = row.get("n_images")
            rate = (row.get("protocol") or {}).get("rate")
            if rate:
                entry["rate"] = rate
        elif "n_images" not in entry:
            entry["n_images"] = row.get("n_images")
    return points


def _point_order(point: str) -> tuple[int, float | str]:
    digits = "".join(ch for ch in point if ch.isdigit() or ch == ".")
    try:
        return 0, float(digits)
    except ValueError:
        return 1, point


def sorted_keys(points):
    return sorted(points, key=lambda k: (k[0], _point_order(k[1]), ILLUMINATIONS.index(k[2])))


def write_csv(points, path: Path) -> None:
    header = ["curve", "point", "illumination", "n_images", "rate", *METRICS]
    for metric in CI_METRICS:
        header += [f"{metric}_ci_low", f"{metric}_ci_high"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        for key in sorted_keys(points):
            values = points[key]
            row = [*key, values.get("n_images"), values.get("rate", "")]
            row += [values.get(metric) for metric in METRICS]
            for metric in CI_METRICS:
                ci = values.get(f"{metric}_ci") or [None, None]
                row += ci
            writer.writerow(["" if cell is None else cell for cell in row])


def _cell(value, digits: int, ci=None) -> str:
    if value is None:
        return "–"
    text = f"{value:.{digits}f}"
    if ci and None not in ci:
        text += f" [{ci[0]:.{digits}f}–{ci[1]:.{digits}f}]"
    return text


def markdown_table(points, illumination: str = "all") -> str:
    names = [name for name, _ in TABLE_COLUMNS]
    lines = [
        "| Đường | Điểm | n | " + " | ".join(names) + " |",
        "|---|---|---|" + "---|" * len(names),
    ]
    for key in sorted_keys(points):
        curve, point, group = key
        if group != illumination:
            continue
        values = points[key]
        cells = [
            _cell(values.get(name), digits, values.get(f"{name}_ci") if name in ("map", "missed_animal_rate") else None)
            for name, digits in TABLE_COLUMNS
        ]
        lines.append(f"| {curve} | {point} | {values.get('n_images', '')} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def plot_rd(points, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves: dict[str, list[dict]] = {}
    for (curve, _, illumination), values in points.items():
        if illumination == "all":
            curves.setdefault(curve, []).append(values)
    palette = plt.get_cmap("tab20").colors
    colours = {curve: palette[index % len(palette)] for index, curve in enumerate(sorted(curves))}
    figure, axes = plt.subplots(2, 3, figsize=(16, 9))
    for axis, metric in zip(axes.flat, RD_PANELS):
        for curve, values in sorted(curves.items()):
            series = sorted((v["bpp"], v[metric]) for v in values if v.get("bpp") is not None and v.get(metric) is not None)
            if series:
                style = "-o" if curve.startswith("B0") else "--."
                axis.plot(*zip(*series), style, color=colours[curve], label=curve,
                          linewidth=2.2 if curve.startswith("B0") else 1)
        axis.set_xscale("log")
        axis.set_xlabel("bpp (pixel ảnh gốc)")
        axis.set_title(metric)
        axis.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)

    points = load_points(args.registry)
    if not points:
        raise SystemExit(f"no {PREFIX}* rows in {args.registry}")
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(points, args.out / "phase2_points.csv")
    (args.out / "phase2_table.md").write_text(markdown_table(points), encoding="utf-8")
    if not args.no_plot:
        plot_rd(points, args.out / "rd_dev.png")
    curves = sorted({curve for curve, _, _ in points})
    print(f"{len({k[:2] for k in points})} points, {len(curves)} curves -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
