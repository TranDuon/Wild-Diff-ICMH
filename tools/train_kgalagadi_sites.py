"""Run resumable site-specific Kgalagadi fine-tuning jobs sequentially."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def manifest_sites(path: str):
    with open(path, "r", encoding="utf-8") as stream:
        return sorted({json.loads(line)["site_id"] for line in stream if line.strip()})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_kgalagadi_colab.yaml")
    parser.add_argument("--manifest", default="data/manifests/kgalagadi_site_split.jsonl")
    parser.add_argument("--run-root", required=True, help="Drive directory holding one run per site")
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument(
        "--init-checkpoint-template",
        default=None,
        help="For H2/control, e.g. /drive/runs/h1/{site}/checkpoints/best.ckpt",
    )
    parser.add_argument("--site", action="append", default=[], help="repeat to select specific KGA:A01 sites")
    parser.add_argument("--max-sites", type=int, default=None, help="process only this many sites this session")
    parser.add_argument(
        "--include-completed", action="store_true",
        help="also launch sites that already contain checkpoints/best.ckpt",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args(argv)

    available = manifest_sites(args.manifest)
    sites = args.site or available
    unknown = sorted(set(sites) - set(available))
    if unknown:
        parser.error(f"unknown site(s): {unknown}")
    launched = 0
    for site_id in sites:
        short_site = site_id.split(":")[-1]
        run_dir = Path(args.run_root) / short_site
        if not args.include_completed and (run_dir / "checkpoints" / "best.ckpt").is_file():
            print(f"[{site_id}] skip: validated best.ckpt already exists")
            continue
        if args.max_sites is not None and launched >= args.max_sites:
            break
        environment = os.environ.copy()
        environment["KGA_SITE_ID"] = site_id
        environment["WILD_RUN_DIR"] = str(run_dir)
        command = [sys.executable, "train.py", "--config", args.config]
        init_checkpoint = args.init_checkpoint
        if args.init_checkpoint_template:
            init_checkpoint = args.init_checkpoint_template.format(site=short_site, site_id=site_id)
        if init_checkpoint:
            command.extend(["--init-checkpoint", init_checkpoint])
        command.extend(args.overrides)
        print(f"[{site_id}] {' '.join(command)}")
        launched += 1
        if not args.dry_run:
            run_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run(command, env=environment, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
