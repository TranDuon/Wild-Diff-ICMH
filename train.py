"""Colab-safe Diff-ICMH training entry point.

Two checkpoint paths are intentionally separate:

* ``init_checkpoint`` is a weights-only warm start from the authors' model.
* ``resume_checkpoint`` is a Lightning checkpoint from this project and
  restores optimizer state, scheduler state, epoch and global step.
"""
from __future__ import annotations

import importlib.util
import json
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

import lightning.pytorch as pl
import torch
from omegaconf import OmegaConf

from utils.common import instantiate_from_config, load_state_dict


def _load_config(path: str):
    """Load an OmegaConf file with an optional repo-relative ``base`` file."""
    path_obj = Path(path)
    config = OmegaConf.load(path_obj)
    base = config.pop("base", None)
    if base is None:
        return config
    base_path = Path(base)
    if not base_path.is_absolute():
        candidate = path_obj.parent / base_path
        base_path = candidate if candidate.exists() else Path(base)
    return OmegaConf.merge(_load_config(str(base_path)), config)


def _torch_load(path: str):
    kwargs = {"map_location": "cpu"}
    try:
        return torch.load(path, mmap=True, weights_only=False, **kwargs)
    except TypeError:  # older torch without mmap/weights_only
        return torch.load(path, **kwargs)


def _run_data_preflight(config):
    preflight = config.get("preflight")
    if not preflight or not preflight.get("enabled", True):
        return []
    module_path = Path(__file__).resolve().parent / "tools" / "data" / "split_check.py"
    spec = importlib.util.spec_from_file_location("wild_split_check", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import data preflight from {module_path}")
    split_check = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(split_check)
    manifests = [Path(value) for value in preflight.get("manifests", [])]
    list_files = dict(preflight.get("lists", {})) or None
    build_info = preflight.get("build_info")
    rows = split_check.assert_no_leakage(manifests, list_files=list_files, build_info=build_info)
    print(f"Data preflight passed: {len(manifests)} manifest(s)")
    return rows


def _run_asset_preflight(config, rows) -> None:
    """Fail before allocating the model if images/tag sidecars are incomplete."""
    overrides = config.data.params.get("dataset_overrides", {})
    data_root = Path(str(overrides.get("data_root")))
    site_id = overrides.get("site_id")
    selected = [row for row in rows if site_id is None or row.get("site_id") == site_id]
    if not selected:
        raise ValueError(f"preflight selected no rows for site_id={site_id!r}")
    missing_images = [
        row["relative_path"] for row in selected
        if not (data_root / row["relative_path"]).is_file()
    ]
    if missing_images:
        preview = ", ".join(missing_images[:5])
        raise FileNotFoundError(f"{len(missing_images)} selected images are missing under {data_root}: {preview}")

    tags_path = overrides.get("tags_path")
    if tags_path:
        tags_path = Path(str(tags_path))
        if not tags_path.is_file():
            raise FileNotFoundError(f"RAM++ tag cache not found: {tags_path}")
        with tags_path.open("r", encoding="utf-8") as stream:
            tagged = {json.loads(line)["image_id"] for line in stream if line.strip()}
        missing_tags = [row["image_id"] for row in selected if row["image_id"] not in tagged]
        if missing_tags:
            preview = ", ".join(missing_tags[:5])
            raise ValueError(f"RAM++ tag cache is missing {len(missing_tags)} selected images: {preview}")

    detections_path = overrides.get("detections_path")
    if detections_path:
        detections_path = Path(str(detections_path))
        if not detections_path.is_file():
            raise FileNotFoundError(f"MegaDetector sidecar not found: {detections_path}")
        if detections_path.suffix.lower() == ".jsonl":
            with detections_path.open("r", encoding="utf-8") as stream:
                records = [json.loads(line) for line in stream if line.strip()]
        else:
            payload = json.loads(detections_path.read_text(encoding="utf-8"))
            records = payload.get("images", payload) if isinstance(payload, dict) else payload
        if not isinstance(records, list):
            raise ValueError("MegaDetector sidecar must contain an images list")
        detected_keys = {
            str(record.get("image_id") or record.get("id") or record.get("file"))
            for record in records if isinstance(record, dict)
        }
        uncovered = [
            row["image_id"] for row in selected
            if not any(str(row.get(key)) in detected_keys for key in ("image_id", "source_file_name", "relative_path"))
        ]
        if uncovered:
            preview = ", ".join(uncovered[:5])
            raise ValueError(f"MegaDetector sidecar has no record for {len(uncovered)} selected images: {preview}")
    print(f"Asset preflight passed: {len(selected)} image(s) for {site_id or 'all sites'}")


def _latest_checkpoint(root_dir: str) -> Optional[str]:
    checkpoint_dir = Path(root_dir) / "checkpoints"
    candidates = list(checkpoint_dir.glob("*.ckpt")) if checkpoint_dir.exists() else []
    if not candidates:
        return None
    return str(max(candidates, key=lambda path: path.stat().st_mtime))


def _resolve_resume(value: Optional[str], root_dir: str) -> Optional[str]:
    if not value:
        return None
    if value == "auto":
        latest = _latest_checkpoint(root_dir)
        if latest:
            print(f"Auto-resume selected {latest}")
        else:
            print("Auto-resume found no project checkpoint; starting a new run")
        return latest
    path = Path(value)
    if not path.is_file():
        raise FileNotFoundError(f"resume checkpoint not found: {path}")
    return str(path)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--config", default="./configs/train_kgalagadi_colab.yaml")
    parser.add_argument("--init-checkpoint", default=None, help="weights-only warm start for a new H1/H2 run")
    parser.add_argument("--resume", default=None, help="project .ckpt path, or 'auto', for full-state resume")
    parser.add_argument("--resume-codec", action="store_true", help="legacy: warm-start only preprocess_model")
    parser.add_argument("overrides", nargs="*", help="OmegaConf dot-list overrides")
    args = parser.parse_args()

    config = _load_config(args.config)
    if args.overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(args.overrides))
    OmegaConf.resolve(config)
    print(OmegaConf.to_yaml(config))

    pl.seed_everything(int(config.lightning.seed), workers=True)
    manifest_rows = _run_data_preflight(config)
    if manifest_rows:
        _run_asset_preflight(config, manifest_rows)

    data_module = instantiate_from_config(config.data)
    model_overrides = [
        value.replace("model.params.", "params.", 1)
        for value in args.overrides
        if value.startswith("model.params.")
    ]
    model_config = OmegaConf.load(config.model.config)
    if config.model.get("params"):
        model_config = OmegaConf.merge(model_config, {"params": config.model.params})
    if model_overrides:
        model_config = OmegaConf.merge(model_config, OmegaConf.from_dotlist(model_overrides))
    OmegaConf.resolve(model_config)
    model = instantiate_from_config(model_config)

    save_dir = str(config.lightning.trainer.default_root_dir)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, Path(save_dir) / "config.yaml")
    OmegaConf.save(model_config, Path(save_dir) / "config_model.yaml")

    resume_value = args.resume if args.resume is not None else config.model.get("resume_checkpoint")
    resume_path = _resolve_resume(resume_value, save_dir)
    init_path = args.init_checkpoint or config.model.get("init_checkpoint") or config.model.get("resume")
    if init_path and not resume_path:
        checkpoint = _torch_load(str(init_path))
        if args.resume_codec:
            state = checkpoint.get("state_dict", checkpoint)
            checkpoint = {
                "state_dict": {key: value for key, value in state.items() if key.startswith("preprocess_model.")}
            }
        message = load_state_dict(model, checkpoint, strict=False)
        print(f"Warm-started weights from {init_path}: {message}")
        del checkpoint
    elif init_path and resume_path:
        print("Ignoring init_checkpoint because a full-state resume checkpoint was selected")

    callbacks = [instantiate_from_config(item) for item in config.lightning.callbacks]
    trainer = pl.Trainer(callbacks=callbacks, **config.lightning.trainer)
    trainer.fit(model, datamodule=data_module, ckpt_path=resume_path)


if __name__ == "__main__":
    main()
