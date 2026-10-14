"""Scalar-only evidence and a separate identity for the bounded H2 ROI pilot.

Reuse the proven H1 callback's batch hooks, never its CSV header or checkpoint
identity. This observes metrics from the existing forward pass; missing values
remain blank instead of becoming fabricated zeros.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re

from utils.checkpoint_contract import validate_project_resume_checkpoint
from utils.training_diagnostics import COMPONENTS as H1_COMPONENTS
from utils.training_diagnostics import LossCSV, validate_optimizer_groups


CHECKPOINT_IDENTITY_KEY = "wild_diff_h2_roi_contract_sha256"
MAX_STEPS = 200
COMPONENTS = (*H1_COMPONENTS, "l_guide_roi", "l_guide_bg", "roi_fraction")
FIELDS = ("session", "time_utc", "kind", "global_step", "main_lr", "entropy_aux_lr",
          "val/objective", *(f"{prefix}/{key}" for prefix in ("T", "V") for key in COMPONENTS))


def _step(value):
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= MAX_STEPS:
        raise RuntimeError("H2 global_step must be an integer between 0 and 200")
    return value


def validate_h2_checkpoint(checkpoint, expected_sha256):
    """Validate already-loaded, user-trusted checkpoint data without mutating it."""
    validate_project_resume_checkpoint(checkpoint)
    if checkpoint.get(CHECKPOINT_IDENTITY_KEY) != expected_sha256:
        raise RuntimeError("H2 checkpoint identity mismatch")
    step = _step(checkpoint.get("global_step"))
    optimizers = checkpoint.get("optimizer_states", [])
    if len(optimizers) != 1:
        raise RuntimeError("H2 requires one optimizer with exactly two named parameter groups")
    validate_optimizer_groups(optimizers[0].get("param_groups", []))
    return step


class H2LossCSV(LossCSV):
    """ROI/background scalars, actual optimizer rates and source-bound identity.

    Explicit paths must stay inside ``trainer.default_root_dir``. The caller can
    supply the frozen digest; otherwise the first fit/load/save binds the current
    contract once, and every later operation checks that exact byte identity.
    """

    def __init__(self, output_path=None, every_n_steps=10, contract_path=None,
                 contract_sha256=None):
        super().__init__(every_n_steps=every_n_steps)
        if contract_sha256 is not None and not re.fullmatch(r"[0-9a-f]{64}", contract_sha256):
            raise ValueError("contract_sha256 must be a lowercase SHA256 digest")
        self.output_path = Path(output_path) if output_path is not None else None
        self.contract_path = Path(contract_path) if contract_path is not None else None
        self.contract_sha256 = contract_sha256

    @staticmethod
    def _contained_path(trainer, supplied, filename):
        root = Path(trainer.default_root_dir).resolve()
        path = supplied if supplied is not None else Path(filename)
        path = (path if path.is_absolute() else root / path).resolve()
        if not path.is_relative_to(root) or path == root:
            raise RuntimeError("H2 diagnostics path must be contained in its run directory")
        return path

    def _identity(self, trainer):
        path = self._contained_path(trainer, self.contract_path, "experiment_contract.json")
        if not path.is_file():
            raise RuntimeError("H2 experiment contract missing before fit/checkpoint")
        payload = path.read_bytes()
        try:
            value = json.loads(payload)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise RuntimeError("H2 experiment contract is not valid JSON") from exc
        if not isinstance(value, dict) or not value:
            raise RuntimeError("H2 experiment contract must be a nonempty JSON object")
        digest = hashlib.sha256(payload).hexdigest()
        if self.contract_sha256 is None:
            self.contract_sha256 = digest
        elif digest != self.contract_sha256:
            raise RuntimeError("H2 experiment contract file hash changed")
        return digest

    def _output(self, trainer):
        return self._contained_path(trainer, self.output_path, "loss_components.csv")

    def on_fit_start(self, trainer, pl_module):
        self._identity(trainer)
        maximum = trainer.max_steps
        if isinstance(maximum, bool) or not isinstance(maximum, int) or not 1 <= maximum <= MAX_STEPS:
            raise RuntimeError("H2 Trainer.max_steps must be bounded between 1 and 200")
        self._last_step = _step(int(trainer.global_step))
        path = self._output(trainer)
        if path.exists() and path.stat().st_size:
            with path.open(newline="", encoding="utf-8") as stream:
                if tuple(next(csv.reader(stream))) != FIELDS:
                    raise RuntimeError("H2 loss CSV header changed; preserve prior evidence")

    def _write(self, trainer, kind):
        if not trainer.is_global_zero:
            return
        self._identity(trainer)
        step = _step(int(trainer.global_step))
        rates = self._groups(trainer)
        row = {"session": self._session, "time_utc": datetime.now(timezone.utc).isoformat(),
               "kind": kind, "global_step": step,
               "main_lr": rates["main"], "entropy_aux_lr": rates["entropy_aux"]}
        metrics = trainer.callback_metrics
        prefix = "T/" if kind == "train" else "V/"
        for name in FIELDS[6:]:
            if name == "val/objective":
                if kind != "validation":
                    continue
            elif not name.startswith(prefix):
                continue
            candidates = (name + "_step", name) if kind == "train" else (name + "_epoch", name)
            value = next((metrics[key] for key in candidates if key in metrics), None)
            if value is None:
                continue
            try:
                scalar = float(value.detach().cpu() if hasattr(value, "detach") else value)
            except (ValueError, TypeError):
                continue
            if math.isfinite(scalar):
                row[name] = scalar
        path = self._output(trainer)
        path.parent.mkdir(parents=True, exist_ok=True)
        new = not path.exists() or not path.stat().st_size
        with path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            if new:
                writer.writeheader()
            writer.writerow(row)
            stream.flush()
        if kind == "train":
            self._last_written = step

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        _step(int(trainer.global_step))
        self._groups(trainer)
        checkpoint[CHECKPOINT_IDENTITY_KEY] = self._identity(trainer)

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        validate_h2_checkpoint(checkpoint, self._identity(trainer))
