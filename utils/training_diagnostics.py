"""Opt-in scalar CSV and identity stamp for the isolated H1 recovery run.

No second forward pass, image sampling or change to the objective. Blank CSV
cells mean the model did not log that component (not a measured zero).
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path
import uuid

from lightning.pytorch.callbacks import Callback

CHECKPOINT_IDENTITY_KEY = "wild_diff_h1_recovery_contract_sha256"
EXPECTED_LRS = {"main": 1e-5, "entropy_aux": 1e-3}
COMPONENTS = ("l_simple", "l_bpp", "q_bpp", "sem_bpp", "tag_bpp",
              "total_bpp", "l_guide", "l_semantic", "loss", "aux_loss", "optim_loss")
FIELDS = ("session", "time_utc", "kind", "global_step", "main_lr", "entropy_aux_lr",
          "val/objective", *(f"{prefix}/{key}" for prefix in ("T", "V") for key in COMPONENTS))


def validate_optimizer_groups(groups):
    """Names as well as numeric rates are part of the experiment contract."""
    names = [group.get("name") for group in groups]
    if len(names) != len(set(names)) or set(names) != set(EXPECTED_LRS):
        raise RuntimeError("Recovery optimizer must have exactly main and entropy_aux groups")
    for group in groups:
        lr = float(group.get("lr", float("nan")))
        if not math.isfinite(lr) or not math.isclose(lr, EXPECTED_LRS[group["name"]], rel_tol=1e-9, abs_tol=0):
            raise RuntimeError(f"Recovery optimizer learning rate mismatch for {group['name']}: {lr}")


class LossCSV(Callback):
    """Sparse current-batch scalars, observed LRs and immutable checkpoint ID."""

    def __init__(self, every_n_steps=10, output_name="loss_components.csv"):
        super().__init__()
        if not isinstance(every_n_steps, int) or every_n_steps < 1:
            raise ValueError("every_n_steps must be a positive integer")
        if Path(output_name).name != output_name:
            raise ValueError("output_name must be a filename")
        self.every_n_steps = every_n_steps
        self.output_name = output_name
        self._last_step = -1
        self._last_written = -1
        self._session = uuid.uuid4().hex

    @staticmethod
    def _identity(trainer):
        contract = Path(trainer.default_root_dir) / "experiment_contract.json"
        if not contract.is_file():
            raise RuntimeError("Recovery experiment contract missing before checkpoint save")
        return hashlib.sha256(contract.read_bytes()).hexdigest()

    @staticmethod
    def _groups(trainer):
        if len(trainer.optimizers) != 1:
            raise RuntimeError("Recovery requires one optimizer with two named parameter groups")
        groups = trainer.optimizers[0].param_groups
        validate_optimizer_groups(groups)
        return {group["name"]: float(group["lr"]) for group in groups}

    def on_fit_start(self, trainer, pl_module):
        self._identity(trainer)
        self._last_step = int(trainer.global_step)
        path = Path(trainer.default_root_dir) / self.output_name
        if path.exists() and path.stat().st_size:
            with path.open(newline="", encoding="utf-8") as stream:
                if tuple(next(csv.reader(stream))) != FIELDS:
                    raise RuntimeError("Recovery loss CSV header changed; preserve prior evidence")

    def on_train_start(self, trainer, pl_module):
        # Lightning restores loop/optimizer state after on_fit_start. Initial
        # counters here prevent an accumulation-only resumed batch from being
        # mislabeled as an optimizer update at the restored global step.
        self._groups(trainer)
        self._last_step = int(trainer.global_step)
        self._last_written = int(trainer.global_step)

    def _write(self, trainer, kind):
        if not trainer.is_global_zero:
            return
        rates = self._groups(trainer)
        row = {"session": self._session, "time_utc": datetime.now(timezone.utc).isoformat(),
               "kind": kind, "global_step": int(trainer.global_step),
               "main_lr": rates["main"], "entropy_aux_lr": rates["entropy_aux"]}
        metrics = trainer.callback_metrics
        for name in FIELDS[6:]:
            prefix = "T/" if kind == "train" else "V/"
            if name != "val/objective" and not name.startswith(prefix):
                continue
            if name == "val/objective" and kind != "validation":
                continue
            # Training uses step values, never stale epoch aggregates.
            candidates = (name + "_step", name) if kind == "train" else (name + "_epoch", name)
            value = next((metrics[key] for key in candidates if key in metrics), None)
            if value is not None:
                try:
                    scalar = float(value.detach().cpu() if hasattr(value, "detach") else value)
                except (ValueError, TypeError):
                    continue
                if math.isfinite(scalar):
                    row[name] = scalar
        path = Path(trainer.default_root_dir) / self.output_name
        path.parent.mkdir(parents=True, exist_ok=True)
        new = not path.exists() or not path.stat().st_size
        with path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            if new:
                writer.writeheader()
            writer.writerow(row)
            stream.flush()
        if kind == "train":
            self._last_written = int(trainer.global_step)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = int(trainer.global_step)
        if step > self._last_step:
            self._last_step = step
            if step % self.every_n_steps == 0:
                self._write(trainer, "train")

    def on_validation_end(self, trainer, pl_module):
        if not trainer.sanity_checking:
            self._write(trainer, "validation")

    def on_train_end(self, trainer, pl_module):
        if int(trainer.global_step) > 0 and int(trainer.global_step) != self._last_written:
            self._write(trainer, "train")

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint[CHECKPOINT_IDENTITY_KEY] = self._identity(trainer)

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        if checkpoint.get(CHECKPOINT_IDENTITY_KEY) != self._identity(trainer):
            raise RuntimeError("Recovery checkpoint identity mismatch")
