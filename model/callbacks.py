from typing import Dict, Any
import json
import os
import time
from pathlib import Path

import numpy as np
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
import torchvision
from PIL import Image
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.rank_zero import rank_zero_only

from .mixins import ImageLoggerMixin


__all__ = [
    "ModelCheckpoint",
    "ImageLogger",
    "ThroughputMonitor",
]


class ThroughputMonitor(Callback):
    """
    Measure steady-state training speed and write it to ``<default_root_dir>/<output_name>``.

    Wall-clock time of a short run is dominated by model construction and validation, so it
    overstates the cost of long runs. This callback times consecutive training batches only:
    the timer restarts after every validation loop and the first ``warmup_batches`` intervals
    are discarded. Checkpoint writes inside training stay included because long runs pay them too.
    """

    def __init__(self, output_name: str = "throughput.json", warmup_batches: int = 8) -> None:
        super().__init__()
        self.output_name = output_name
        self.warmup_batches = int(warmup_batches)
        self._last = None
        self._seen = 0
        self._intervals = []
        self._validation_runs = 0
        self._start_step = None

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._last = None
        self._seen = 0
        self._intervals = []
        self._validation_runs = 0
        self._start_step = int(trainer.global_step)

    def on_validation_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._last = None
        self._validation_runs += 1

    def on_train_batch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: STEP_OUTPUT,
        batch: Any, batch_idx: int
    ) -> None:
        now = time.perf_counter()
        if self._last is not None:
            self._seen += 1
            if self._seen > self.warmup_batches:
                self._intervals.append(now - self._last)
        self._last = now

    @rank_zero_only
    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self._intervals:
            print("ThroughputMonitor: no steady-state batches measured; nothing written", flush=True)
            return
        try:
            self._write_report(trainer)
        except Exception as exc:  # measurement must never block the final checkpoint save
            print(f"ThroughputMonitor: could not write report: {exc!r}", flush=True)

    def _write_report(self, trainer: pl.Trainer) -> None:
        accumulate = int(trainer.accumulate_grad_batches)
        median_batch = float(np.median(self._intervals))
        report = {
            "schema_version": 1,
            "batches_measured": len(self._intervals),
            "warmup_batches": self.warmup_batches,
            "validation_runs_excluded": self._validation_runs,
            "median_seconds_per_batch": median_batch,
            "mean_seconds_per_batch": float(np.mean(self._intervals)),
            "accumulate_grad_batches": accumulate,
            "seconds_per_optimizer_step": median_batch * accumulate,
            "global_step_start": self._start_step,
            "global_step_end": int(trainer.global_step),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
        output = Path(trainer.default_root_dir) / self.output_name
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + ".part")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, output)
        print(
            f"ThroughputMonitor: {report['seconds_per_optimizer_step']:.2f} s/optimizer step "
            f"({report['batches_measured']} batches) -> {output}",
            flush=True,
        )

class ImageLogger(Callback):
    """
    Log images during training or validating.
    
    TODO: Support validating.
    """
    
    def __init__(
        self,
        log_every_n_steps: int=2000,
        log_start_step: int=6000,
        max_images_each_step: int=4,
        log_images_kwargs: Dict[str, Any]=None
    ) -> "ImageLogger":
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.log_start_step = log_start_step
        self.max_images_each_step = max_images_each_step
        self.log_images_kwargs = log_images_kwargs or dict()

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        assert isinstance(pl_module, ImageLoggerMixin)

    @rank_zero_only
    def on_train_batch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: STEP_OUTPUT,
        batch: Any, batch_idx: int
    ) -> None:
        if pl_module.global_step % self.log_every_n_steps == 0 and pl_module.global_step > self.log_start_step:
            training_states = {module: module.training for module in pl_module.modules()}
            pl_module.eval()
            
            with torch.no_grad():
                # returned images should be: nchw, rgb, [0, 1]
                images, _ = pl_module.log_images(batch, **self.log_images_kwargs)
            
            # save images
            save_dir = os.path.join(pl_module.logger.save_dir, "image_log", "train")
            os.makedirs(save_dir, exist_ok=True)
            for image_key in images:
                image = images[image_key].detach().cpu()
                N = min(self.max_images_each_step, len(image))
                grid = torchvision.utils.make_grid(image[:N], nrow=4)
                # chw -> hwc (hw if gray)
                grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1).numpy()
                grid = (grid * 255).clip(0, 255).astype(np.uint8)
                filename = "{}_step-{:06}_e-{:06}_b-{:06}.png".format(
                    image_key, pl_module.global_step, pl_module.current_epoch, batch_idx
                )
                path = os.path.join(save_dir, filename)
                Image.fromarray(grid).save(path)
            
            # Restore modes without changing requires_grad. Lightning's
            # freeze()/unfreeze() would accidentally unfreeze SD/VAE/RAM++.
            for module, was_training in training_states.items():
                module.training = was_training
