"""Timer-only resume semantics, runnable without the full diffusion stack."""
import ast
from copy import deepcopy
from datetime import timedelta
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


class ReferenceTimer:
    """Small public-API stand-in for Lightning 2.6.1's cumulative Timer."""

    def __init__(self, duration=None, interval="step", verbose=True):
        if isinstance(duration, str):
            try:
                days, hours, minutes, seconds = map(int, duration.split(":"))
            except ValueError as error:
                raise ValueError("invalid duration") from error
            duration = timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)
        if isinstance(duration, dict):
            duration = timedelta(**duration)
        self.duration = duration.total_seconds() if duration is not None else None
        self.elapsed = 0

    @property
    def state_key(self):
        return type(self).__qualname__

    def time_remaining(self, stage="train"):
        return None if self.duration is None else self.duration - self.elapsed

    def time_elapsed(self, stage="train"):
        return self.elapsed

    def load_state_dict(self, state_dict):
        self.elapsed = state_dict.get("time_elapsed", {}).get("train", 0)

    def on_fit_start(self, trainer, model=None):
        trainer.should_stop = self.elapsed >= self.duration


def timer_api():
    path = ROOT / "utils/session_timer.py"
    if not path.exists():
        # The current implementation is Lightning's unmodified cumulative timer.
        return SimpleNamespace(SessionTimer=ReferenceTimer, configure_session_timer=lambda callbacks, options, enabled: (callbacks, options))
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))]
    namespace = {"Timer": ReferenceTimer, "math": math, "timedelta": timedelta}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), "exec"), namespace)
    return SimpleNamespace(**namespace)


def test_exhausted_timer_resume_gets_fresh_bounded_allowance():
    api = timer_api()
    timer = api.SessionTimer(duration="00:01:30:00")
    timer.load_state_dict({"time_elapsed": {"train": 5400, "validate": 25}})
    trainer = SimpleNamespace(should_stop=False)
    timer.on_fit_start(trainer)
    assert not trainer.should_stop, "An exhausted previous session must not stop the new fit"
    assert timer.time_remaining() == 5400


def test_timer_reset_preserves_checkpoint_learning_state_and_timer_payload():
    api = timer_api()
    checkpoint = {"global_step": 862, "state_dict": {"weight": [1.25]},
                  "optimizer_states": [{"momentum": [0.7]}], "lr_schedulers": [{"last_epoch": 862}],
                  "loops": {"fit_loop": {"epoch_loop.batch_progress": 6896}},
                  "callbacks": {ReferenceTimer().state_key: {"time_elapsed": {"train": 5400}}}}
    before = deepcopy(checkpoint)
    timer = api.SessionTimer(duration="00:01:30:00")
    assert timer.state_key == ReferenceTimer().state_key
    timer.load_state_dict(checkpoint["callbacks"][timer.state_key])
    assert timer.time_elapsed() == 0
    assert checkpoint == before
    # A subsequent invocation must also reset time saved by the custom timer.
    again = api.SessionTimer(duration="00:01:30:00")
    again.load_state_dict({"time_elapsed": {"train": 5401}})
    assert again.time_remaining() == 5400


def test_only_one_timer_and_default_training_stays_cumulative():
    api = timer_api()
    other = object()
    original = {"max_time": "00:01:30:00", "max_steps": 1000}
    callbacks, options = api.configure_session_timer([other], original, enabled=False)
    assert callbacks == [other] and options == original
    legacy = ReferenceTimer(duration=options["max_time"])
    legacy.load_state_dict({"time_elapsed": {"train": 5400}})
    assert legacy.time_remaining() == 0
    callbacks, options = api.configure_session_timer([other], original, enabled=True)
    assert other in callbacks and sum(isinstance(callback, ReferenceTimer) for callback in callbacks) == 1
    assert "max_time" not in options and options["max_steps"] == 1000
    assert original["max_time"] == "00:01:30:00"


@pytest.mark.parametrize("duration", [None, "00:00:00:00", timedelta(seconds=-1), {}])
def test_session_flag_requires_positive_finite_time_limit(duration):
    api = timer_api()
    with pytest.raises(ValueError, match="positive|finite|max_time"):
        api.configure_session_timer([], {"max_time": duration}, enabled=True)


def test_session_flag_rejects_ambiguous_existing_timer():
    api = timer_api()
    with pytest.raises(ValueError, match="Timer"):
        api.configure_session_timer([ReferenceTimer(duration="00:00:01:00")],
                                    {"max_time": "00:01:30:00"}, enabled=True)


def test_training_opt_in_does_not_replace_full_state_resume():
    tree = ast.parse((ROOT / "train.py").read_text(encoding="utf-8"))
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    source = ast.unparse(main)
    assert "--session-time-limit" in source
    assert "enabled=args.session_time_limit" in source
    assert "ckpt_path=resume_path" in source
    assert "weights_only=True" not in source


def test_real_lightning_cpu_fit_save_resume_retains_learning_state(tmp_path):
    """Real Trainer restore, including repeated exhausted-Timer resumes, no GPU."""
    torch = pytest.importorskip("torch")
    pl = pytest.importorskip("lightning.pytorch")
    from lightning.pytorch.callbacks import Callback, Timer
    from utils.session_timer import SessionTimer, configure_session_timer

    class TinyModel(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1, 1)

        def training_step(self, batch, batch_idx):
            return (self.linear(batch) - 1).square().mean()

        def configure_optimizers(self):
            optimizer = torch.optim.SGD(self.parameters(), lr=.05, momentum=.9)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=.9)
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    class RestoreProbe(Callback):
        def __init__(self, expected):
            self.expected = expected
            self.observed = False

        def on_train_start(self, trainer, model):
            saved = self.expected
            assert trainer.global_step == saved["global_step"]
            for key, value in model.state_dict().items():
                assert torch.equal(value, saved["state_dict"][key])
            optimizer = trainer.optimizers[0].state_dict()
            assert optimizer["param_groups"] == saved["optimizer_states"][0]["param_groups"]
            for key, item in optimizer["state"].items():
                assert torch.equal(item["momentum_buffer"], saved["optimizer_states"][0]["state"][key]["momentum_buffer"])
            assert trainer.lr_scheduler_configs[0].scheduler.state_dict() == saved["lr_schedulers"][0]
            self.observed = True

    def make_trainer(callbacks, target, session=False):
        options = {"accelerator": "cpu", "devices": 1, "max_steps": target, "max_epochs": -1,
                   "max_time": "00:00:01:00", "logger": False, "enable_checkpointing": False,
                   "enable_model_summary": False, "enable_progress_bar": False,
                   "num_sanity_val_steps": 0, "default_root_dir": str(tmp_path)}
        callbacks, options = configure_session_timer(callbacks, options, enabled=session)
        trainer = pl.Trainer(callbacks=callbacks, **options)
        assert sum(isinstance(callback, Timer) for callback in trainer.callbacks) == 1
        return trainer

    data = torch.utils.data.DataLoader(torch.ones(40, 1), batch_size=2)
    first = make_trainer([], 2)
    first.fit(TinyModel(), train_dataloaders=data)
    path = tmp_path / "original.ckpt"
    first.save_checkpoint(path)
    saved = torch.load(path, weights_only=False)
    saved["callbacks"][Timer().state_key]["time_elapsed"]["train"] = 60
    torch.save(saved, path)
    original_bytes = path.read_bytes()
    # Reproduce the bug with an unchanged Trainer: depleted budget stops at step 2.
    legacy = make_trainer([], 4)
    legacy.fit(TinyModel(), train_dataloaders=data, ckpt_path=path)
    assert legacy.global_step == 2
    for target in (4, 6):
        probe = RestoreProbe(saved)
        resumed = make_trainer([probe], target, session=True)
        resumed.fit(TinyModel(), train_dataloaders=data, ckpt_path=path)
        assert probe.observed and resumed.global_step == target
        assert any(isinstance(callback, SessionTimer) for callback in resumed.callbacks)
        assert path.read_bytes() == original_bytes
        output = tmp_path / f"resumed-{target}.ckpt"
        resumed.save_checkpoint(output)
        saved = torch.load(output, weights_only=False)
        assert saved["global_step"] == target and "loops" in saved
        saved["callbacks"][Timer().state_key]["time_elapsed"]["train"] = 60
        torch.save(saved, output)
        path = output
        original_bytes = path.read_bytes()
