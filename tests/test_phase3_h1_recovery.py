"""Recovery gates must not alter the paid-for pilot experiment."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def api():
    if not (ROOT / "tools/phase3_h1_recovery.py").exists():
        # Intentional RED: the previous notebook has no review-bound gate.
        return SimpleNamespace(validate_acknowledgement=lambda *a, **k: None)
    return importlib.import_module("tools.phase3_h1_recovery")


def test_training_requires_explicit_user_enablement(tmp_path):
    with pytest.raises(RuntimeError, match="explicit"):
        api().validate_acknowledgement(tmp_path / "review.json", "", enabled=False)


def test_mismatched_or_unverified_review_is_rejected(tmp_path):
    module = api()
    path = tmp_path / "review.json"
    path.write_text(json.dumps({"schema_version": 1, "protocol_verified": False,
                               "training_allowed": True, "recommendation": "conditional_low_lr_trial"}))
    if not hasattr(module, "digest"):
        pytest.skip("RED gate covered by explicit-consent test")
    with pytest.raises(RuntimeError, match="verified"):
        module.validate_acknowledgement(path, module.digest(path), enabled=True)
    with pytest.raises(RuntimeError, match="hash"):
        module.validate_acknowledgement(path, "f" * 64, enabled=True)


def report(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema_version": 1, "evidence_sha256": "a" * 64,
                               "protocol_verified": True, "training_allowed": True,
                               "recommendation": "conditional_low_lr_trial",
                               "selection": {"image_ids": [str(i) for i in range(30)]}}))
    return api().digest(path)


def test_recovery_paths_do_not_collide_with_old_run(tmp_path):
    module = api()
    p = module.paths(tmp_path)
    from tools import phase3_h1 as old
    old_paths = old.paths(tmp_path)
    assert p["run"] != old_paths["run"]
    assert p["phase"] != old_paths["phase"]
    assert p["quick"] == old_paths["quick"]
    assert p["run"].parent.name == "h1_low_lr_recovery_ls512"


def fixture_paths(tmp_path, monkeypatch):
    module = api()
    p = module.paths(tmp_path)
    p["checkpoints"] = tmp_path / "trusted_authors"
    author = p["checkpoints"] / module.AUTHOR_RELATIVE
    author.parent.mkdir(parents=True)
    author.write_bytes(b"trusted author weights")
    p["quick"].parent.mkdir(parents=True)
    p["quick"].write_text("\n".join(str(i) for i in range(30)) + "\n")
    (p["quick"].parent / "protocol.json").write_text(json.dumps(module.PROTOCOL))
    ack = report(p["review"])
    monkeypatch.setattr(module, "prepare_assets", lambda p: None)
    return module, p, author, ack


def test_initial_train_uses_weights_only_and_is_capped(tmp_path, monkeypatch):
    module, p, author, ack = fixture_paths(tmp_path, monkeypatch)
    commands = []
    monkeypatch.setattr(module, "run_logged", lambda command, *a: commands.append(command))
    monkeypatch.setattr(module, "checkpoint_step", lambda *a: 200)
    module.train(p, 1.0, 1.54, enabled=True, acknowledgement=ack)
    command = [str(v) for v in commands[0]]
    assert command[command.index("--init-checkpoint") + 1] == str(author)
    assert "--resume" not in command  # config disables automatic resume initially
    assert "--session-time-limit" in command
    assert "lightning.trainer.max_steps=200" in command
    contract = json.loads((p["run"] / "experiment_contract.json").read_text())
    assert contract["optimizer_lrs"] == {"main": 1e-5, "entropy_aux": 1e-3}
    assert contract["author_sha256"] == module.digest(author)


def test_only_own_valid_checkpoint_can_resume(tmp_path, monkeypatch):
    module, p, author, ack = fixture_paths(tmp_path, monkeypatch)
    commands = []
    monkeypatch.setattr(module, "run_logged", lambda command, *a: commands.append(command))
    monkeypatch.setattr(module, "checkpoint_step", lambda *a: 200)
    module.train(p, 1.0, 1.54, enabled=True, acknowledgement=ack)
    last = p["run"] / "checkpoints/last.ckpt"
    last.parent.mkdir(parents=True)
    last.touch()
    steps = iter([100, 200])
    monkeypatch.setattr(module, "checkpoint_step", lambda *a: next(steps))
    commands.clear()
    module.train(p, 1.0, 1.54, enabled=True, acknowledgement=ack)
    command = [str(v) for v in commands[0]]
    assert command[command.index("--resume") + 1] == str(last)
    assert "--init-checkpoint" not in command


def test_completed_trial_does_not_retrain(tmp_path, monkeypatch):
    module, p, author, ack = fixture_paths(tmp_path, monkeypatch)
    monkeypatch.setattr(module, "run_logged", lambda *a: None)
    monkeypatch.setattr(module, "checkpoint_step", lambda *a: 200)
    module.train(p, 1.0, 1.54, enabled=True, acknowledgement=ack)
    last = p["run"] / "checkpoints/last.ckpt"
    last.parent.mkdir(parents=True)
    last.touch()
    monkeypatch.setattr(module, "run_logged", lambda *a: pytest.fail("Must not retrain"))
    module.train(p, 1.0, 1.54, enabled=True, acknowledgement=ack)


def test_checkpoint_identity_and_learning_rates_fail_closed():
    module = api()
    checkpoint = {"wild_diff_checkpoint_contract_version": 2, "global_step": 100,
                  module.CHECKPOINT_IDENTITY_KEY: "a" * 64,
                  "optimizer_states": [{"param_groups": [
                      {"name": "main", "lr": 1e-5}, {"name": "entropy_aux", "lr": 1e-3}]}]}
    before = deepcopy(checkpoint)
    assert module.validate_recovery_checkpoint(checkpoint, "a" * 64) == 100
    assert checkpoint == before
    with pytest.raises(RuntimeError, match="identity"):
        module.validate_recovery_checkpoint(checkpoint, "b" * 64)
    checkpoint["optimizer_states"][0]["param_groups"][0]["lr"] = 1e-4
    with pytest.raises(RuntimeError, match="learning rate"):
        module.validate_recovery_checkpoint(checkpoint, "a" * 64)


@pytest.mark.parametrize("hours", [0, float("nan"), float("inf"), 3, 0.00001])
def test_invalid_session_rejected_without_asset_setup(tmp_path, monkeypatch, hours):
    module = api()
    monkeypatch.setattr(module, "prepare_assets", lambda p: pytest.fail("No heavy preparation"))
    with pytest.raises(ValueError, match="Session"):
        module.train(module.paths(tmp_path), hours, 1.54, enabled=True, acknowledgement="a" * 64)


def test_config_overrides_main_lr_only_without_roi():
    from tests.test_phase3_h1 import load_config
    value = load_config(ROOT / api().CONFIG)
    params = value["model"]["params"]
    assert float(params["learning_rate"]) == 1e-5
    assert float(params["aux_learning_rate"]) == 1e-3
    assert not params["roi_loss_enabled"] and not params["roi_semantic_enabled"]
    assert value["data"]["params"]["dataset_overrides"]["bbox_crop_probability"] == 0
    assert value["lightning"]["trainer"]["max_steps"] == 200
    assert value["model"]["resume_checkpoint"] is None
    assert any(c["target"] == "utils.training_diagnostics.LossCSV" for c in value["lightning"]["callbacks"])


def test_review_selection_cannot_silently_change(tmp_path, monkeypatch):
    module, p, author, ack = fixture_paths(tmp_path, monkeypatch)
    p["quick"].write_text("\n".join(str(i + 1) for i in range(30)) + "\n")
    monkeypatch.setattr(module, "run_logged", lambda *a: pytest.fail("No GPU before identity check"))
    with pytest.raises(RuntimeError, match="frozen dev30"):
        module.train(p, .5, 1.54, enabled=True, acknowledgement=ack)


def test_evaluation_needs_consent_and_exact_completed_trial(tmp_path, monkeypatch):
    module, p, author, ack = fixture_paths(tmp_path, monkeypatch)
    monkeypatch.setattr(module, "run_logged", lambda *a: None)
    monkeypatch.setattr(module, "checkpoint_step", lambda *a: 200)
    module.train(p, .5, 1.54, enabled=True, acknowledgement=ack)
    last = p["run"] / "checkpoints/last.ckpt"
    last.parent.mkdir(parents=True)
    last.touch()
    monkeypatch.setattr(module, "run_logged", lambda *a: pytest.fail("No GPU before eval gate"))
    with pytest.raises(RuntimeError, match="explicit"):
        module.evaluate(p, 1.54, "detector-python")
    monkeypatch.setattr(module, "checkpoint_step", lambda *a: 199)
    with pytest.raises(RuntimeError, match="exactly 200"):
        module.evaluate(p, 1.54, "detector-python", enabled=True, acknowledgement=ack)


def test_status_is_metadata_only(tmp_path, monkeypatch):
    module = api()
    monkeypatch.setattr(module, "checkpoint_step", lambda *a: pytest.fail("No multi-GB checkpoint reads"))
    monkeypatch.setattr(module, "run_logged", lambda *a: pytest.fail("No subprocess"))
    module.status(module.paths(tmp_path), 1.54)


def test_evaluation_binds_archive_source_before_cached_review(tmp_path, monkeypatch):
    module, p, author, ack = fixture_paths(tmp_path, monkeypatch)
    monkeypatch.setattr(module, "run_logged", lambda *a: None)
    monkeypatch.setattr(module, "checkpoint_step", lambda *a: 200)
    module.train(p, .5, 1.54, enabled=True, acknowledgement=ack)
    last = p["run"] / "checkpoints/last.ckpt"
    last.parent.mkdir(parents=True)
    last.touch()
    folder = p["phase"] / "archive/step_000200_quick_ddim50"
    folder.mkdir(parents=True)
    (folder / "run_info.json").write_text('{"actual_decoder_metadata":true}')
    monkeypatch.setattr(module.old, "snapshot", lambda p: (last, 200))
    monkeypatch.setattr(module.old, "assert_archive_complete", lambda *a: None)
    from tools import review_h1_cached as audit
    loaded = []

    def load_actual_source(root, trial_dir):
        source = json.loads((trial_dir / "review_source.json").read_text())
        assert Path(source["archive"]) == folder
        assert (folder / "evaluation_contract.json").is_file()
        assert (trial_dir / "evaluation_contract.json").is_file()
        loaded.append(source)
        return {"trial": "cached"}

    monkeypatch.setattr(audit, "load_drive_bundle", load_actual_source)
    def review_with_actual_trial_losses(value, *, logs_dir):
        assert logs_dir == p["run"]  # never read old-pilot loss curves as trial evidence
        return {"points": value}
    monkeypatch.setattr(audit, "review_bundle", review_with_actual_trial_losses)
    module.evaluate(p, 1.54, "detector-python", enabled=True, acknowledgement=ack)
    assert loaded and (p["phase"] / "eval/step_000200_quick/done.json").is_file()


def test_scoring_retry_preserves_complete_decode_and_source_metadata(tmp_path, monkeypatch):
    module, p, author, ack = fixture_paths(tmp_path, monkeypatch)
    monkeypatch.setattr(module, "run_logged", lambda *a: None)
    monkeypatch.setattr(module, "checkpoint_step", lambda *a: 200)
    module.train(p, .5, 1.54, enabled=True, acknowledgement=ack)
    last = p["run"] / "checkpoints/last.ckpt"
    last.parent.mkdir(parents=True)
    last.touch()
    folder = p["phase"] / "archive/step_000200_quick_ddim50"
    folder.mkdir(parents=True)
    original_metadata = '{"created_at":"first_decode_time"}'
    (folder / "run_info.json").write_text(original_metadata)
    monkeypatch.setattr(module.old, "snapshot", lambda p: (last, 200))
    monkeypatch.setattr(module.old, "assert_archive_complete", lambda *a: None)
    commands = []

    def fail_first_quality(command, *args):
        commands.append(command)
        if "inference_partition.py" in command:
            pytest.fail("Complete archive must never be decoded again")
        if len(commands) == 1:
            raise RuntimeError("simulated quality failure")

    monkeypatch.setattr(module, "run_logged", fail_first_quality)
    from tools import review_h1_cached as audit
    monkeypatch.setattr(audit, "load_drive_bundle", lambda *a, **k: {})
    monkeypatch.setattr(audit, "review_bundle", lambda value, **kwargs: {"points": {}})
    with pytest.raises(RuntimeError, match="quality failure"):
        module.evaluate(p, 1.54, "detector-python", enabled=True, acknowledgement=ack)
    module.evaluate(p, 1.54, "detector-python", enabled=True, acknowledgement=ack)
    output = p["phase"] / "eval/step_000200_quick"
    assert (output / "run_info.json").read_text() == original_metadata
    assert (output / "done.json").is_file()


def test_no_expansion_or_foreign_resume_cli_options():
    for args in (["train", "--target-steps", "1000"], ["train", "--resume", "old.ckpt"],
                 ["evaluate", "--scope", "full"], ["evaluate", "--allow-full"]):
        with pytest.raises(SystemExit):
            api().main(args)


def test_loss_csv_records_available_components_once_per_step(tmp_path):
    pytest.importorskip("lightning.pytorch")
    from utils.training_diagnostics import LossCSV, CHECKPOINT_IDENTITY_KEY
    import csv
    path = tmp_path / "experiment_contract.json"
    path.write_text('{"isolated": true}')
    trainer = SimpleNamespace(default_root_dir=str(tmp_path), global_step=0,
                              is_global_zero=True, sanity_checking=False,
                              optimizers=[SimpleNamespace(param_groups=[
                                  {"name": "main", "lr": 1e-5},
                                  {"name": "entropy_aux", "lr": 1e-3}])],
                              callback_metrics={"T/l_bpp_step": .012, "T/l_guide_step": 2.5,
                                                "val/objective": 3.0})
    callback = LossCSV(every_n_steps=10)
    callback.on_fit_start(trainer, None)
    trainer.global_step = 10
    for _ in range(8):
        callback.on_train_batch_end(trainer, None, None, None, 0)
    callback.on_validation_end(trainer, None)
    callback.on_train_end(trainer, None)
    rows = list(csv.DictReader((tmp_path / "loss_components.csv").open(newline="", encoding="utf-8")))
    assert len(rows) == 2
    assert rows[0]["T/l_bpp"] == "0.012" and rows[0]["T/l_semantic"] == ""
    assert rows[0]["main_lr"] == "1e-05" and rows[0]["entropy_aux_lr"] == "0.001"
    assert rows[1]["val/objective"] == "3.0" and rows[1]["T/l_bpp"] == ""
    payload = {}
    callback.on_save_checkpoint(trainer, None, payload)
    assert payload[CHECKPOINT_IDENTITY_KEY] == api().digest(path)
    callback.on_load_checkpoint(trainer, None, payload)
    with pytest.raises(RuntimeError, match="identity"):
        callback.on_load_checkpoint(trainer, None, {})


def test_loss_csv_real_lightning_fit_and_own_resume(tmp_path):
    pl = pytest.importorskip("lightning.pytorch")
    torch = pytest.importorskip("torch")
    from torch.utils.data import DataLoader, TensorDataset
    from utils.training_diagnostics import LossCSV, CHECKPOINT_IDENTITY_KEY
    from utils.session_timer import SessionTimer
    import csv

    class Tiny(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.main_weight = torch.nn.Parameter(torch.tensor(1.0))
            self.aux_weight = torch.nn.Parameter(torch.tensor(.1))

        def training_step(self, batch, batch_idx):
            loss = self.main_weight.square() + self.aux_weight.square()
            self.log_dict({"T/l_bpp": self.main_weight.square(), "T/l_guide": self.aux_weight.square(),
                           "T/optim_loss": loss}, on_step=True, on_epoch=True)
            return loss

        def validation_step(self, batch, batch_idx):
            self.log("val/objective", self.main_weight.square(), on_epoch=True, batch_size=1)

        def configure_optimizers(self):
            return torch.optim.AdamW([{"params": [self.main_weight], "name": "main", "lr": 1e-5},
                                      {"params": [self.aux_weight], "name": "entropy_aux", "lr": 1e-3}])

        def on_save_checkpoint(self, checkpoint):
            checkpoint["wild_diff_checkpoint_contract_version"] = 2

    (tmp_path / "experiment_contract.json").write_text('{"tiny_cpu_test":true}')
    loader = DataLoader(TensorDataset(torch.ones(12, 1)), batch_size=1)
    def trainer(max_steps):
        return pl.Trainer(default_root_dir=str(tmp_path), accelerator="cpu", max_steps=max_steps,
                          max_epochs=-1, logger=False, enable_progress_bar=False,
                          enable_model_summary=False, enable_checkpointing=False,
                          num_sanity_val_steps=0, val_check_interval=4, check_val_every_n_epoch=None,
                          limit_val_batches=1, accumulate_grad_batches=2,
                          callbacks=[LossCSV(every_n_steps=1), SessionTimer(duration="00:00:01:00")])
    first = trainer(2)
    first.fit(Tiny(), loader, loader)
    checkpoint = tmp_path / "tiny_cpu.ckpt"
    first.save_checkpoint(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert api().validate_recovery_checkpoint(payload, api().digest(tmp_path / "experiment_contract.json")) == 2
    assert payload[CHECKPOINT_IDENTITY_KEY]
    second = trainer(3)
    second.fit(Tiny(), loader, loader, ckpt_path=str(checkpoint))
    assert second.global_step == 3
    rows = list(csv.DictReader((tmp_path / "loss_components.csv").open(newline="", encoding="utf-8")))
    training = [row for row in rows if row["kind"] == "train"]
    assert [int(row["global_step"]) for row in training] == [1, 2, 3]
    assert all(row["T/l_bpp"] and row["main_lr"] == "1e-05" for row in training)
