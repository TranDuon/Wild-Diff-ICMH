"""Isolated H2 CSV/identity hooks and a real, tiny CPU Lightning resume."""
import csv
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
pl = pytest.importorskip("lightning.pytorch")
torch = pytest.importorskip("torch")
from utils.h2_training_diagnostics import CHECKPOINT_IDENTITY_KEY, FIELDS, H2LossCSV, validate_h2_checkpoint


def contract(tmp_path):
    path = tmp_path / "experiment_contract.json"
    path.write_text(json.dumps({"experiment": "H2_roi_v1_alpha2_200", "target_steps": 200,
                               "optimizer_lrs": {"main": 1e-5, "entropy_aux": 1e-3}}), encoding="utf-8")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def fake_trainer(tmp_path):
    return SimpleNamespace(default_root_dir=str(tmp_path), global_step=0, max_steps=200,
                           is_global_zero=True, sanity_checking=False,
                           optimizers=[SimpleNamespace(param_groups=[
                               {"name": "main", "lr": 1e-5}, {"name": "entropy_aux", "lr": 1e-3}])],
                           callback_metrics={"T/l_guide_roi_step": 2., "T/l_guide_bg_step": 1.,
                                             "T/roi_fraction_step": .25, "T/optim_loss_step": 3.})


def payload(sha):
    return {"wild_diff_checkpoint_contract_version": 2, "global_step": 100,
            CHECKPOINT_IDENTITY_KEY: sha, "optimizer_states": [{"param_groups": [
                {"name": "main", "lr": 1e-5}, {"name": "entropy_aux", "lr": 1e-3}]}]}


def test_own_csv_records_roi_once_without_missing_value_fabrication(tmp_path):
    path, sha = contract(tmp_path)
    trainer = fake_trainer(tmp_path)
    callback = H2LossCSV(contract_path=path, contract_sha256=sha)
    callback.on_fit_start(trainer, None)
    callback.on_train_start(trainer, None)
    trainer.global_step = 10
    for _ in range(8):
        callback.on_train_batch_end(trainer, None, None, None, 0)
    callback.on_train_end(trainer, None)
    with (tmp_path / "loss_components.csv").open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        assert tuple(reader.fieldnames) == FIELDS
        rows = list(reader)
    assert len(rows) == 1
    assert rows[0]["T/l_guide_roi"] == "2.0" and rows[0]["T/l_guide_bg"] == "1.0"
    assert rows[0]["T/roi_fraction"] == "0.25" and rows[0]["T/l_semantic"] == ""
    checkpoint = {}
    callback.on_save_checkpoint(trainer, None, checkpoint)
    assert checkpoint == {CHECKPOINT_IDENTITY_KEY: sha}
    assert "wild_diff_h1_recovery_contract_sha256" not in checkpoint


def test_changed_contract_wrong_header_and_outside_paths_fail_closed(tmp_path):
    path, sha = contract(tmp_path)
    trainer = fake_trainer(tmp_path)
    callback = H2LossCSV(contract_path=path, contract_sha256=sha)
    callback.on_fit_start(trainer, None)
    path.write_text('{"changed":true}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="hash changed"):
        callback.on_save_checkpoint(trainer, None, {})
    path, sha = contract(tmp_path)
    (tmp_path / "loss_components.csv").write_text("old_h1_header\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="header"):
        H2LossCSV(contract_sha256=sha).on_fit_start(trainer, None)
    with pytest.raises(RuntimeError, match="contained"):
        H2LossCSV(output_path=tmp_path.parent / "wrong.csv")._output(trainer)


def test_h2_checkpoint_identity_rates_step_and_optimizer_count(tmp_path):
    _, sha = contract(tmp_path)
    checkpoint = payload(sha)
    before = deepcopy(checkpoint)
    assert validate_h2_checkpoint(checkpoint, sha) == 100
    assert checkpoint == before
    for change in ({CHECKPOINT_IDENTITY_KEY: "f" * 64}, {"global_step": 201},
                   {"global_step": True}, {"optimizer_states": []}):
        altered = {**checkpoint, **change}
        with pytest.raises(RuntimeError):
            validate_h2_checkpoint(altered, sha)
    checkpoint["optimizer_states"][0]["param_groups"][0]["lr"] = 1e-4
    with pytest.raises(RuntimeError, match="learning rate"):
        validate_h2_checkpoint(checkpoint, sha)
    trainer = fake_trainer(tmp_path)
    trainer.max_steps = 201
    with pytest.raises(RuntimeError, match="bounded"):
        H2LossCSV().on_fit_start(trainer, None)


def test_real_cpu_fit_resume_restores_two_group_states_and_no_duplicate_rows(tmp_path):
    from torch.utils.data import DataLoader, TensorDataset
    path, sha = contract(tmp_path)

    class Tiny(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.main_weight = torch.nn.Parameter(torch.tensor(1.))
            self.aux_weight = torch.nn.Parameter(torch.tensor(.1))

        def training_step(self, batch, batch_idx):
            guide = self.main_weight.square()
            aux = self.aux_weight.square()
            loss = guide + aux
            self.log_dict({"T/l_guide": guide, "T/l_guide_roi": guide,
                           "T/l_guide_bg": guide * .5, "T/roi_fraction": torch.tensor(.25),
                           "T/aux_loss": aux, "T/optim_loss": loss}, on_step=True, on_epoch=True)
            return loss

        def configure_optimizers(self):
            return torch.optim.AdamW([{"params": [self.main_weight], "name": "main", "lr": 1e-5},
                                      {"params": [self.aux_weight], "name": "entropy_aux", "lr": 1e-3}])

        def on_save_checkpoint(self, checkpoint):
            checkpoint["wild_diff_checkpoint_contract_version"] = 2

    def trainer(steps):
        return pl.Trainer(default_root_dir=str(tmp_path), accelerator="cpu", max_steps=steps,
                          max_epochs=-1, logger=False, enable_progress_bar=False,
                          enable_model_summary=False, enable_checkpointing=False,
                          num_sanity_val_steps=0, accumulate_grad_batches=2,
                          callbacks=[H2LossCSV(every_n_steps=1, contract_path=path, contract_sha256=sha)])

    loader = DataLoader(TensorDataset(torch.ones(12, 1)), batch_size=1)
    first = trainer(2)
    first.fit(Tiny(), loader)
    checkpoint = tmp_path / "tiny_h2.ckpt"
    first.save_checkpoint(checkpoint)
    original = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert validate_h2_checkpoint(original, sha) == 2
    assert len(original["optimizer_states"][0]["state"]) == 2
    assert all(float(value["step"]) == 2 for value in original["optimizer_states"][0]["state"].values())
    second = trainer(3)
    second.fit(Tiny(), loader, ckpt_path=str(checkpoint))
    assert second.global_step == 3
    resumed = tmp_path / "tiny_h2_resumed.ckpt"
    second.save_checkpoint(resumed)
    final = torch.load(resumed, map_location="cpu", weights_only=False)
    assert validate_h2_checkpoint(final, sha) == 3
    assert all(float(value["step"]) == 3 for value in final["optimizer_states"][0]["state"].values())
    with (tmp_path / "loss_components.csv").open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row["kind"] == "train"]
    assert [int(row["global_step"]) for row in rows] == [1, 2, 3]
    assert all(row["T/l_guide_roi"] and row["T/l_guide_bg"] and row["T/roi_fraction"] for row in rows)
    assert all(row["main_lr"] == "1e-05" and row["entropy_aux_lr"] == "0.001" for row in rows)
