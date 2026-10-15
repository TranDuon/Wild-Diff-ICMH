"""Exercise production asset preflight without loading the codec stack."""
import ast
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


def asset_gate():
    tree = ast.parse((Path(__file__).parents[1] / "train.py").read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_run_asset_preflight")
    namespace = {"Path": Path, "os": os}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "production-asset-gate", "exec"), namespace)
    return namespace[node.name]


def config(root, splits=None):
    class Config(dict):
        pass
    value = Config(preflight={} if splits is None else {"asset_splits": splits})
    value.data = SimpleNamespace(params={"dataset_overrides": {"data_root": str(root)}})
    return value


def test_h2_asset_gate_does_not_require_held_out_test_images(tmp_path, monkeypatch):
    monkeypatch.delenv("WILD_DATA_SOURCE_ROOT", raising=False)
    rows = [{"image_id": split, "relative_path": split + ".png", "split": split} for split in ("train", "val", "test")]
    for split in ("train", "val"):
        (tmp_path / (split + ".png")).touch()
    asset_gate()(config(tmp_path, ["train", "val"]), rows)
    with pytest.raises(FileNotFoundError, match="test.png"):
        asset_gate()(config(tmp_path), rows)


@pytest.mark.parametrize("splits", [[], ["unknown"], ["train", "train"], "train"])
def test_asset_scope_fails_closed(tmp_path, splits, monkeypatch):
    monkeypatch.delenv("WILD_DATA_SOURCE_ROOT", raising=False)
    with pytest.raises(ValueError, match="asset_splits"):
        asset_gate()(config(tmp_path, splits), [{"image_id": "a", "relative_path": "a.png", "split": "train"}])


def _rows():
    return [{"image_id": split, "relative_path": split + ".png", "split": split} for split in ("train", "val", "test")]


def test_source_root_satisfies_preflight_when_local_root_is_empty(tmp_path, monkeypatch):
    local, source = tmp_path / "local", tmp_path / "drive"
    local.mkdir()
    source.mkdir()
    for split in ("train", "val"):
        (source / (split + ".png")).touch()
    monkeypatch.setenv("WILD_DATA_SOURCE_ROOT", str(source))
    asset_gate()(config(local, ["train", "val"]), _rows())


def test_source_root_missing_image_fails_naming_source(tmp_path, monkeypatch):
    local, source = tmp_path / "local", tmp_path / "drive"
    local.mkdir()
    source.mkdir()
    (source / "val.png").touch()
    (local / "train.png").touch()  # a local copy does not excuse a missing original on Drive
    monkeypatch.setenv("WILD_DATA_SOURCE_ROOT", str(source))
    with pytest.raises(FileNotFoundError, match="1 selected images are missing under") as error:
        asset_gate()(config(local, ["train", "val"]), _rows())
    assert str(source) in str(error.value) and "train.png" in str(error.value)


def test_empty_source_root_counts_as_unset_and_checks_local(tmp_path, monkeypatch):
    source = tmp_path / "drive"
    source.mkdir()
    (source / "train.png").touch()
    (source / "val.png").touch()
    monkeypatch.setenv("WILD_DATA_SOURCE_ROOT", "")
    with pytest.raises(FileNotFoundError, match="2 selected images are missing under"):
        asset_gate()(config(tmp_path / "local", ["train", "val"]), _rows())
