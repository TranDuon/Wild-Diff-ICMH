"""Exercise production asset preflight without loading the codec stack."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


def asset_gate():
    tree = ast.parse((Path(__file__).parents[1] / "train.py").read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_run_asset_preflight")
    namespace = {"Path": Path}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "production-asset-gate", "exec"), namespace)
    return namespace[node.name]


def config(root, splits=None):
    class Config(dict):
        pass
    value = Config(preflight={} if splits is None else {"asset_splits": splits})
    value.data = SimpleNamespace(params={"dataset_overrides": {"data_root": str(root)}})
    return value


def test_h2_asset_gate_does_not_require_held_out_test_images(tmp_path):
    rows = [{"image_id": split, "relative_path": split + ".png", "split": split} for split in ("train", "val", "test")]
    for split in ("train", "val"):
        (tmp_path / (split + ".png")).touch()
    asset_gate()(config(tmp_path, ["train", "val"]), rows)
    with pytest.raises(FileNotFoundError, match="test.png"):
        asset_gate()(config(tmp_path), rows)


@pytest.mark.parametrize("splits", [[], ["unknown"], ["train", "train"], "train"])
def test_asset_scope_fails_closed(tmp_path, splits):
    with pytest.raises(ValueError, match="asset_splits"):
        asset_gate()(config(tmp_path, splits), [{"image_id": "a", "relative_path": "a.png", "split": "train"}])
