"""CPU-only tests for opt-in WILD_DATA_SOURCE_ROOT copy-on-demand."""
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def world(tmp_path, monkeypatch, camera_trap_dataset_module):
    monkeypatch.delenv("WILD_DATA_SOURCE_ROOT", raising=False)
    source, local = tmp_path / "drive", tmp_path / "local"
    (source / "site").mkdir(parents=True)
    local.mkdir()
    relative = "site/a.jpg"
    Image.new("RGB", (40, 30), (120, 60, 30)).save(source / relative, "JPEG")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({
        "image_id": "KGA:a", "split": "train", "relative_path": relative,
        "site_id": "KGA:A01", "sequence_id": "KGA:S1"}) + "\n", encoding="utf-8")
    return SimpleNamespace(mod=camera_trap_dataset_module, source=source, local=local,
                           relative=relative, manifest=manifest, tmp=tmp_path)


def build(world, monkeypatch, source=True):
    if source:
        monkeypatch.setenv("WILD_DATA_SOURCE_ROOT", str(world.source))
    return world.mod.CameraTrapDataset(str(world.manifest), str(world.local), "train",
                                       out_size=16, crop_type="center", use_hflip=False)


def partials(root):
    return list(root.rglob("*.partial-*"))


def test_copies_on_first_read_then_reuses_local_without_source(world, monkeypatch):
    dataset = build(world, monkeypatch)
    sample = dataset[0]
    target = world.local / world.relative
    assert target.read_bytes() == (world.source / world.relative).read_bytes()
    assert sample["jpg"].shape == (16, 16, 3) and sample["image_id"] == "KGA:a"
    (world.source / world.relative).unlink()
    assert dataset[0]["image_id"] == "KGA:a"  # source never consulted once local exists
    assert not partials(world.local)


def test_missing_everywhere_raises_and_leaves_nothing(world, monkeypatch):
    (world.source / world.relative).unlink()
    dataset = build(world, monkeypatch)
    with pytest.raises(FileNotFoundError, match="a.jpg"):
        dataset[0]
    assert not (world.local / world.relative).exists() and not partials(world.local)


@pytest.mark.parametrize("bad", ["../escape.jpg", "/abs/escape.jpg", "site/../../escape.jpg", "C:/x.jpg", ""])
def test_unsafe_paths_rejected(world, bad):
    (world.tmp / "escape.jpg").write_bytes(b"x")
    with pytest.raises(ValueError):
        world.mod._copy_on_demand(world.source, world.local, bad)
    assert not list(world.local.rglob("*"))
    assert sorted(p.name for p in world.tmp.iterdir() if p.is_file()) == ["escape.jpg", "manifest.jsonl"]


def test_transient_copy_error_is_retried(world, monkeypatch):
    monkeypatch.setattr(world.mod.time, "sleep", lambda s: None)
    real, calls = world.mod.shutil.copyfile, []

    def flaky(src, dst):
        calls.append(1)
        if len(calls) == 1:
            raise OSError("drive hiccup")
        return real(src, dst)
    monkeypatch.setattr(world.mod.shutil, "copyfile", flaky)
    dataset = build(world, monkeypatch)
    dataset[0]
    assert len(calls) == 2
    assert (world.local / world.relative).is_file() and not partials(world.local)


def test_persistent_copy_error_raises_without_leftovers(world, monkeypatch):
    monkeypatch.setattr(world.mod.time, "sleep", lambda s: None)

    def broken(src, dst):
        raise OSError("always")
    monkeypatch.setattr(world.mod.shutil, "copyfile", broken)
    dataset = build(world, monkeypatch)
    with pytest.raises(OSError, match="failed to copy image"):
        dataset[0]
    assert not (world.local / world.relative).exists() and not partials(world.local)


def test_incomplete_copy_never_appears_under_final_name(world, monkeypatch):
    monkeypatch.setattr(world.mod.time, "sleep", lambda s: None)

    def truncated(src, dst):
        with open(dst, "wb") as stream:
            stream.write(b"x")
    monkeypatch.setattr(world.mod.shutil, "copyfile", truncated)
    with pytest.raises(OSError, match="failed to copy image"):
        world.mod._copy_on_demand(world.source, world.local, world.relative)
    assert not (world.local / world.relative).exists() and not partials(world.local)


def test_copy_consumes_no_global_rng(world):
    random.seed(7)
    np.random.seed(7)
    py_state, np_state = random.getstate(), np.random.get_state()
    world.mod._copy_on_demand(world.source, world.local, world.relative)
    assert (world.local / world.relative).is_file()
    assert random.getstate() == py_state
    after = np.random.get_state()
    assert after[0] == np_state[0] and (after[1] == np_state[1]).all() and after[2:] == np_state[2:]


def test_source_root_must_be_a_directory(world, monkeypatch):
    monkeypatch.setenv("WILD_DATA_SOURCE_ROOT", str(world.tmp / "not_mounted"))
    with pytest.raises(FileNotFoundError, match="WILD_DATA_SOURCE_ROOT"):
        world.mod.CameraTrapDataset(str(world.manifest), str(world.local), "train", out_size=16)


def test_env_unset_keeps_legacy_behaviour(world, monkeypatch):
    monkeypatch.setattr(world.mod.time, "sleep", lambda s: None)
    dataset = build(world, monkeypatch, source=False)
    assert dataset.source_root is None
    with pytest.raises(OSError, match="failed to load image"):
        dataset[0]
    assert not (world.local / world.relative).exists()  # source dir never copied from
    target = world.local / world.relative
    target.parent.mkdir(parents=True)
    Image.new("RGB", (40, 30)).save(target, "JPEG")
    assert dataset[0]["image_id"] == "KGA:a"
