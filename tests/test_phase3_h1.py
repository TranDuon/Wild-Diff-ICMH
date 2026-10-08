"""CPU contracts: notebook compilation, frozen subsets, safe resume, gating."""
import ast
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import phase3_h1 as h1


def load_config(path):
    value = yaml.safe_load(path.read_text())
    base = value.pop("base", None)
    def merge(a, b):
        result = dict(a)
        for key, item in b.items():
            result[key] = merge(result[key], item) if isinstance(item, dict) and isinstance(result.get(key), dict) else item
        return result
    return merge(load_config(path.parent / base), value) if base else value


def test_pilot_keeps_h1_and_replaces_expensive_callbacks():
    config = load_config(ROOT / h1.CONFIG)
    data = config["data"]["params"]["dataset_overrides"]
    assert data["site_id"] is None and data["processing_long_side"] == 512
    assert data["bbox_crop_probability"] == 0
    model = config["model"]["params"]
    assert model["validation_mode"] == "loss_only"
    assert model["external_text_conditioning"] and not model["roi_loss_enabled"]
    assert model["control_stage_config"]["params"]["control_model_ratio"] == 1
    trainer = config["lightning"]["trainer"]
    assert trainer["max_steps"] == 500 and trainer["max_epochs"] == -1
    assert trainer["val_check_interval"] == 800 and trainer["check_val_every_n_epoch"] is None
    assert trainer["limit_val_batches"] == 2
    names = [c["target"] for c in config["lightning"]["callbacks"]]
    assert "model.callbacks.ImageLogger" not in names
    assert len(names) == 4 and "model.callbacks.TrainingProgressMonitor" in names
    assert not any(c["params"].get("monitor") == "avg_lpips" for c in config["lightning"]["callbacks"])


def test_notebook_compiles_is_generated_and_heavy_options_are_off():
    sys.path.insert(0, str(ROOT / "tools"))
    import build_colab_phase3_notebook as builder
    notebook = json.loads((ROOT / "Wild_Diff_ICMH_Phase3_H1.ipynb").read_text(encoding="utf-8"))
    assert notebook == builder.build()
    cells = notebook["cells"]
    source = "\n".join(c["source"] for c in cells if c["cell_type"] == "code")
    for index, cell in enumerate(cells):
        if cell["cell_type"] == "code":
            compile(cell["source"], f"cell-{index}", "exec")
    assert "BRANCH = 'main'" in source and "BRANCH = 'phase2'" not in source
    assert "git', 'pull', '--ff-only'" in source
    assert "local_changes" in source and "after == remote" in source
    for name in ("RUN_EXTENSION", "RUN_FULL_DEV", "DISCONNECT_NOW"):
        assert f"{name} = False" in source
    assert "precompute_ram_tags.py" not in source and "hf_hub_download" not in source
    assert "phase3_h1.py" in source
    assert "size_bin=all" not in source


def fixtures():
    rows = [{"image_id": str(i), "split": "val", "is_empty": i % 2 == 0} for i in range(60)]
    illumination = {str(i): {"illumination": "night" if i % 3 == 0 else "day"} for i in range(60)}
    detections = {str(i): {"detections": ([] if i % 2 == 0 else
                  [{"category": "1", "conf": .9, "bbox": [0, 0, .05 if i % 5 == 0 else .5, .5]}])} for i in range(60)}
    return rows, [str(i) for i in range(60)], illumination, detections


def test_selection_fixed_balanced_validation_only():
    rows, dev, illum, det = fixtures()
    selection = h1.select_quick(rows, dev, illum, det)
    assert len(selection) == len(set(selection)) == 30
    assert selection == h1.select_quick(list(reversed(rows)), list(reversed(dev)), illum, det)
    chosen = [row for row in rows if row["image_id"] in selection]
    assert {row["is_empty"] for row in chosen} == {True, False}
    assert {illum[row["image_id"]]["illumination"] for row in chosen} == {"day", "night"}
    rows[0]["split"] = "test"
    with pytest.raises(ValueError, match="validation only"):
        h1.select_quick(rows, dev, illum, det)


def test_selection_rejects_missing_sidecars_and_duplicate_ids():
    rows, dev, illum, det = fixtures()
    with pytest.raises(ValueError, match="Duplicate"):
        h1.select_quick(rows, dev + [dev[0]], illum, det)
    del illum[dev[0]]
    with pytest.raises(ValueError, match="Missing"):
        h1.select_quick(rows, dev, illum, det)


def test_frozen_artifacts_never_overwrite(tmp_path):
    target = tmp_path / "selection.txt"
    h1.freeze(target, "a\n")
    h1.freeze(target, "a\n")
    with pytest.raises(RuntimeError, match="Frozen artifact changed"):
        h1.freeze(target, "b\n")
    assert target.read_text() == "a\n"


@pytest.mark.parametrize("args", [["evaluate", "--scope", "full"], ["train", "--target-steps", "1000"]])
def test_expensive_actions_require_explicit_approval(args):
    with pytest.raises(SystemExit) as exc:
        h1.main(args)
    assert exc.value.code == 2


def test_loss_only_validation_does_not_decode_or_create_metrics():
    tree = ast.parse((ROOT / "model/diffeic.py").read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DiffEIC")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "validation_step")
    branch = ast.unparse(method.body[0])
    assert "loss_only" in branch and "shared_step" in branch
    assert "log_images" not in branch and "val/objective" in branch
    assert isinstance(method.body[0].body[-1], ast.Return)


def test_child_failure_logs_and_ledgers_then_raises(tmp_path):
    p = h1.paths(tmp_path)
    with pytest.raises(RuntimeError, match="dừng với mã 3"):
        h1.run_logged([sys.executable, "-c", "print('specific failure'); raise SystemExit(3)"], p, "unit", 1.54)
    assert "specific failure" in (tmp_path / "logs/p3_unit.log").read_text()
    record = h1.read_jsonl(p["phase"] / "resource_usage.jsonl")[0]
    assert record["exit_code"] == 3 and record["cu_estimate"] >= 0


def test_already_trained_no_extra_optimizer_steps(tmp_path, monkeypatch):
    p = h1.paths(tmp_path)
    p["phase"].mkdir(parents=True)
    h1.write_json(p["phase"] / "protocol.json", {})
    last = p["run"] / "checkpoints/last.ckpt"
    last.parent.mkdir(parents=True)
    last.touch()
    monkeypatch.setattr(h1, "prepare", lambda p: None)
    monkeypatch.setattr(h1, "checkpoint_step", lambda path: 500)
    monkeypatch.setattr(h1, "run_logged", lambda *args: pytest.fail("Must not train again"))
    h1.train(p, 500, 1.5, 1.54)


def test_invalid_checkpoint_does_not_silently_restart(tmp_path, monkeypatch):
    p = h1.paths(tmp_path)
    p["phase"].mkdir(parents=True)
    h1.write_json(p["phase"] / "protocol.json", {})
    last = p["run"] / "checkpoints/last.ckpt"
    last.parent.mkdir(parents=True)
    last.touch()
    monkeypatch.setattr(h1, "prepare", lambda p: None)
    def invalid(path):
        raise RuntimeError("unsafe checkpoint")
    monkeypatch.setattr(h1, "checkpoint_step", invalid)
    monkeypatch.setattr(h1, "run_logged", lambda *args: pytest.fail("Must not restart"))
    with pytest.raises(RuntimeError, match="unsafe checkpoint"):
        h1.train(p, 500, 1.5, 1.54)


def test_loss_only_validation_executes_without_sampling():
    from types import SimpleNamespace
    tree = ast.parse((ROOT / "model/diffeic.py").read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DiffEIC")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "validation_step")
    method.decorator_list = []
    module = ast.Module(body=[method], type_ignores=[])
    namespace = {}
    exec(compile(ast.fix_missing_locations(module), "validation-extracted", "exec"), namespace)
    logged = []
    model = SimpleNamespace(validation_mode="loss_only", first_stage_key="jpg",
                            shared_step=lambda batch: (1.25, {}), log=lambda *a, **k: logged.append((a, k)),
                            log_images=lambda *a, **k: pytest.fail("No diffusion sampling during loss-only validation"))
    result = namespace["validation_step"](model, {"jpg": SimpleNamespace(shape=(1, 256, 256, 3))}, 0)
    assert result == 1.25 and logged[0][0] == ("val/objective", 1.25)
    assert logged[0][1]["batch_size"] == 1


def test_evaluation_reuses_b0_across_milestones_and_retains_same_protocol(tmp_path, monkeypatch):
    p = h1.paths(tmp_path)
    p["quick"].parent.mkdir(parents=True)
    p["quick"].write_text("image1\n")
    h1.write_json(p["b0"] / "run_info.json", {"processing_long_side": 512, "steps": 50,
                                              "sampler": "ddim", "seed": 231, "c_cfg_scale": 3.0})
    checkpoint = tmp_path / "fake.ckpt"
    checkpoint.write_bytes(b"trusted test fixture")
    step = [500]
    monkeypatch.setattr(h1, "prepare", lambda p: None)
    monkeypatch.setattr(h1, "snapshot", lambda p: (checkpoint, step[0]))
    monkeypatch.setattr(h1, "assert_archive_complete", lambda root, selected: None)
    calls = []
    def fake_job(command, paths, name, rate):
        command = list(map(str, command))
        calls.append((name, command))
        if "--output" not in command:
            return
        output = Path(command[command.index("--output") + 1])
        if "tools/evaluate_kgalagadi.py" in command:
            h1.write_json(output.with_suffix(".summary.json"), {"groups": {"illumination=all|content=all": {"n": 1}}})
        elif "tools/eval_machine.py" in command:
            h1.write_json(output, {"groups": {"illumination=all|content=all": {"n": 1}}})
    monkeypatch.setattr(h1, "run_logged", fake_job)
    h1.evaluate(p, "quick", 1.54, "detect-python")
    step[0] = 1000
    h1.evaluate(p, "quick", 1.54, "detect-python")
    assert sum(name.endswith("_B0") and name.startswith("quality_") for name, _ in calls) == 1
    assert sum(name.endswith("_H1") and name.startswith("quality_") for name, _ in calls) == 2
    decodes = [command for name, command in calls if name.startswith("decode_")]
    assert len(decodes) == 2
    for command in decodes:
        assert command[command.index("--steps") + 1] == "50"
        assert command[command.index("--split") + 1] == "val"
        assert command[command.index("--seed") + 1] == "231"
        assert command[command.index("--processing-long-side") + 1] == "512"
        assert "--skip-existing" in command
    assert (p["phase"] / "eval/step_001000_quick/comparison.json").is_file()


def test_snapshot_isolated_between_smoke_and_pilot_and_config_is_repairable(tmp_path, monkeypatch):
    p = h1.paths(tmp_path)
    monkeypatch.setattr(h1, "checkpoint_step", lambda path: 20)
    snapshots = []
    for name in ("h1_pooled_smoke_ls512", "h1_pooled_pilot_ls512"):
        p["run"] = tmp_path / "runs" / name / "lambda_2"
        source = p["run"] / "checkpoints/last.ckpt"
        source.parent.mkdir(parents=True)
        source.write_bytes(name.encode())
        (p["run"] / "config_model.yaml").write_text("model: test\n")
        first, _ = h1.snapshot(p)
        second, _ = h1.snapshot(p)
        assert first == second
        assert first.parent.parent.joinpath("config_model.yaml").is_file()
        snapshots.append(first)
    assert snapshots[0] != snapshots[1]


def test_sparse_progress_does_not_print_eight_times_per_accumulated_step(capsys):
    from types import SimpleNamespace
    tree = ast.parse((ROOT / "model/callbacks.py").read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TrainingProgressMonitor")
    namespace = {"Callback": object, "rank_zero_only": lambda function: function}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), "progress", "exec"), namespace)
    callback = namespace["TrainingProgressMonitor"]()
    trainer = SimpleNamespace(global_step=50, max_steps=500, callback_metrics={"T/optim_loss_step": 1.25})
    for batch in range(8):
        callback.on_train_batch_end(trainer, None, None, None, batch)
    assert capsys.readouterr().out.count("optimizer step 50/500") == 1
