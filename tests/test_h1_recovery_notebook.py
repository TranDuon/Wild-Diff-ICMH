"""Recovery notebook: safe defaults, CPU-first review and streamed CLI contract."""
import ast
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def builder():
    sys.path.insert(0, str(ROOT / "tools"))
    return importlib.import_module("build_colab_h1_recovery_notebook")


def code_cells():
    return [c["source"] for c in builder().build()["cells"] if c["cell_type"] == "code"]


def test_generated_notebook_matches_builder_and_compiles():
    expected = builder().build()
    actual = json.loads((ROOT / "Wild_Diff_ICMH_H1_Recovery.ipynb").read_text(encoding="utf-8"))
    assert actual == expected
    assert "accelerator" not in actual["metadata"]  # CPU review does not allocate a GPU.
    for index, cell in enumerate(actual["cells"]):
        if cell["cell_type"] == "code":
            compile(cell["source"], f"recovery-cell-{index}", "exec")
            assert cell["outputs"] == [] and cell["execution_count"] is None


def test_cpu_review_precedes_explicit_heavy_consent():
    sources = code_cells()
    review_index = next(i for i, s in enumerate(sources) if "recovery('review')" in s)
    cpu = "\n".join(sources[:review_index + 1])
    for forbidden in ("nvidia-smi", "import torch", "PytorchWildlife", "copy_checkpoint", "source_files", "requirements-colab"):
        assert forbidden not in cpu
    assert "pycocotools" in cpu and "phase3_h1_recovery.py" in cpu
    controls = "\n".join(sources[review_index + 1:])
    for name in ("RUN_TRIAL", "RUN_EVALUATION", "DISCONNECT_NOW"):
        assert f"{name} = False" in controls
    assert "ACKNOWLEDGE_REVIEW = ''" in controls
    assert "hashlib.sha256" in controls
    assert "--acknowledge-review" in controls and "--enable" in controls
    assert "SESSION_HOURS = 0.5" in controls
    assert "assert 0 < SESSION_HOURS <= 2" in controls
    for forbidden in ("--target-steps", "--allow-full", "--allow-extension", "--scope', 'full'", "RUN_FULL_DEV", "RUN_EXTENSION"):
        assert forbidden not in controls


def test_fixed_pull_is_safe_and_does_not_need_gpu():
    source = next(s for s in code_cells() if "def safe_pull(" in s)
    assert "local_changes" in source and "--porcelain" in source
    assert "--ff-only" in source and "after == remote" in source
    assert "shell=True" not in source and "nvidia-smi" not in source


def test_all_gpu_cells_are_gated_before_first_expensive_action():
    for source in code_cells():
        if any(token in source for token in ("requirements-colab", "source_files", "checkpoint_relatives", "PytorchWildlife")):
            tree = ast.parse(source)
            assert len(tree.body) == 1 and isinstance(tree.body[0], ast.If)
            condition = ast.unparse(tree.body[0].test)
            assert "RUN_TRIAL" in condition or "RUN_EVALUATION" in condition
    train = next(s for s in code_cells() if "recovery('train'" in s)
    evaluate = next(s for s in code_cells() if "recovery('evaluate'" in s)
    assert train.lstrip().startswith("if RUN_TRIAL:")
    assert evaluate.lstrip().startswith("if RUN_EVALUATION:")


def test_disabled_gpu_cells_execute_without_gpu_assets_or_setup(capsys):
    for source in code_cells():
        if source.lstrip().startswith("if RUN_TRIAL") or source.lstrip().startswith("if RUN_EVALUATION"):
            # Deliberately provide no Torch, paths, helper, pip or detector stubs.
            exec(compile(source, "disabled-gpu-cell", "exec"), {"RUN_TRIAL": False, "RUN_EVALUATION": False})
    assert "Bỏ qua" in capsys.readouterr().out
    sources = "\n".join(code_cells())
    assert "ram_plus_swin_large_14m.pth" not in sources


@pytest.mark.parametrize("changed", ["training_allowed", "protocol_verified", "recommendation", "hash"])
def test_ack_gate_rejects_unapproved_unverified_or_changed_review(tmp_path, changed):
    source = next(s for s in code_cells() if "def require_review_acknowledgement(" in s)
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef))
    path = tmp_path / "review.json"
    report = {"training_allowed": True, "protocol_verified": True, "recommendation": "conditional_low_lr_trial"}
    path.write_text(json.dumps(report), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    namespace = {"hashlib": hashlib, "json": json, "REVIEW_PATH": path, "REVIEW_SHA256": digest,
                 "ACKNOWLEDGE_REVIEW": digest, "SESSION_HOURS": 0.5}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "review-ack-gate", "exec"), namespace)
    namespace["require_review_acknowledgement"]()
    if changed == "hash":
        namespace["ACKNOWLEDGE_REVIEW"] = "0" * 64
    else:
        report[changed] = False if changed != "recommendation" else "do_not_train"
        path.write_text(json.dumps(report), encoding="utf-8")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        namespace.update(REVIEW_SHA256=digest, ACKNOWLEDGE_REVIEW=digest)
    with pytest.raises(AssertionError):
        namespace["require_review_acknowledgement"]()


def test_result_cell_displays_new_cached_review_point_schema(tmp_path, capsys):
    source = next(s for s in code_cells() if "comparisons = sorted" in s)
    artifact_root = tmp_path / "phase3" / "h1_low_lr_recovery"
    comparison = artifact_root / "eval" / "step_000200_quick" / "comparison.json"
    comparison.parent.mkdir(parents=True)
    comparison.write_text(json.dumps({
        "points": {
            "B0_2": {"quality": {"n": 30, "bpp": 0.005, "psnr": 22.3, "lpips": 0.54},
                     "machine": {"map": 0.39, "missed_animal_rate": 0.3, "hallucination_rate": 0.0}},
            "H1_low_lr_200": {"quality": {"n": 30, "bpp": 0.0045, "psnr": 22.1, "lpips": 0.53},
                              "machine": {"map": 0.37, "missed_animal_rate": 0.35, "hallucination_rate": 0.1}},
        },
        "recommendation": "stop_review", "protocol_verified": True,
        "loss_review": {"status": "unavailable", "missing": ["actual optimizer learning rates"]},
    }), encoding="utf-8")
    calls = []
    namespace = {"recovery": lambda action: calls.append(action), "RECOVERY_ARTIFACTS": artifact_root,
                 "REVIEW_PATH": artifact_root / "review.json", "RUN_DIR": tmp_path / "run", "json": json}
    exec(compile(source, "recovery-result-cell", "exec"), namespace)
    captured = capsys.readouterr().out
    assert calls == ["status"]
    for expected in ("B0_2", "H1_low_lr_200", "0.005", "0.0045", "22.1", "0.37", "0.35", "0.1",
                     "stop_review", "actual optimizer learning rates"):
        assert expected in captured


@pytest.mark.parametrize("return_code", [0, 7])
def test_recovery_wrapper_streams_real_cpu_child_and_raises(tmp_path, capsys, return_code):
    source = next(s for s in code_cells() if "def recovery(" in s)
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "recovery")
    child = tmp_path / "tools" / "phase3_h1_recovery.py"
    child.parent.mkdir()
    child.write_text("import sys\nprint('review stdout', flush=True)\n"
                     "print('review stderr', file=sys.stderr, flush=True)\n"
                     f"raise SystemExit({return_code})\n", encoding="utf-8")
    namespace = {"sys": sys, "subprocess": subprocess, "REPO": tmp_path, "DRIVE_ROOT": tmp_path,
                 "CU_RATE": 1.54}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "recovery-launcher", "exec"), namespace)
    if return_code:
        with pytest.raises(subprocess.CalledProcessError) as exc:
            namespace["recovery"]("review")
        assert exc.value.returncode == return_code
    else:
        namespace["recovery"]("review")
    assert capsys.readouterr().out.endswith("review stdout\nreview stderr\n")
    assert "stdout=subprocess.PIPE" in source and "stderr=subprocess.STDOUT" in source
    assert "flush=True" in source

