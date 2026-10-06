"""Static contracts for the generated Phase 2 Colab notebooks (Prepare + Eval)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PREPARE = ROOT / "Wild_Diff_ICMH_Phase2_Prepare.ipynb"
EVAL = ROOT / "Wild_Diff_ICMH_Phase2_Eval.ipynb"


def _cells(path):
    return json.loads(path.read_text(encoding="utf-8"))["cells"]


def _code(path):
    return "\n".join(cell["source"] for cell in _cells(path) if cell["cell_type"] == "code")


@pytest.mark.parametrize("path", [PREPARE, EVAL], ids=["prepare", "eval"])
def test_every_code_cell_compiles_and_has_no_placeholders(path):
    for index, cell in enumerate(_cells(path)):
        if cell["cell_type"] == "code":
            compile(cell["source"], f"{path.name}-cell-{index}", "exec")
            assert "__RUN_LOGGED__" not in cell["source"] and "__DETECT_ENV__" not in cell["source"]


@pytest.mark.parametrize("path", [PREPARE, EVAL], ids=["prepare", "eval"])
def test_notebooks_clone_the_phase2_branch_and_never_ask_for_compute_units(path):
    code = _code(path)
    assert "BRANCH = 'phase2'" in code and "'ver2'" not in code
    assert "CU_AVAILABLE" not in code and "sessions.jsonl" not in code


def test_eval_is_one_linear_workflow_on_dev_images_only():
    headings = [cell["source"].splitlines()[0] for cell in _cells(EVAL) if cell["cell_type"] == "markdown"][1:]
    numbers = [int(heading.split("Bước ")[1].split()[0]) for heading in headings]
    assert numbers == list(range(1, 11))
    code = _code(EVAL)
    assert "Bước 2A" not in "\n".join(headings)
    for heavy in ("tools/data/label_illumination.py", "inference_partition.py", "precompute_ram_tags.py",
                  "hf_hub_download", "RUN_B0"):
        assert heavy not in code
    assert "Ảnh dev cục bộ" in code
    assert "tools/data/domain_stats.py" not in code  # done once in Prepare (P2-4)


def test_baselines_run_on_exactly_the_images_b0_decoded():
    code = _code(EVAL)
    assert "BASELINE_DEV = max(" in code
    assert code.count("'--dev-list', BASELINE_DEV,") == 2
    assert "'--dev-list', DEV_LIST," not in code


def test_eval_uses_only_names_defined_by_earlier_cells():
    import ast
    import builtins

    defined = set(dir(builtins))
    for cell in _cells(EVAL):
        if cell["cell_type"] != "code":
            continue
        assigned, used = set(), set()
        for node in ast.walk(ast.parse(cell["source"])):
            if isinstance(node, ast.Name):
                (assigned if isinstance(node.ctx, ast.Store) else used).add(node.id)
            elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                assigned.add(node.name)
            elif isinstance(node, ast.arg):
                assigned.add(node.arg)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                assigned.update((alias.asname or alias.name).split(".")[0] for alias in node.names)
        assert not (used - defined - assigned), cell["source"].splitlines()[0]
        defined |= assigned


def test_eval_checks_b0_and_prepare_outputs_before_running():
    code = _code(EVAL)
    assert "Notebook Prepare chưa xong" in code
    assert "ARCHIVE.glob('B0_*/*/run_info.json')" in code
    assert "CHƯA ĐỦ" in code


def test_baseline_reconstructions_stay_off_drive_and_are_scored_once():
    code = _code(EVAL)
    assert "BASELINE_ROOT = Path('/content/p2_baselines')" in code
    assert "'--output', BASELINE_ROOT / curve / f'q{quality}'" in code
    assert "pack_bitstreams(root, curve, point)" in code
    assert "if scored(curve, f'q{quality}'):" in code


def test_each_baseline_point_is_scored_as_soon_as_it_is_coded():
    cells = [cell["source"] for cell in _cells(EVAL) if cell["cell_type"] == "code"]
    defining = [index for index, source in enumerate(cells) if "def score_archive(root):" in source]
    callers = [index for index, source in enumerate(cells) if "score_archive(BASELINE_ROOT / curve / f'q{quality}')" in source]
    assert len(defining) == 1 and len(callers) == 2 and defining[0] < min(callers)
    assert any("score_archive(run_info.parent)" in source for source in cells)
    code = _code(EVAL)
    # the bitstream tar is on Drive before a point is marked done
    assert code.index("pack_bitstreams(root, curve, point)  # before") < code.index("done_marker.write_text(")


def test_last_cell_records_the_session_and_disconnects_without_raising():
    last = [cell["source"] for cell in _cells(EVAL) if cell["cell_type"] == "code"][-1]
    assert "raise" not in last
    assert "AUTO_DISCONNECT = True" in last
    assert last.index("drive.flush_and_unmount()") < last.index("runtime.unassign()")


def test_partial_archives_are_scored_on_their_dev_list_and_rescored_when_they_grow():
    code = _code(EVAL)
    assert code.count("'--archived-only'") == 3
    assert "archive_dev = info.get('dev_list') or DEV_LIST" in code
    assert "json.loads(done_marker.read_text())['images'] >= archived" in code


def test_detector_env_does_not_depend_on_ensurepip():
    for path in (PREPARE, EVAL):
        code = _code(path)
        assert "'-m', 'venv'" not in code
        assert "'-m', 'virtualenv', '--system-site-packages'" in code
        assert "'PytorchWildlife', hub_pin" in code


def test_prepare_keeps_the_frozen_dev_set_gated_b0_and_cached_probe():
    code = _code(PREPARE)
    assert "assert dev_ids(check) == dev_ids(DEV_LIST)" in code
    assert "RUN_B0 = False" in code and "(512, LAMBDAS, 100)" in code
    assert "hashlib.sha256(f'20261005:{i}'.encode())" in code
    assert "PROBE_FILE = P2 / 'probe_timing.json'" in code
    assert "'--skip-existing'" in code


def test_every_phase2_tool_is_wired_into_a_notebook():
    code = _code(PREPARE) + _code(EVAL)
    for tool in ("tools/evaluate_kgalagadi.py", "tools/eval_machine.py", "tools/detect/run_megadetector.py",
                 "tools/baselines/run_classical.py", "tools/baselines/run_compressai_zoo.py",
                 "tools/data/label_illumination.py", "tools/data/build_dev_set.py", "tools/data/domain_stats.py"):
        assert tool in code
