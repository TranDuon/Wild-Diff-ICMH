"""Static contracts for the generated Phase 2 Colab notebook."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "Wild_Diff_ICMH_Phase2_Eval.ipynb"


def _notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _code():
    return "\n".join(cell["source"] for cell in _notebook()["cells"] if cell["cell_type"] == "code")


def test_every_phase2_code_cell_compiles():
    for index, cell in enumerate(_notebook()["cells"]):
        if cell["cell_type"] == "code":
            compile(cell["source"], f"phase2-cell-{index}", "exec")


def test_phase2_reuses_setup_on_its_own_branch():
    code = _code()
    assert "BRANCH = 'phase2'" in code and "BRANCH = 'ver2'" not in code
    assert "CU_AVAILABLE_AT_START = None" in code and "CU_AVAILABLE_NOW = None" in code


def test_expensive_b0_decode_is_gated_and_resumable():
    code = _code()
    assert "RUN_B0 = False" in code
    assert "'--skip-existing'" in code
    assert "'--dev-list', DEV_LIST" in code


def test_dev_list_is_never_overwritten_once_frozen():
    code = _code()
    assert "assert dev_ids(check) == dev_ids(DEV_LIST)" in code


def test_all_methods_share_one_evaluator_and_detector():
    code = _code()
    for tool in ("tools/evaluate_kgalagadi.py", "tools/eval_machine.py", "tools/detect/run_megadetector.py",
                 "tools/baselines/run_classical.py", "tools/baselines/run_compressai_zoo.py",
                 "tools/data/label_illumination.py", "tools/data/build_dev_set.py"):
        assert tool in code
