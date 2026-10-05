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
    assert "BRANCH = 'phase2'" in code and "'ver2'" not in code
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


def test_partial_archives_are_scored_on_what_they_hold_and_rescored_when_they_grow():
    code = _code()
    assert code.count("'--archived-only'") == 3
    assert "json.loads(done_marker.read_text())['images'] >= archived" in code


def test_timing_probe_runs_once_per_project_not_per_session():
    code = _code()
    assert "PROBE_FILE = P2 / 'probe_timing.json'" in code
    assert "if str(side) not in probe_seconds:" in code


def test_detector_env_does_not_depend_on_ensurepip():
    code = _code()
    assert "'-m', 'venv'" not in code
    assert "'-m', 'virtualenv', '--system-site-packages'" in code
    assert "DETECT_READY.touch()" in code
    assert "'PytorchWildlife', hub_pin" in code


def test_b0_subsets_are_seeded_random_not_first_by_name():
    code = _code()
    assert "hashlib.sha256(f'20261005:{i}'.encode())" in code
    assert "limit=B0_LIMIT" not in code
    assert "(512, LAMBDAS, 100)" in code


def test_each_archive_is_scored_on_the_dev_list_it_was_decoded_for():
    code = _code()
    assert "archive_dev = info.get('dev_list') or DEV_LIST" in code
    assert "'--dev-list', DEV_LIST, '--image-root', root" not in code
    source = (ROOT / "inference_partition.py").read_text(encoding="utf-8")
    assert "'dev_list': os.path.abspath(args.dev_list) if args.dev_list else None" in source


def test_baseline_reconstructions_stay_off_drive():
    code = _code()
    assert "BASELINE_ROOT = Path('/content/p2_baselines')" in code
    assert "'--output', BASELINE_ROOT / curve / f'q{quality}'" in code
    assert "ARCHIVE / curve / f'q{quality}'" not in code
    assert "ARCHIVE.glob('B0_*/*/run_info.json')" in code
    assert "pack_bitstreams(root, curve, point)" in code


def test_shortcut_cell_reaches_baselines_without_full_setup():
    cells = [cell["source"] for cell in _notebook()["cells"]]
    shortcut = next(index for index, source in enumerate(cells) if "Ảnh dev cục bộ" in source)
    p2_8 = next(index for index, source in enumerate(cells) if "def run_classical" in source)
    assert shortcut < p2_8
    source = cells[shortcut]
    assert "def run_logged" in source and "DETECT_READY.touch()" in source
    assert "def scored" in source and "def dev_ids" in source
    assert "Đường tắt cần" in source
    assert not any("__RUN_LOGGED__" in c or "__DETECT_ENV__" in c for c in cells)
