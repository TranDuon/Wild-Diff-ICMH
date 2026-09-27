"""Static contracts for the Colab Phase-1 completion path."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_inference_migrates_checkpoint_before_validation_and_load():
    source = (ROOT / "inference_partition.py").read_text(encoding="utf-8")
    function = source[source.index("def _load_checkpoint"):source.index("def process")]
    migration = function.index("migrate_legacy_entropy_bottleneck_checkpoint")
    validation = function.index("validate_entropy_bottleneck_checkpoint")
    shape_check = function.index("state_dict_shape_mismatches")
    loading = function.index("load_state_dict")
    assert migration < validation < shape_check < loading


def test_inference_resolves_run_architecture_before_model_construction():
    source = (ROOT / "inference_partition.py").read_text(encoding="utf-8")
    main = source[source.index("def main()"):]
    resolution = main.index("resolve_run_model_config(args.config, args.ckpt_lc)")
    config_load = main.index("OmegaConf.load(resolved_config)")
    construction = main.index("instantiate_from_config(model_config)")
    assert resolution < config_load < construction


def test_generated_notebook_checks_resume_and_end_to_end_metrics():
    notebook = json.loads(
        (ROOT / "Wild_Diff_ICMH_Kgalagadi_Train.ipynb").read_text(encoding="utf-8")
    )
    code = "\n".join(
        cell["source"] for cell in notebook["cells"] if cell["cell_type"] == "code"
    )
    assert "optimizer_states" in code
    assert "RESUME_TARGET_STEP = SMOKE_GLOBAL_STEP + 1" in code
    assert "RESUME THÀNH CÔNG" in code
    assert "'h1_v2'" in code
    assert "'--limit', '2'" in code
    assert "'--results-registry', str(RESULTS_REGISTRY)" in code
    assert "END-TO-END THÀNH CÔNG" in code
    assert "str(RUN_DIR / 'config_model.yaml')" in code
    assert "run_and_log(decode_command" in code
    assert code.count("'--crop-size', '256'") == 2


def test_generated_notebook_has_permanent_hotfix_pull_cell():
    notebook = json.loads(
        (ROOT / "Wild_Diff_ICMH_Kgalagadi_Train.ipynb").read_text(encoding="utf-8")
    )
    markdown_text = "\n".join(
        cell["source"] for cell in notebook["cells"] if cell["cell_type"] == "markdown"
    )
    code_text = "\n".join(
        cell["source"] for cell in notebook["cells"] if cell["cell_type"] == "code"
    )

    assert "Bước 2A — Cập nhật bản sửa mới nhất" in markdown_text
    assert "['git', 'pull', '--ff-only', 'origin', BRANCH]" in code_text
    assert "['git', 'status', '--porcelain']" in code_text
    assert "assert after == remote" in code_text
    assert "CẬP NHẬT THÀNH CÔNG" in code_text


def test_model_marks_every_project_checkpoint_with_current_contract():
    source = (ROOT / "model" / "diffeic.py").read_text(encoding="utf-8")
    function = source[source.index("    def on_save_checkpoint"):source.index("    def on_load_checkpoint")]
    marker = function.index('checkpoint["wild_diff_checkpoint_contract_version"]')
    compact_early_return = function.index("if not self.compact_checkpoint")
    assert marker < compact_early_return
