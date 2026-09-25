"""Regression checks for the dependency contract used by the Colab notebook."""

from __future__ import annotations

import ast
import json
import importlib.util
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from utils.checkpoint_contract import state_dict_shape_mismatches


ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = ROOT / "requirements-colab.txt"
GENERATOR = ROOT / "tools" / "build_colab_training_notebook.py"
NOTEBOOK = ROOT / "Wild_Diff_ICMH_Kgalagadi_Train.ipynb"
RAM_SOURCE = ROOT / "src" / "recognize-anything"
RAM_BERT = RAM_SOURCE / "ram" / "models" / "bert.py"
RAM_PLUS = RAM_SOURCE / "ram" / "models" / "ram_plus.py"
RAM_TAGGER = ROOT / "tools" / "precompute_ram_tags.py"
TRAIN_ENTRYPOINT = ROOT / "train.py"
TRAIN_CONFIG = ROOT / "configs" / "train_kgalagadi_colab.yaml"

COMPRESSAI_NON_BASE_REQUIREMENTS = {
    "einops",
    "matplotlib",
    "pandas",
    "pybind11",
    "pytorch-msssim",
    "setuptools",
    "tomli",
    "torch-geometric",
    "tqdm",
    "typing-extensions",
    "wheel",
}


def requirement_names() -> set[str]:
    names = set()
    for raw_line in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        match = re.match(r"[A-Za-z0-9_.-]+", line)
        if match:
            names.add(match.group(0).lower().replace("_", "-"))
    return names


class ColabDependencyContractTests(unittest.TestCase):
    def test_kgalagadi_author_checkpoint_uses_matching_full_width_control_module(self):
        config = TRAIN_CONFIG.read_text(encoding="utf-8")
        self.assertRegex(
            config,
            r"(?s)model:.*?params:.*?control_stage_config:\s+params:\s+control_model_ratio: 1\.0",
        )

        source = TRAIN_ENTRYPOINT.read_text(encoding="utf-8")
        self.assertIn("_validate_author_checkpoint_contract(init_path, model_config)", source)
        self.assertIn("do not suppress control-model tensor size mismatches", source)

    def test_checkpoint_shape_contract_reports_shared_tensor_mismatch(self):
        class ShapedValue:
            def __init__(self, shape):
                self.shape = shape

        class Model:
            def state_dict(self):
                return {"weight": ShapedValue((2, 4))}

        checkpoint = {"state_dict": {"weight": ShapedValue((2, 3))}}

        self.assertEqual(
            state_dict_shape_mismatches(Model(), checkpoint),
            [("weight", (2, 3), (2, 4))],
        )

    def test_ram_plus_inference_does_not_download_a_bert_tokenizer(self):
        source = RAM_PLUS.read_text(encoding="utf-8")
        self.assertIn("if stage == 'train_from_scratch'", source)
        self.assertIn("else None", source)
        self.assertIn("if self.tokenizer is not None:", source)

        tree = ast.parse(source)
        tokenizer_calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "init_tokenizer"
        ]
        self.assertEqual(len(tokenizer_calls), 1)

    def test_ram_tag_preflight_reports_missing_images_before_model_load(self):
        tree = ast.parse(RAM_TAGGER.read_text(encoding="utf-8"))
        validate_node = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_validate_inputs"
        )
        namespace = {"Path": Path}
        exec(compile(ast.Module(body=[validate_node], type_ignores=[]), str(RAM_TAGGER), "exec"), namespace)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "ram.pth"
            checkpoint.write_bytes(b"x" * (1024 * 1024))
            rows = [{"relative_path": "site/missing.jpg"}]
            with self.assertRaisesRegex(FileNotFoundError, r"(?s)Missing 1/1.*missing\.jpg"):
                namespace["_validate_inputs"](rows, root / "images", checkpoint)

    def test_ram_tagger_direct_script_bootstraps_repo_root(self):
        source = RAM_TAGGER.read_text(encoding="utf-8")
        root_setup = "REPO_ROOT = Path(__file__).resolve().parents[1]"
        path_setup = "sys.path.insert(0, str(REPO_ROOT))"
        project_import = "from model.lfgcm import TagGCM"

        self.assertIn(root_setup, source)
        self.assertIn(path_setup, source)
        self.assertIn(project_import, source)
        self.assertLess(source.index(root_setup), source.index(project_import))
        self.assertLess(source.index(path_setup), source.index(project_import))

        with tempfile.TemporaryDirectory() as temporary:
            result = subprocess.run(
                [sys.executable, str(RAM_TAGGER), "--help"],
                cwd=temporary,
                capture_output=True,
                text=True,
                timeout=30,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Cache RAM++ tags once", result.stdout)
        self.assertNotIn("No module named 'model'", result.stderr)

    def test_generated_step_six_is_t4_safe_and_preserves_child_traceback(self):
        generator = GENERATOR.read_text(encoding="utf-8")
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        code = "\n".join(
            cell["source"]
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )

        for source in (generator, code):
            self.assertIn("ram_batch_size = 1 if gpu_memory_gib < 20 else 2", source)
            self.assertNotIn("'--batch-size', '8'", source)
            self.assertIn("stderr=subprocess.STDOUT", source)
            self.assertIn("tag_log_path", source)

    def test_generated_step_seven_streams_and_persists_child_traceback(self):
        generator = GENERATOR.read_text(encoding="utf-8")
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        code = "\n".join(
            cell["source"]
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )

        for source in (generator, code):
            self.assertIn("train_log_path", source)
            self.assertIn("PYTHONUNBUFFERED='1'", source)
            self.assertIn("stderr=subprocess.STDOUT", source)
            self.assertIn("stdout=subprocess.PIPE", source)
            self.assertIn("Smoke test training dừng với mã", source)

    def test_training_entrypoint_reports_startup_phases_and_checks_checkpoints(self):
        source = TRAIN_ENTRYPOINT.read_text(encoding="utf-8")

        for phase in range(1, 7):
            self.assertIn(f"[Train {phase}/6]", source)
        self.assertIn("init checkpoint not found", source)
        self.assertIn("Stable Diffusion checkpoint not found", source)

    def test_ram_bert_uses_current_transformers_helper_modules(self):
        tree = ast.parse(RAM_BERT.read_text(encoding="utf-8"))
        imports = {
            node.module: {alias.name for alias in node.names}
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and node.module in {
                "transformers.modeling_utils",
                "transformers.pytorch_utils",
            }
        }

        self.assertEqual(imports["transformers.modeling_utils"], {"PreTrainedModel"})
        self.assertEqual(
            imports["transformers.pytorch_utils"],
            {
                "apply_chunking_to_forward",
                "find_pruneable_heads_and_indices",
                "prune_linear_layer",
            },
        )

    def test_colab_keeps_transformers_on_supported_major(self):
        requirements = REQUIREMENTS.read_text(encoding="utf-8")
        self.assertRegex(requirements, r"(?m)^transformers>=4\.41,<5$")
        self.assertNotRegex(requirements, r"(?m)^transformers[^\n]*<6$")

    def test_compressai_no_deps_has_complete_explicit_runtime_closure(self):
        missing = COMPRESSAI_NON_BASE_REQUIREMENTS - requirement_names()
        self.assertFalse(missing, f"Missing explicit CompressAI dependencies: {sorted(missing)}")

    def test_generated_step_three_checks_the_original_missing_import(self):
        generator = GENERATOR.read_text(encoding="utf-8")
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        code = "\n".join(
            cell["source"]
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )

        for source in (generator, code):
            self.assertIn("'compressai==1.2.8'", source)
            self.assertIn("'--no-deps', '--no-build-isolation'", source)
            self.assertIn("core_versions_after == core_versions_before", source)
            self.assertIn("import pytorch_msssim", source)
            self.assertIn("import compressai.ans", source)
            self.assertIn("import compressai.entropy_models", source)
            self.assertIn("import compressai.losses", source)
            self.assertIn("import compressai.zoo", source)
            self.assertIn("'--force-reinstall', 'src/recognize-anything'", source)
            self.assertNotIn("'-e', 'src/recognize-anything'", source)
            self.assertIn("importlib.invalidate_caches()", source)
            self.assertIn("import ram", source)
            self.assertIn("from model.lfgcm import TagGCM", source)

    def test_regular_ram_install_is_visible_without_interpreter_restart(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "site-packages"
            target.mkdir()
            sys.path.insert(0, str(target))
            try:
                self.assertIsNone(importlib.util.find_spec("ram"))
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "--quiet",
                        "--no-cache-dir",
                        "--no-deps",
                        "--no-build-isolation",
                        "--target",
                        str(target),
                        str(RAM_SOURCE),
                    ],
                    check=True,
                )
                importlib.invalidate_caches()
                spec = importlib.util.find_spec("ram")
                self.assertIsNotNone(spec)
                self.assertTrue((target / "ram" / "__init__.py").is_file())
                installed_bert = (target / "ram" / "models" / "bert.py").read_text(
                    encoding="utf-8"
                )
                self.assertIn("from transformers.pytorch_utils import (", installed_bert)
                self.assertNotIn(
                    "from transformers.modeling_utils import (\n    PreTrainedModel,\n    apply_chunking_to_forward,",
                    installed_bert,
                )
            finally:
                sys.path.remove(str(target))
                importlib.invalidate_caches()

    def test_generated_step_three_activates_ram_in_the_current_kernel(self):
        generator = GENERATOR.read_text(encoding="utf-8")
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        code = "\n".join(
            cell["source"]
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )

        regular_install = "'--force-reinstall', 'src/recognize-anything'"
        cache_refresh = "importlib.invalidate_caches()"
        ram_import = "import ram"
        taggcm_import = "from model.lfgcm import TagGCM"

        for source in (generator, code):
            self.assertIn(regular_install, source)
            self.assertIn(cache_refresh, source)
            self.assertIn(ram_import, source)
            self.assertLess(source.index(regular_install), source.index(cache_refresh))
            self.assertLess(source.index(cache_refresh), source.index(ram_import))
            self.assertLess(source.index(ram_import), source.index(taggcm_import))


if __name__ == "__main__":
    unittest.main()
