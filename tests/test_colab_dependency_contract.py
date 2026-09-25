"""Regression checks for the dependency contract used by the Colab notebook."""

from __future__ import annotations

import json
import importlib.util
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = ROOT / "requirements-colab.txt"
GENERATOR = ROOT / "tools" / "build_colab_training_notebook.py"
NOTEBOOK = ROOT / "Wild_Diff_ICMH_Kgalagadi_Train.ipynb"
RAM_SOURCE = ROOT / "src" / "recognize-anything"

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
