"""Regression checks for the dependency contract used by the Colab notebook."""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = ROOT / "requirements-colab.txt"
GENERATOR = ROOT / "tools" / "build_colab_training_notebook.py"
NOTEBOOK = ROOT / "Wild_Diff_ICMH_Kgalagadi_Train.ipynb"

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
            self.assertIn("from model.lfgcm import TagGCM", source)


if __name__ == "__main__":
    unittest.main()
