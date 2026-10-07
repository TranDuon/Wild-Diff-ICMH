"""Every project source must at least compile (catches syntax errors in modules
whose imports -- lightning, CUDA -- are unavailable in the local test env)."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("model", "utils", "dataset", "tools", "ldm")


def test_project_python_sources_compile():
    files = [ROOT / "train.py", ROOT / "inference_partition.py"]
    for package in PACKAGES:
        files.extend((ROOT / package).rglob("*.py"))
    for path in files:
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
