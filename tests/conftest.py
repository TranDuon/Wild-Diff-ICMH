"""Shared fixtures for CPU-only tests."""
import importlib
import importlib.machinery
import importlib.util
import sys
import types

import pytest


@pytest.fixture
def camera_trap_dataset_module(monkeypatch):
    """Import ``dataset.camera_trap_dataset`` with or without torch installed.

    The Windows dev box has no torch, but the dataset module only needs
    ``torch.utils.data.Dataset`` as a base class, so a minimal stub is installed
    there. Colab (real torch) imports the genuine module. The module is removed
    from ``sys.modules`` afterwards so a stub-bound copy never leaks into other
    tests.
    """
    sys.modules.pop("dataset.camera_trap_dataset", None)
    if importlib.util.find_spec("torch") is None:
        for name in ("torch", "torch.utils", "torch.utils.data"):
            stub = types.ModuleType(name)
            stub.__spec__ = importlib.machinery.ModuleSpec(name, None)
            monkeypatch.setitem(sys.modules, name, stub)
        sys.modules["torch"].utils = sys.modules["torch.utils"]
        sys.modules["torch.utils"].data = sys.modules["torch.utils.data"]
        sys.modules["torch.utils.data"].Dataset = type("Dataset", (), {})
    module = importlib.import_module("dataset.camera_trap_dataset")
    try:
        yield module
    finally:
        sys.modules.pop("dataset.camera_trap_dataset", None)
