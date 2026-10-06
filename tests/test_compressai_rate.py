"""Estimated-rate mode of the CompressAI baseline and its evaluator path."""
from __future__ import annotations

import json
import sys
import types

import numpy as np
import pytest
from PIL import Image

from tools.baselines.run_compressai_zoo import AUTOREGRESSIVE, resolve_rate


def test_autoregressive_models_default_to_estimated_rate():
    assert set(AUTOREGRESSIVE) == {"mbt2018", "cheng2020-attn"}
    assert resolve_rate("mbt2018", "auto") == "estimated"
    assert resolve_rate("cheng2020-attn", "auto") == "estimated"
    assert resolve_rate("bmshj2018-hyperprior", "auto") == "coded"
    assert resolve_rate("mbt2018", "coded") == "coded"


def _stub_compressai(monkeypatch, torch):
    class Stub(torch.nn.Module):
        def forward(self, x):
            likelihood = torch.full((1, 1, x.shape[-2] // 16, x.shape[-1] // 16), 0.5)
            return {"x_hat": x * 0.5, "likelihoods": {"y": likelihood, "z": likelihood}}

        def compress(self, x):  # pragma: no cover - must not be reached
            raise AssertionError("estimated mode must not entropy-code")

    compressai = types.ModuleType("compressai")
    compressai.__version__ = "stub"
    compressai.set_entropy_coder = lambda name: None
    zoo = types.ModuleType("compressai.zoo")
    zoo.models = {"mbt2018": lambda **kwargs: Stub()}
    compressai.zoo = zoo
    monkeypatch.setitem(sys.modules, "compressai", compressai)
    monkeypatch.setitem(sys.modules, "compressai.zoo", zoo)


def test_estimated_rate_archive_is_scored_by_the_evaluator(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("pytorch_msssim")
    from tools import evaluate_kgalagadi
    from tools.baselines import run_compressai_zoo

    _stub_compressai(monkeypatch, torch)
    relative = "snapshot_kgalagadi/S/a.jpg"
    source = tmp_path / "images" / relative
    source.parent.mkdir(parents=True)
    pixels = (np.random.default_rng(0).random((200, 260, 3)) * 255).astype(np.uint8)
    Image.fromarray(pixels).save(source, quality=95)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({
        "image_id": "KGA:a", "relative_path": relative, "split": "val", "site_id": "KGA:S",
        "sequence_id": "KGA:S#1", "illumination": "day", "is_empty": False,
    }) + "\n", encoding="utf-8")
    archive = tmp_path / "archive"

    assert run_compressai_zoo.main([
        "--manifest", str(manifest), "--data-root", str(tmp_path / "images"), "--dev-list", "",
        "--model", "mbt2018", "--quality", "1", "--processing-long-side", "128",
        "--device", "cpu", "--output", str(archive),
    ]) == 0
    assert json.loads((archive / "run_info.json").read_text())["rate"] == "estimated"
    (logged,) = [json.loads(line) for line in (archive / "decode_log.jsonl").read_text().splitlines()]
    # 128x98 coded -> padded to 128x128 -> 8x8 latents, two tensors at 1 bit each
    assert logged["estimated_bits"] == pytest.approx(128.0)
    assert logged["bpp"] == pytest.approx(128.0 / (260 * 200))
    assert not (archive / "snapshot_kgalagadi" / "S" / "data").exists()

    # the frozen-manifest leakage gate is covered elsewhere; this fixture is a single row
    monkeypatch.setattr(evaluate_kgalagadi, "assert_no_leakage", lambda *args, **kwargs: None)
    output = tmp_path / "eval.jsonl"
    assert evaluate_kgalagadi.main([
        "--manifest", str(manifest), "--split", "val", "--illumination-sidecar", "",
        "--data-root", str(tmp_path / "images"), "--reconstruction-root", str(archive),
        "--method", "compressai-mbt2018_ls128", "--device", "cpu", "--archived-only", "--output", str(output),
        "--bootstrap-resamples", "10",
    ]) == 0
    (row,) = [json.loads(line) for line in output.read_text().splitlines()]
    assert row["rate_source"] == "estimated"
    assert row["bpp"] == pytest.approx(logged["bpp"])
    summary = json.loads(output.with_suffix(".summary.json").read_text())
    assert summary["protocol"]["rate"] == "estimated"
