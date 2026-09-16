"""Fast local checks; no GPU, network, or multi-GB checkpoint needed."""

import ast
import json
import shutil
import sys
import tempfile
import types
from pathlib import Path


root = Path(__file__).resolve().parents[1]
notebook = json.loads((root / "Diff_ICMH_Colab_T4_Fixed.ipynb").read_text(encoding="utf-8"))
cells = notebook["cells"]
assert len(cells) == 21
assert all(not cell.get("outputs") for cell in cells)
for index, cell in enumerate(cells):
    if cell["cell_type"] == "code":
        ast.parse("".join(cell["source"]), filename=f"cell-{index}")

with tempfile.TemporaryDirectory() as directory:
    staging = Path(directory) / "repo"
    sources = [
        "model/diffeic.py",
        "ldm/models/diffusion/ddpm.py",
        "model/callbacks.py",
        "ldm/modules/encoders/modules.py",
        "inference_partition.py",
    ]
    for relative in sources:
        destination = staging / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / relative, destination)

    patch_cell = "".join(cells[8]["source"])
    patch_cell = patch_cell.replace(
        'REPO = Path("/content/Diff-ICMH")', f"REPO = Path({str(staging)!r})"
    )
    exec(compile(patch_cell, "cell-8", "exec"), {})
    once = {relative: (staging / relative).read_bytes() for relative in sources}
    exec(compile(patch_cell, "cell-8", "exec"), {})
    twice = {relative: (staging / relative).read_bytes() for relative in sources}
    assert once == twice, "Notebook patches are not idempotent"
    for relative in sources:
        ast.parse((staging / relative).read_text(encoding="utf-8"), filename=relative)
    inference = (staging / "inference_partition.py").read_text(encoding="utf-8")
    assert 'tagger.to("cpu")' in inference
    assert "mmap=True" in inference
    assert "preprocess_tag_model(control.cpu(), return_ids=True)" in inference

    # Exercise the missing-checkpoint recovery without downloading multi-GB files.
    checkpoint_root = Path(directory) / "checkpoints"
    sd = checkpoint_root / "sd2p1" / "v2-1_512-ema-pruned.ckpt"
    lc = checkpoint_root / "nested" / "model.ckpt"
    ram = checkpoint_root / "ram" / "ram_plus_swin_large_14m.pth"
    calls = []

    def fake_download(*, repo_id, filename, local_dir):
        calls.append((repo_id, filename))
        destination = Path(local_dir) / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"checkpoint")
        return str(destination)

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.hf_hub_download = fake_download
    previous_hub = sys.modules.get("huggingface_hub")
    sys.modules["huggingface_hub"] = fake_hub
    try:
        download_source = "".join(cells[12]["source"])
        download_source = download_source.replace("4.0)", "0.000000001)")
        download_source = download_source.replace("9.0)", "0.000000001)")
        download_source = download_source.replace("2.0)", "0.000000001)")
        context = {
            "SD_CKPT": sd, "LC_CKPT": lc, "RAM_CKPT": ram,
            "LC_RELATIVE": Path("nested/model.ckpt"), "CHECKPOINTS": checkpoint_root,
            "Path": Path,
        }
        exec(compile(download_source, "cell-12", "exec"), context)
        assert len(calls) == 3
        context["ensure_checkpoints"]()
        assert len(calls) == 3, "Existing checkpoints were downloaded again"
        lc.unlink()
        context["ensure_checkpoints"]()
        assert len(calls) == 4 and lc.is_file(), "Missing checkpoint was not restored"
    finally:
        if previous_hub is None:
            del sys.modules["huggingface_hub"]
        else:
            sys.modules["huggingface_hub"] = previous_hub

print("Notebook syntax, source patches, and checkpoint recovery passed")
