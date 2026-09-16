"""Build a clean, rerunnable Colab T4 notebook from the user's debug notebook."""

import json
from pathlib import Path


SOURCE = Path(r"C:\Users\Duong_TD\Downloads\Diff_ICMH_Colab_T4_Clean (1).ipynb")
TARGET = Path(__file__).resolve().parents[1] / "Diff_ICMH_Colab_T4_Fixed.ipynb"


def lines(source):
    return source.strip("\n").splitlines(keepends=True)


notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
cells = notebook["cells"]

cells[0]["source"] = lines("""
# Diff-ICMH — Colab T4, kiểm tra suy luận BPP=2

Chạy **Runtime → Run all** từ đầu đến cuối trên Colab T4. Sau khi Colab tạo runtime
mới, chạy lại từ đầu: `/content` không được lưu qua runtime mới. Nếu chỉ ngắt kết nối
mà runtime vẫn còn, các cell tải file sẽ kiểm tra và bỏ qua file đã đủ dung lượng.

Notebook này thử **một ảnh 256 px** trước để giảm nguy cơ hết VRAM trên T4. Có thể
tăng `MAX_SIDE` ở cell upload sau khi lượt thử đầu tiên thành công. File kết quả và
log nằm trong `/content/Diff-ICMH/test_output_bpp2_t4`.
""")

cells[3]["source"] = lines("""
# Cell 2 — Clone the official source, or restore missing tracked files
import subprocess
from pathlib import Path

REPO = Path("/content/Diff-ICMH")
REPO_URL = "https://github.com/RuoyuFeng/Diff-ICMH.git"
if not REPO.exists():
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(REPO)], check=True)
elif not (REPO / ".git").is_dir():
    raise RuntimeError(f"{REPO} exists but is not a Git clone. Move that folder and rerun Cell 2.")

required_source = [
    "configs/model/diffeic.yaml", "inference_partition.py", "model/diffeic.py",
    "ldm/modules/encoders/modules.py", "src/recognize-anything/setup.py",
]
missing = [relative for relative in required_source if not (REPO / relative).is_file()]
if missing:
    print("Restoring missing tracked files:", missing)
    subprocess.run(["git", "restore", "--source=HEAD", "--", *missing], cwd=REPO, check=True)
if any(not (REPO / relative).is_file() for relative in required_source):
    raise RuntimeError("Repository clone is incomplete. Rerun Cell 2 on a fresh runtime.")

commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
print("Repo:", REPO)
print("Commit:", commit)
""")

cells[7]["source"] = lines("""
## Phase 3 — Vá tương thích cho bản mã nguồn đã clone

Các vá dưới đây có thể chạy lại. RAM+ được giữ trên CPU để dành VRAM của T4 cho
codec, SD 2.1 và control module. Cấu hình control ratio = 1.0 theo lệnh inference
trong README của tác giả.
""")

cells[8]["source"] = lines('''
# Cell 5 — Compatibility and T4 memory patches
from pathlib import Path

REPO = Path("/content/Diff-ICMH")

def replace_once(relative_path, old, new, label):
    path = REPO / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"Repo source missing: {path}. Rerun from Cell 2.")
    source = path.read_text(encoding="utf-8")
    if new in source:
        print(f"OK (already patched): {label}")
        return
    if old not in source:
        raise RuntimeError(f"Source changed; cannot apply {label}: {path}")
    path.write_text(source.replace(old, new, 1), encoding="utf-8")
    print(f"OK (patched): {label}")

replace_once(
    "model/diffeic.py",
    "from pytorch_lightning.utilities.types import EPOCH_OUTPUT",
    """try:
    from pytorch_lightning.utilities.types import EPOCH_OUTPUT
except ImportError:
    from typing import Any
    EPOCH_OUTPUT = Any""",
    "Lightning EPOCH_OUTPUT",
)

for relative_path in ["ldm/models/diffusion/ddpm.py", "model/callbacks.py"]:
    replace_once(
        relative_path,
        "from pytorch_lightning.utilities.distributed import rank_zero_only",
        "from pytorch_lightning.utilities.rank_zero import rank_zero_only",
        f"Lightning rank_zero_only in {relative_path}",
    )

replace_once(
    "ldm/modules/encoders/modules.py",
    "model, _, _ = open_clip.create_model_and_transforms(arch, device=torch.device('cpu'), pretrained=version)",
    """# SD 2.1 checkpoint supplies these text encoder weights.
        model, _, _ = open_clip.create_model_and_transforms(
            arch, device=torch.device('cpu'), pretrained=None
        )""",
    "skip redundant OpenCLIP download",
)

old_loader = """    ckpt_sd = torch.load(args.ckpt_sd, map_location="cpu")['state_dict']
    ckpt_lc = torch.load(args.ckpt_lc, map_location="cpu")['state_dict']
    ckpt_sd.update(ckpt_lc)
    msg = load_state_dict(model, ckpt_sd, strict=False)
    print(f"Messgae of load state dict: {msg}")"""

new_loader = """    # Load checkpoints sequentially so both full state dicts are never resident together.
    import gc

    def load_checkpoint(path, label):
        print(f"Loading {label}: {path}")
        try:
            checkpoint = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        except TypeError:  # older PyTorch without mmap/weights_only
            checkpoint = torch.load(path, map_location="cpu")
        state_dict = checkpoint["state_dict"]
        message = load_state_dict(model, state_dict, strict=False)
        print(f"{label}: missing={len(message.missing_keys)}, unexpected={len(message.unexpected_keys)}")
        del state_dict, checkpoint
        gc.collect()

    load_checkpoint(args.ckpt_sd, "Stable Diffusion 2.1")
    load_checkpoint(args.ckpt_lc, "Diff-ICMH BPP=2")"""

replace_once("inference_partition.py", old_loader, new_loader, "sequential checkpoint loading")

replace_once(
    "inference_partition.py",
    "model.preprocess_tag_model(control, return_ids=True)",
    "model.preprocess_tag_model(control.cpu(), return_ids=True)",
    "run RAM+ on CPU",
)

replace_once(
    "inference_partition.py",
    "    model.freeze()\\n    model.to(args.device)",
    """    model.freeze()
    # Temporarily detach RAM+ so model.to(cuda) never copies its 3 GiB to T4.
    tagger = model.preprocess_tag_model
    if args.device == "cuda" and tagger.enabled:
        del model._modules["preprocess_tag_model"]
        model.to(args.device)
        model.preprocess_tag_model = tagger.to("cpu")
    else:
        model.to(args.device)""",
    "keep RAM+ off T4 VRAM",
)

print("Compatibility patches ready.")
''')

cells[11]["source"] = lines("""
# Cell 7 — Checkpoint paths
import shutil
from pathlib import Path

REPO = Path("/content/Diff-ICMH")
CHECKPOINTS = REPO / "checkpoints"
SD_CKPT = CHECKPOINTS / "sd2p1" / "v2-1_512-ema-pruned.ckpt"
LC_RELATIVE = Path("difficmh_models/CNscale1.0_1_1_2_2_WTagGCM_bs16x1_lr0.00005_cfg7.0/model.ckpt")
LC_CKPT = CHECKPOINTS / LC_RELATIVE
RAM_CKPT = CHECKPOINTS / "ram" / "ram_plus_swin_large_14m.pth"

for path in (SD_CKPT, LC_CKPT, RAM_CKPT):
    path.parent.mkdir(parents=True, exist_ok=True)

print(f"Disk free: {shutil.disk_usage('/content').free / 1024**3:.1f} GiB")
for path in (SD_CKPT, LC_CKPT, RAM_CKPT):
    print(path, f"{path.stat().st_size / 1024**3:.2f} GiB" if path.is_file() else "missing")
""")

cells[12]["source"] = lines("""
# Cell 8 — Download exactly the three checkpoints needed for BPP=2
from huggingface_hub import hf_hub_download

downloads = [
    ("Stable Diffusion 2.1", "Manojb/stable-diffusion-2-1-base",
     "v2-1_512-ema-pruned.ckpt", SD_CKPT.parent, SD_CKPT, 4.0),
    ("Diff-ICMH BPP=2", "RuoyuFeng/Diff-ICMH", LC_RELATIVE.as_posix(),
     CHECKPOINTS, LC_CKPT, 9.0),
    ("RAM+", "xinyu1205/recognize-anything-plus-model",
     "ram_plus_swin_large_14m.pth", RAM_CKPT.parent, RAM_CKPT, 2.0),
]

def ensure_checkpoints():
    for label, repo_id, filename, local_dir, expected_path, min_gib in downloads:
        print(f"\\n=== {label} ===")
        if expected_path.is_file() and expected_path.stat().st_size >= min_gib * 1024**3:
            print(f"Already present: {expected_path}")
            continue
        local_dir.mkdir(parents=True, exist_ok=True)
        downloaded = Path(hf_hub_download(
            repo_id=repo_id, filename=filename, local_dir=str(local_dir)
        ))
        if downloaded.resolve() != expected_path.resolve():
            raise RuntimeError(f"Unexpected checkpoint path: {downloaded} != {expected_path}")
        size_gib = expected_path.stat().st_size / 1024**3
        if size_gib < min_gib:
            raise RuntimeError(f"Incomplete {label} checkpoint ({size_gib:.2f} GiB)")
        print(f"Ready: {expected_path} ({size_gib:.2f} GiB)")

ensure_checkpoints()
""")

cells[14]["source"] = lines("""
# Cell 9 — Generate the inference config from the cloned repository
from omegaconf import OmegaConf

BASE_CONFIG = REPO / "configs" / "model" / "diffeic.yaml"
COLAB_CONFIG = REPO / "configs" / "model" / "diffeic_colab_bpp2.yaml"

def ensure_colab_config():
    if not BASE_CONFIG.is_file():
        raise FileNotFoundError(f"Missing {BASE_CONFIG}. Rerun from Cell 2 after a runtime reset.")
    cfg = OmegaConf.load(BASE_CONFIG)
    cfg.params.sync_path = None
    cfg.params.synch_control = False
    cfg.params.control_stage_config.params.control_model_ratio = 1.0
    cfg.params.preprocess_semantic_config.params.enabled = False
    cfg.params.preprocess_tag_config.params.enabled = True
    cfg.params.preprocess_tag_config.params.pretrained = str(RAM_CKPT)
    cfg.params.c_cfg_scale = 5.0
    cfg.params.calculate_metrics = {}  # inference_partition.py computes its own metrics
    OmegaConf.save(cfg, COLAB_CONFIG)
    print("Saved:", COLAB_CONFIG)

ensure_colab_config()
""")

cells[15]["source"] = lines("""
# Cell 10 — Upload one image and prepare a small T4 smoke test
from io import BytesIO
from google.colab import files
from IPython.display import display
from PIL import Image, ImageOps

INPUT_DIR = REPO / "test_input_t4"
ORIGINAL_DIR = REPO / "original_input_t4"
OUTPUT_DIR = REPO / "test_output_bpp2_t4"
INFERENCE_OK = False
for directory in (INPUT_DIR, ORIGINAL_DIR, OUTPUT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

print("Upload exactly one image")
uploaded = files.upload()
if len(uploaded) != 1:
    raise RuntimeError(f"Expected one image, received {len(uploaded)} files")

original_name, payload = next(iter(uploaded.items()))
safe_name = Path(original_name).name
original_path = ORIGINAL_DIR / safe_name
with Image.open(BytesIO(payload)) as opened:
    image = ImageOps.exif_transpose(opened).convert("RGB")
original_path.write_bytes(payload)

MAX_SIDE = 256  # increase only after the first complete T4 run
width, height = image.size
scale = min(1.0, MAX_SIDE / max(width, height))
target_w = max(64, int(width * scale) // 64 * 64)
target_h = max(64, int(height * scale) // 64 * 64)
test_image = image.resize((target_w, target_h), Image.Resampling.LANCZOS)
test_path = INPUT_DIR / f"test_{target_w}x{target_h}.png"

# This dedicated folder must contain only the current test image.
for old_file in INPUT_DIR.iterdir():
    if old_file.is_file():
        old_file.unlink()
test_image.save(test_path)
print("Original:", image.size, original_path)
print("Test:", test_image.size, test_path)
display(test_image)
""")

cells[16]["source"] = lines("""
# Cell 11 — Check every required file before spending GPU time
import subprocess

required = {
    "Config": COLAB_CONFIG,
    "SD 2.1": SD_CKPT,
    "Diff-ICMH BPP=2": LC_CKPT,
    "RAM+": RAM_CKPT,
    "Input": test_path,
}
for label, path in required.items():
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing: {path}. Rerun from Cell 2.")
    print(f"OK {label:18} {path.stat().st_size / 1024**3:6.2f} GiB")

if len([p for p in INPUT_DIR.iterdir() if p.is_file()]) != 1:
    raise RuntimeError("test_input_t4 must contain exactly one file")

source = (REPO / "inference_partition.py").read_text(encoding="utf-8")
assert 'tagger.to("cpu")' in source and 'mmap=True' in source
subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv"], check=True)
print("Preflight passed")
""")

cells[19]["source"] = lines("""
## Phase 6 — Inference trên T4

Cell sau tự kiểm tra lại các checkpoint và tạo lại config nếu chúng bị mất trong
cùng một kernel. Nếu Colab đã tạo **runtime mới**, chạy lại toàn bộ notebook để
cài dependency và clone mã nguồn trước khi inference.
""")

cells[20]["source"] = lines("""
# Cell 12 — One image inference, with a persistent log for diagnosis
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/content/Diff-ICMH")
if "ensure_checkpoints" not in globals() or "ensure_colab_config" not in globals():
    raise RuntimeError("Colab kernel was reset. Choose Runtime > Run all from the first cell.")
if not (REPO / "inference_partition.py").is_file():
    raise RuntimeError("Repository is missing. Choose Runtime > Run all from the first cell.")

ensure_checkpoints()
ensure_colab_config()

INPUT_DIR = REPO / "test_input_t4"
OUTPUT_DIR = REPO / "test_output_bpp2_t4"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
if not INPUT_DIR.is_dir():
    raise RuntimeError("Input image is missing. Rerun Cell 10 to upload it.")
images = [p for p in INPUT_DIR.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}]
if len(images) != 1:
    raise RuntimeError(f"Expected one input image in {INPUT_DIR}; found {len(images)}. Rerun Cell 10.")
if "test_path" not in globals() or images[0].resolve() != test_path.resolve():
    raise RuntimeError("The current input was not prepared in this kernel. Rerun Cell 10.")
expected_output = OUTPUT_DIR / f"{images[0].stem}.png"
INFERENCE_OK = False

env = os.environ.copy()
env["PYTHONPATH"] = os.pathsep.join([
    str(REPO / "src" / "recognize-anything"), str(REPO), env.get("PYTHONPATH", "")
])
env["PYTHONUNBUFFERED"] = "1"
env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
env["OMP_NUM_THREADS"] = "2"
env["MKL_NUM_THREADS"] = "2"

command = [
    sys.executable, "-u", "inference_partition.py",
    "--ckpt_sd", str(SD_CKPT), "--ckpt_lc", str(LC_CKPT),
    "--config", str(COLAB_CONFIG), "--input", str(INPUT_DIR),
    "--output", str(OUTPUT_DIR), "--device", "cuda",
    "--sampler", "ddpm", "--steps", "20", "--seed", "231",
]
log_path = OUTPUT_DIR / "inference.log"
print("Input:", images[0])
print("Output:", expected_output)
print("Log:", log_path)

run_started_ns = time.time_ns()
with log_path.open("w", encoding="utf-8") as log:
    process = subprocess.Popen(
        command, cwd=REPO, env=env, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    try:
        for line in process.stdout:
            print(line, end="")
            log.write(line)
            log.flush()
        return_code = process.wait()
    except KeyboardInterrupt:
        process.terminate()
        process.wait()
        raise

if return_code != 0:
    raise RuntimeError(f"Inference failed (exit {return_code}). Read {log_path}")
metrics_path = OUTPUT_DIR / "bpp.txt"
if not expected_output.is_file() or not metrics_path.is_file():
    raise RuntimeError(f"Inference exited without the expected image/metrics. Read {log_path}")
if expected_output.stat().st_mtime_ns < run_started_ns or metrics_path.stat().st_mtime_ns < run_started_ns:
    raise RuntimeError(f"Only stale output was found. Read {log_path}")

INFERENCE_OK = True
print("Inference complete:", expected_output)
""")

cells[21]["source"] = lines("""
# Cell 13 — Display image and metrics; download result and log
import shutil
from google.colab import files
from IPython.display import display
from PIL import Image

if not globals().get("INFERENCE_OK", False):
    raise RuntimeError("Inference has not completed in this kernel. Run Cell 12 first.")
print((OUTPUT_DIR / "bpp.txt").read_text(encoding="utf-8"))
print("Reconstruction:", expected_output)
display(Image.open(expected_output))

zip_path = Path(shutil.make_archive("/content/Diff_ICMH_BPP2_T4_result", "zip", OUTPUT_DIR))
print("ZIP:", zip_path)
files.download(str(zip_path))
""")

cells[22]["source"] = lines("""
## Ghi chú

- Sau khi tạo runtime Colab mới, dùng **Runtime → Run all**. File trong `/content`
  có thể mất, kể cả khi output notebook vẫn hiển thị dòng “Ready” cũ.
- Nếu lần chạy 256 px thành công, có thể nâng `MAX_SIDE` ở Cell 10 rồi chạy lại
  Cell 10–13. Mức 512 px có thể vượt VRAM T4.
- Kết quả, `bpp.txt`, bitstream và `inference.log` nằm trong ZIP ở Cell 13.
""")

# Old reconnect diagnostics were out of order and displayed stale state.
for index in (18, 17):
    del cells[index]

for cell in cells:
    if cell["cell_type"] == "code":
        cell["execution_count"] = None
        cell["outputs"] = []

TARGET.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(TARGET)
