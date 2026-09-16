"""Build the step-1 Colab baseline notebook from inspected notebook cells."""

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KAGGLE = json.loads((ROOT / "Diff_ICMH_Kaggle_Quick_Run.ipynb").read_text(encoding="utf-8"))
T4 = json.loads((ROOT / "Diff_ICMH_Colab_T4_Fixed.ipynb").read_text(encoding="utf-8"))
TARGET = ROOT / "Diff_ICMH_Colab_Baseline.ipynb"


def source(notebook, index):
    return "".join(notebook["cells"][index]["source"])


def markdown(value):
    return {"cell_type": "markdown", "metadata": {}, "source": value}


def code(value):
    ast.parse(value)
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": value}


preflight = '''from pathlib import Path
import os
import platform
import shutil
import subprocess
import sys
from urllib.request import Request, urlopen

import torch
import torchvision

WORKING_DIR = Path('/content')
if not WORKING_DIR.is_dir():
    raise RuntimeError('Run this notebook on Google Colab.')
if sys.version_info[:2] != (3, 12):
    raise RuntimeError('Select Runtime > Change runtime type > Runtime version 2026.07 (Python 3.12), then rerun.')
if not torch.cuda.is_available():
    raise RuntimeError('Select a GPU runtime in Runtime > Change runtime type, then rerun.')

GPU_NAME = torch.cuda.get_device_name(0)
GPU_GIB = torch.cuda.get_device_properties(0).total_memory / 2**30
free_gib = shutil.disk_usage(WORKING_DIR).free / 2**30
if GPU_GIB < 14:
    raise RuntimeError(f'{GPU_NAME} has {GPU_GIB:.1f} GiB VRAM; this full checkpoint needs a T4/L4 class GPU.')
if free_gib < 24:
    raise RuntimeError(f'Only {free_gib:.1f} GiB free; three checkpoints and temporary files need at least 24 GiB.')

for url in ('https://github.com', 'https://huggingface.co'):
    with urlopen(Request(url, headers={'User-Agent': 'Wild-Diff-ICMH-Colab-baseline/1.0'}), timeout=20) as response:
        if response.status >= 400:
            raise RuntimeError(f'HTTPS preflight failed: {url}, status={response.status}')

BASE_TORCH_VERSION = torch.__version__
BASE_TORCHVISION_VERSION = torchvision.__version__
BASE_CUDA_VERSION = torch.version.cuda
print('Python:', sys.version.split()[0], platform.platform())
print('Torch:', BASE_TORCH_VERSION, 'Torchvision:', BASE_TORCHVISION_VERSION)
print('CUDA ABI:', BASE_CUDA_VERSION, 'GPU:', GPU_NAME, f'{GPU_GIB:.1f} GiB')
print(f'Free disk: {free_gib:.1f} GiB')
subprocess.run(['nvidia-smi'], check=False)
'''

patches = source(T4, 8).replace('REPO = Path("/content/Diff-ICMH")', 'REPO = REPO_DIR')
patches = patches.replace(
    '    model.freeze()\n    # Temporarily detach RAM+',
    '    model.freeze()\n    assert all(not p.requires_grad for p in model.model.diffusion_model.parameters()), "SD UNet is not frozen"\n    # Temporarily detach RAM+',
)

clone = source(KAGGLE, 4).replace(
    'https://github.com/TranDuon/Wild-Diff-ICMH.git',
    'https://github.com/RuoyuFeng/Diff-ICMH.git',
).replace(
    'aebaff6f61d0253e09e3f482d5887aa50e0a539a',
    '01366f0afe8983eab423bb9a804beb2e02043100',
)
dependencies = source(KAGGLE, 6).replace('Kaggle ABI', 'Colab ABI').replace(
    'Start a current Kaggle GPU image', 'Start a current Colab GPU runtime'
)
dependencies = dependencies.replace(
    "pip_install('compressai==1.2.8'",
    "pip_install('torch-geometric>=2.6,<3')\npip_install('compressai==1.2.8'",
)
compat = source(KAGGLE, 8).replace(
    'compat_env = os.environ.copy()',
    "write_compat('pytorch_lightning/utilities/rank_zero.py', '''\n"
    "    from lightning.pytorch.utilities.rank_zero import rank_zero_only\n"
    "    __all__ = ['rank_zero_only']\n"
    "''')\n\ncompat_env = os.environ.copy()",
)

config = '''from omegaconf import OmegaConf

BASE_CONFIG = REPO_DIR / 'configs' / 'model' / 'diffeic.yaml'
COLAB_CONFIG = REPO_DIR / 'configs' / 'model' / f'diffeic_colab_bpp{BPP_WEIGHT}.yaml'
cfg = OmegaConf.load(BASE_CONFIG)
# The CLI loads SD and Diff-ICMH checkpoints sequentially. Avoid a third SD load in __init__.
cfg.params.sync_path = None
cfg.params.synch_control = False
cfg.params.control_stage_config.params.control_model_ratio = CONTROL_MODULE_SCALE
cfg.params.preprocess_semantic_config.params.enabled = False
cfg.params.preprocess_tag_config.params.enabled = True
cfg.params.preprocess_tag_config.params.pretrained = str(CKPT_RAM)
cfg.params.c_cfg_scale = CFG_SCALE
cfg.params.calculate_metrics = {}
OmegaConf.save(cfg, COLAB_CONFIG)
print('Inference config:', COLAB_CONFIG)
'''

stage = '''from PIL import Image

# This is a checkpoint/load smoke test, not a camera-trap benchmark.
SOURCE_IMAGE = REPO_DIR / 'data' / 'kodak_subset' / 'kodim01.png'
if not SOURCE_IMAGE.is_file():
    raise FileNotFoundError(SOURCE_IMAGE)
INPUT_DIR = WORKING_DIR / 'difficmh_baseline_input'
OUTPUT_DIR = WORKING_DIR / f'difficmh_baseline_bpp{BPP_WEIGHT}_{STEPS}steps'
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
for old in INPUT_DIR.iterdir():
    if old.is_file():
        old.unlink()

with Image.open(SOURCE_IMAGE) as source_image:
    source_rgb = source_image.convert('RGB')
    width, height = source_rgb.size
    crop_size = 256
    left, top = (width - crop_size) // 2, (height - crop_size) // 2
    if left < 0 or top < 0:
        raise RuntimeError(f'{SOURCE_IMAGE} is smaller than 256x256')
    smoke_crop = source_rgb.crop((left, top, left + crop_size, top + crop_size))

STAGED_IMAGE = INPUT_DIR / 'kodim01_center256.png'
smoke_crop.save(STAGED_IMAGE)
print('Source:', SOURCE_IMAGE, (width, height))
print('Staged 256x256 crop:', STAGED_IMAGE)
'''

run = '''# The first pass checks full model loading, compression, decompression, and decoding.
# For a quality baseline after this succeeds, set SMOKE_TEST=False above (50 steps).
env = compat_env.copy()
env['CUDA_VISIBLE_DEVICES'] = '0'
env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
env['OMP_NUM_THREADS'] = '2'
command = [
    sys.executable, '-u', 'inference_partition.py',
    '--ckpt_sd', str(CKPT_SD), '--ckpt_lc', str(CKPT_LC),
    '--config', str(COLAB_CONFIG), '--input', str(INPUT_DIR),
    '--output', str(OUTPUT_DIR), '--sampler', 'ddpm',
    '--steps', str(STEPS), '--seed', str(SEED), '--device', 'cuda',
]
print('Running:', command)
with subprocess.Popen(
    command, cwd=REPO_DIR, env=env, stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT, text=True, bufsize=1,
) as process:
    for line in process.stdout:
        print(line, end='')
    return_code = process.wait()
if return_code:
    raise RuntimeError(f'Diff-ICMH inference failed with exit code {return_code}.')
'''

report = """import json
import textwrap
from IPython.display import display
from PIL import Image

RECONSTRUCTION = OUTPUT_DIR / f'{STAGED_IMAGE.stem}.png'
BITSTREAM = OUTPUT_DIR / 'data' / STAGED_IMAGE.stem
CLI_METRICS = OUTPUT_DIR / 'bpp.txt'
for item in (RECONSTRUCTION, BITSTREAM, CLI_METRICS):
    if not item.is_file() or item.stat().st_size == 0:
        raise RuntimeError(f'Missing or empty baseline output: {item}')
with Image.open(STAGED_IMAGE) as original, Image.open(RECONSTRUCTION) as decoded:
    if decoded.size != original.size:
        raise RuntimeError(f'Reconstruction size {decoded.size} != input {original.size}')
    width, height = original.size
    display(original, decoded)

metric_program = textwrap.dedent('''
    import json, sys, pyiqa, torch
    from PIL import Image
    from torchvision.transforms.functional import pil_to_tensor
    image_paths = sys.argv[1:3]
    tensors = [pil_to_tensor(Image.open(p).convert('RGB')).unsqueeze(0).float().cuda()/255 for p in image_paths]
    results = {name: float(pyiqa.create_metric(name, device='cuda')(*tensors).item())
               for name in ('psnr', 'ssim', 'lpips')}
    print('METRICS_JSON=' + json.dumps(results, sort_keys=True))
''')
metric_run = subprocess.run(
    [sys.executable, '-c', metric_program, str(STAGED_IMAGE), str(RECONSTRUCTION)],
    cwd=REPO_DIR, env=compat_env, check=True, text=True, capture_output=True,
)
metric_line = next((line for line in metric_run.stdout.splitlines() if line.startswith('METRICS_JSON=')), None)
if metric_line is None:
    raise RuntimeError(f'Metric subprocess returned no JSON: {metric_run.stdout} {metric_run.stderr}')
metrics = json.loads(metric_line.split('=', 1)[1])
actual_bpp = 8 * BITSTREAM.stat().st_size / (width * height)
baseline = {
    'status': 'gpu_verified_baseline' if not SMOKE_TEST else 'gpu_verified_smoke',
    'repository_commit': checked_out_ref,
    'checkpoint_repo': 'RuoyuFeng/Diff-ICMH',
    'checkpoint_relative_path': f'difficmh_models/{FOLDER_NAME}/model.ckpt',
    'sd_checkpoint': CKPT_SD.name,
    'bpp_weight': BPP_WEIGHT,
    'control_model_ratio': CONTROL_MODULE_SCALE,
    'tag_guidance_enabled': True,
    'sd_unet_frozen': True,
    'source_image': str(SOURCE_IMAGE.relative_to(REPO_DIR)),
    'input_transform': 'center crop 256x256',
    'input_pixels': [width, height],
    'sampler': 'ddpm', 'sampler_steps': STEPS, 'seed': SEED,
    'bitstream_bytes': BITSTREAM.stat().st_size,
    'bpp_actual': actual_bpp,
    **metrics,
}
REPORT_PATH = OUTPUT_DIR / 'baseline_report.json'
REPORT_PATH.write_text(json.dumps(baseline, indent=2, sort_keys=True) + '\\n', encoding='utf-8')
print(json.dumps(baseline, indent=2, sort_keys=True))
print('CLI report:', CLI_METRICS.read_text(encoding='utf-8'))
print('Saved:', REPORT_PATH)
"""

export = '''from IPython.display import FileLink, display

archive_base = WORKING_DIR / f'difficmh_baseline_bpp{BPP_WEIGHT}_{STEPS}steps'
archive_path = Path(shutil.make_archive(str(archive_base), 'zip', root_dir=OUTPUT_DIR))
print('Downloadable baseline package:', archive_path)
display(FileLink(str(archive_path)))
'''

cells = [
    markdown("# Diff-ICMH — bước 1: baseline checkpoint gốc trên Colab\n\n"
             "Chọn GPU T4 và runtime **2026.07 (Python 3.12)**, rồi chạy **Runtime → Run all**. Lượt mặc định dùng BPP_WEIGHT=2, "
             "crop giữa 256×256 của `kodim01.png`, 10 bước DDPM để kiểm tra nạp "
             "checkpoint, nén, giải nén và giải mã. Sau khi thành công, đặt "
             "`SMOKE_TEST=False` để chạy 50 bước. Notebook tạo `baseline_report.json`, "
             "ảnh tái tạo, bitstream và ZIP. Đây chưa phải đánh giá Camera Trap.\n"),
    markdown("## 1. Kiểm tra GPU, dung lượng và kết nối"),
    code(preflight),
    markdown("## 2. Lấy đúng phiên bản mã nguồn đã kiểm tra"),
    code(clone),
    markdown("## 3. Cài thư viện và kiểm tra import"),
    code(dependencies),
    code(compat),
    markdown("## 4. Giảm bộ nhớ T4 và giữ SD 2.1 đóng băng"),
    code(patches),
    markdown("## 5. Tải đúng ba checkpoint cho một mức nén"),
    code(source(KAGGLE, 10)),
    code(config),
    markdown("## 6. Tạo một ảnh thử cố định từ Kodak"),
    code(stage),
    markdown("## 7. Chạy baseline gốc"),
    code(run),
    markdown("## 8. Kiểm tra ảnh, bitstream và chỉ số"),
    code(report),
    code(export),
    markdown("**Trạng thái:** Notebook này được kiểm tra tĩnh tại workspace. "
             "Chỉ coi baseline hoàn tất sau khi Colab chạy đến `baseline_report.json` "
             "và có ảnh cùng bitstream không rỗng."),
]

notebook = {
    "cells": cells,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python"}},
    "nbformat": 4,
    "nbformat_minor": 5,
}
TARGET.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(TARGET)
