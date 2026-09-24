"""Generate the small, reviewable Colab entry notebook from source strings."""
from __future__ import annotations

import json
from pathlib import Path


def markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source}


cells = [
    markdown(
        "# Wild-Diff-ICMH — fine-tune Kgalagadi trên Colab\n\n"
        "Chạy từng cell. Mặc định notebook chỉ smoke-test **20 optimizer step cho KGA:A01**; "
        "không tự đốt tài nguyên chạy cả 20 địa điểm."
    ),
    code(
        "from google.colab import drive\n"
        "drive.mount('/content/drive')\n"
    ),
    code(
        "from pathlib import Path\n"
        "import os, shutil, subprocess, sys, torch\n\n"
        "REPO_URL = 'https://github.com/TranDuon/Wild-Diff-ICMH.git'\n"
        "BRANCH = 'main'\n"
        "REPO = Path('/content/Wild-Diff-ICMH')\n"
        "if (REPO / '.git').is_dir():\n"
        "    subprocess.run(['git', 'pull', '--ff-only', 'origin', BRANCH], cwd=REPO, check=True)\n"
        "else:\n"
        "    subprocess.run(['git', 'clone', '--branch', BRANCH, REPO_URL, str(REPO)], check=True)\n"
        "os.chdir(REPO)\n"
        "assert torch.cuda.is_available(), 'Runtime > Change runtime type > chọn GPU'\n"
        "gpu = torch.cuda.get_device_properties(0)\n"
        "print(torch.__version__, torch.version.cuda, gpu.name, f'{gpu.total_memory/2**30:.1f} GiB')\n"
        "subprocess.run(['nvidia-smi'], check=False)\n"
    ),
    code(
        "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements-colab.txt'], check=True)\n"
        "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', 'compressai==1.2.8'], check=True)\n"
        "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', '-e', 'src/recognize-anything'], check=True)\n"
        "print('Cài xong. Nếu Colab yêu cầu restart runtime, restart rồi chạy lại từ đầu; các lệnh là idempotent.')\n"
    ),
    code(
        "DRIVE_ROOT = Path('/content/drive/MyDrive/wild_diff_icmh')\n"
        "DRIVE_IMAGES = DRIVE_ROOT / 'images'\n"
        "LOCAL_IMAGES = Path('/content/data/wild_diff_icmh/images')\n"
        "LOCAL_IMAGES.parent.mkdir(parents=True, exist_ok=True)\n"
        "assert (DRIVE_IMAGES / 'snapshot_kgalagadi').is_dir(), f'Không thấy dữ liệu: {DRIVE_IMAGES}'\n"
        "subprocess.run(['rsync', '-a', '--info=progress2', str(DRIVE_IMAGES) + '/', str(LOCAL_IMAGES) + '/'], check=True)\n"
        "subprocess.run([sys.executable, 'tools/data/split_check.py', 'data/manifests/kgalagadi_site_split.jsonl', '--build-info', 'data/manifests/build_info.json'], check=True)\n"
    ),
    code(
        "from huggingface_hub import hf_hub_download\n"
        "CKPT_ROOT = DRIVE_ROOT / 'checkpoints'\n"
        "BPP_WEIGHT = 2\n"
        "folder = f'CNscale1.0_1_1_{BPP_WEIGHT}_2_WTagGCM_bs16x1_lr0.00005_cfg7.0'\n"
        "hf_hub_download(repo_id='Manojb/stable-diffusion-2-1-base', filename='v2-1_512-ema-pruned.ckpt', local_dir=CKPT_ROOT/'sd2p1')\n"
        "hf_hub_download(repo_id='xinyu1205/recognize-anything-plus-model', filename='ram_plus_swin_large_14m.pth', local_dir=CKPT_ROOT/'ram')\n"
        "hf_hub_download(repo_id='RuoyuFeng/Diff-ICMH', filename=f'difficmh_models/{folder}/model.ckpt', local_dir=CKPT_ROOT)\n"
        "AUTHOR_CKPT = CKPT_ROOT / 'difficmh_models' / folder / 'model.ckpt'\n"
        "repo_checkpoints = REPO / 'checkpoints'\n"
        "if not repo_checkpoints.exists():\n"
        "    repo_checkpoints.symlink_to(CKPT_ROOT, target_is_directory=True)\n"
        "elif repo_checkpoints.resolve() != CKPT_ROOT.resolve():\n"
        "    raise RuntimeError(f'{repo_checkpoints} đã tồn tại nhưng không trỏ tới {CKPT_ROOT}')\n"
        "print(AUTHOR_CKPT)\n"
    ),
    code(
        "SITE = 'KGA:A01'\n"
        "RAM_CKPT = CKPT_ROOT / 'ram' / 'ram_plus_swin_large_14m.pth'\n"
        "TAGS_PATH = DRIVE_ROOT / 'tags' / f\"{SITE.replace(':', '_')}.jsonl\"\n"
        "tag_command = [sys.executable, 'tools/precompute_ram_tags.py', '--data-root', str(LOCAL_IMAGES), '--checkpoint', str(RAM_CKPT), '--site-id', SITE, '--output', str(TAGS_PATH)]\n"
        "subprocess.run(tag_command, cwd=REPO, check=True)\n"
    ),
    code(
        "RUN_DIR = DRIVE_ROOT / 'runs' / 'h1' / SITE.split(':')[-1]\n"
        "env = os.environ.copy()\n"
        "env.update(WILD_DATA_ROOT=str(LOCAL_IMAGES), KGA_SITE_ID=SITE, KGA_TAGS=str(TAGS_PATH), BPP_WEIGHT=str(BPP_WEIGHT), WILD_RUN_DIR=str(RUN_DIR), PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')\n"
        "command = [sys.executable, 'train.py', '--config', 'configs/train_kgalagadi_colab.yaml', '--init-checkpoint', str(AUTHOR_CKPT), 'lightning.trainer.max_steps=20', 'lightning.trainer.val_check_interval=10', 'lightning.trainer.check_val_every_n_epoch=1', 'lightning.trainer.limit_val_batches=2']\n"
        "print(' '.join(command))\n"
        "subprocess.run(command, cwd=REPO, env=env, check=True)\n"
    ),
    markdown(
        "## Sau khi smoke test thành công\n\n"
        "Xem VRAM/thời gian trong log. Sau đó bỏ bốn override smoke-test hoặc dùng "
        "`tools/train_kgalagadi_sites.py --max-sites 1`. "
        "Checkpoint nằm trên Drive và lần chạy sau tự resume từ `last.ckpt`. Xem `COLAB_TRAINING.md` cho H2, H3 và đánh giá."
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"name": "Wild_Diff_ICMH_Kgalagadi_Train.ipynb", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

output = Path(__file__).resolve().parents[1] / "Wild_Diff_ICMH_Kgalagadi_Train.ipynb"
output.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(output)
