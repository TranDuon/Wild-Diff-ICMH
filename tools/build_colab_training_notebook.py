"""Generate the ordered Colab entry notebook from reviewable source strings."""
from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


def markdown(source: str):
    return {"cell_type": "markdown", "metadata": {}, "source": dedent(source).strip()}


def code(source: str):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).strip() + "\n",
    }


cells = [
    markdown(
        """
        # Wild-Diff-ICMH — Kgalagadi trên Google Colab

        ## Cách chạy

        **Mỗi khi Colab cấp runtime mới hoặc runtime bị ngắt:** chọn GPU rồi chạy lần lượt
        từ **Bước 1 → Bước 7**. Không chạy lại cell cài đặt cũ dùng trực tiếp
        `requirements-colab.txt` vì nó có thể thay NumPy của Colab và gây lỗi ABI.

        - Bước 1–3: phải chạy ở mỗi runtime mới.
        - Bước 4–5: cũng chạy ở mỗi runtime mới; cell tự bỏ qua file đã có trong `/content`.
        - Bước 6: tag lưu trên Drive; cell tự bỏ qua nếu đã đủ, hoặc tiếp tục phần còn thiếu.
        - Bước 7: chạy smoke test 20 step; checkpoint trên Drive và tự resume.

        Trong **cùng một runtime**, nếu một cell đã có dấu tích xanh thì không cần chạy lại,
        trừ khi cell đó vừa báo lỗi.
        """
    ),
    markdown("## Bước 1 — Gắn Google Drive (mỗi runtime mới)"),
    code(
        """
        from google.colab import drive

        drive.mount('/content/drive')
        """
    ),
    markdown("## Bước 2 — Lấy mã nguồn và kiểm tra GPU (mỗi runtime mới)"),
    code(
        """
        from pathlib import Path
        import os
        import subprocess
        import sys

        REPO_URL = 'https://github.com/TranDuon/Wild-Diff-ICMH.git'
        BRANCH = 'main'
        REPO = Path('/content/Wild-Diff-ICMH')

        if (REPO / '.git').is_dir():
            subprocess.run(['git', 'pull', '--ff-only', 'origin', BRANCH], cwd=REPO, check=True)
        else:
            subprocess.run(['git', 'clone', '--branch', BRANCH, REPO_URL, str(REPO)], check=True)

        os.chdir(REPO)
        subprocess.run(['nvidia-smi'], check=True)
        print('Mã nguồn:', REPO)
        """
    ),
    markdown(
        """
        ## Bước 3 — Cài dependencies an toàn (mỗi runtime mới)

        Cell này chủ động giữ nguyên NumPy/SciPy/Torch có sẵn của Colab,
        cài đủ dependency runtime của CompressAI, rồi import `TagGCM` để bắt lỗi ngay tại đây.
        """
    ),
    code(
        """
        source_requirements = REPO / 'requirements-colab.txt'
        safe_requirements = Path('/content/requirements-colab-safe.txt')
        core_constraints = Path('/content/colab-core-constraints.txt')

        from importlib import metadata as importlib_metadata

        core_distributions = ('numpy', 'scipy', 'torch', 'torchvision')
        core_versions_before = {
            name: importlib_metadata.version(name)
            for name in core_distributions
        }

        filtered_lines = []
        for line in source_requirements.read_text(encoding='utf-8').splitlines():
            normalized = line.strip().lower()
            if normalized.startswith(('numpy', 'scipy')):
                print('Giữ bản Colab, bỏ qua:', line)
                continue
            filtered_lines.append(line)

        safe_requirements.write_text(
            '\\n'.join(filtered_lines) + '\\n',
            encoding='utf-8',
        )

        core_constraints.write_text(
            '\\n'.join(
                f'{name}=={version}'
                for name, version in core_versions_before.items()
            ) + '\\n',
            encoding='utf-8',
        )

        subprocess.run([
            sys.executable, '-m', 'pip', 'install', '-q',
            '-r', str(safe_requirements),
            '-c', str(core_constraints),
        ], check=True)
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', '-q',
            '--no-deps', '--no-build-isolation',
            'compressai==1.2.8',
        ], check=True)
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', '-q', '--no-deps',
            '-e', 'src/recognize-anything',
        ], cwd=REPO, check=True)

        import numpy as np
        import numpy.random as npr
        import scipy
        import torch
        import torch_geometric
        import pytorch_msssim
        import compressai.ans
        import compressai.entropy_models
        import compressai.losses
        import compressai.zoo
        from model.lfgcm import TagGCM

        core_versions_after = {
            name: importlib_metadata.version(name)
            for name in core_distributions
        }
        assert core_versions_after == core_versions_before, (
            'Dependency install changed Colab core packages: '
            f'before={core_versions_before}, after={core_versions_after}'
        )
        assert torch.cuda.is_available(), 'Vào Runtime > Change runtime type > chọn GPU'
        gpu = torch.cuda.get_device_properties(0)
        print('Python:', sys.version.split()[0])
        print('NumPy:', np.__version__, np.__file__)
        print('SciPy:', scipy.__version__)
        print('Torch:', torch.__version__, torch.version.cuda)
        print('PyG:', torch_geometric.__version__)
        print('Core packages preserved:', core_versions_after)
        print('GPU:', gpu.name, f'{gpu.total_memory / 2**30:.1f} GiB')
        print('Kiểm tra numpy.random:', npr.rand(3))
        print('THÀNH CÔNG: môi trường và TagGCM đã sẵn sàng.')
        """
    ),
    markdown(
        """
        ## Bước 4 — Chép ảnh từ Drive vào ổ cục bộ (mỗi runtime mới)

        `/content` bị xóa khi runtime ngắt, nên runtime mới phải chép lại. Cell dùng 6 luồng,
        có thanh phần trăm và chỉ chép các file còn thiếu. Sau đó nó luôn chạy `split_check.py`.
        """
    ),
    code(
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from tqdm.auto import tqdm

        DRIVE_ROOT = Path('/content/drive/MyDrive/wild_diff_icmh')
        DRIVE_IMAGES = DRIVE_ROOT / 'images' / 'snapshot_kgalagadi'
        LOCAL_IMAGES = Path('/content/data/wild_diff_icmh/images')
        LOCAL_SITE_IMAGES = LOCAL_IMAGES / 'snapshot_kgalagadi'

        assert DRIVE_IMAGES.is_dir(), f'Không thấy dữ liệu: {DRIVE_IMAGES}'
        LOCAL_SITE_IMAGES.mkdir(parents=True, exist_ok=True)

        print('Đang quét file trên Drive...')
        source_files = [path for path in DRIVE_IMAGES.rglob('*') if path.is_file()]
        pending = []
        for src in source_files:
            dst = LOCAL_SITE_IMAGES / src.relative_to(DRIVE_IMAGES)
            if not dst.exists() or dst.stat().st_size != src.stat().st_size:
                pending.append((src, dst, src.stat().st_size))

        remaining_bytes = sum(size for _, _, size in pending)
        print('Tổng file:', len(source_files))
        print('Còn phải copy:', len(pending), 'file')
        print('Dung lượng còn lại:', f'{remaining_bytes / 2**30:.2f} GiB')

        def copy_one(item):
            src, dst, size = item
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(dst.name + '.part')
            try:
                with src.open('rb') as fin, tmp.open('wb') as fout:
                    while chunk := fin.read(8 * 1024 * 1024):
                        fout.write(chunk)
                os.replace(tmp, dst)
                return size
            except Exception:
                if tmp.exists():
                    tmp.unlink()
                raise

        if pending:
            with ThreadPoolExecutor(max_workers=6) as executor:
                futures = [executor.submit(copy_one, item) for item in pending]
                with tqdm(
                    total=remaining_bytes,
                    unit='B', unit_scale=True, unit_divisor=1024,
                    desc='Copy Kgalagadi',
                ) as progress:
                    for future in as_completed(futures):
                        progress.update(future.result())
        else:
            print('Ảnh cục bộ đã đầy đủ, không cần copy lại.')

        subprocess.run([
            sys.executable, 'tools/data/split_check.py',
            'data/manifests/kgalagadi_site_split.jsonl',
            '--build-info', 'data/manifests/build_info.json',
        ], cwd=REPO, check=True)
        print('Ảnh và manifest đã sẵn sàng.')
        """
    ),
    markdown(
        """
        ## Bước 5 — Khôi phục 3 checkpoint từ Drive vào `/content` (mỗi runtime mới)

        Checkpoint gốc đã sao lưu trên Drive. Cell có tiến trình, chép song song 2 file và
        tự bỏ qua file cục bộ đã đủ. Không cần tải lại Hugging Face.
        """
    ),
    code(
        """
        DRIVE_CKPT_ROOT = DRIVE_ROOT / 'checkpoints'
        CKPT_ROOT = Path('/content/data/wild_diff_icmh/checkpoints')
        BPP_WEIGHT = 2
        folder = f'CNscale1.0_1_1_{BPP_WEIGHT}_2_WTagGCM_bs16x1_lr0.00005_cfg7.0'

        checkpoint_relatives = [
            Path('sd2p1/v2-1_512-ema-pruned.ckpt'),
            Path('ram/ram_plus_swin_large_14m.pth'),
            Path('difficmh_models') / folder / 'model.ckpt',
        ]
        backup_items = []
        for relative in checkpoint_relatives:
            src = DRIVE_CKPT_ROOT / relative
            dst = CKPT_ROOT / relative
            assert src.is_file(), f'Thiếu checkpoint trên Drive: {src}'
            if not dst.exists() or dst.stat().st_size != src.stat().st_size:
                backup_items.append((src, dst, src.stat().st_size))

        total_bytes = sum(size for _, _, size in backup_items)
        if backup_items:
            def copy_checkpoint(item, progress):
                src, dst, _ = item
                dst.parent.mkdir(parents=True, exist_ok=True)
                tmp = dst.with_name(dst.name + '.part')
                try:
                    with src.open('rb') as fin, tmp.open('wb') as fout:
                        while chunk := fin.read(8 * 1024 * 1024):
                            fout.write(chunk)
                            progress.update(len(chunk))
                    os.replace(tmp, dst)
                    return dst
                except Exception:
                    if tmp.exists():
                        tmp.unlink()
                    raise

            with tqdm(
                total=total_bytes,
                unit='B', unit_scale=True, unit_divisor=1024,
                desc='Khôi phục checkpoint',
            ) as progress:
                with ThreadPoolExecutor(max_workers=2) as executor:
                    futures = [
                        executor.submit(copy_checkpoint, item, progress)
                        for item in backup_items
                    ]
                    for future in as_completed(futures):
                        print('\\nĐã khôi phục:', future.result())
        else:
            print('Checkpoint cục bộ đã đầy đủ, không cần copy lại.')

        AUTHOR_CKPT = CKPT_ROOT / 'difficmh_models' / folder / 'model.ckpt'
        RAM_CKPT = CKPT_ROOT / 'ram' / 'ram_plus_swin_large_14m.pth'
        repo_checkpoints = REPO / 'checkpoints'
        if not repo_checkpoints.exists():
            repo_checkpoints.symlink_to(CKPT_ROOT, target_is_directory=True)
        elif repo_checkpoints.resolve() != CKPT_ROOT.resolve():
            raise RuntimeError(f'{repo_checkpoints} không trỏ tới {CKPT_ROOT}')

        print('RAM++:', RAM_CKPT)
        print('Diff-ICMH:', AUTHOR_CKPT)
        print('Checkpoint đã sẵn sàng.')
        """
    ),
    markdown(
        """
        ## Bước 6 — Tạo RAM++ tags cho KGA:A01

        File tag nằm trên Drive nên **không mất khi runtime ngắt**. Cell tự resume và hiển thị
        thanh tiến trình; nếu đã đủ tag thì bỏ qua hoàn toàn.
        """
    ),
    code(
        """
        import json

        SITE = 'KGA:A01'
        TAGS_PATH = DRIVE_ROOT / 'tags' / f"{SITE.replace(':', '_')}.jsonl"
        manifest_path = REPO / 'data/manifests/kgalagadi_site_split.jsonl'

        with manifest_path.open('r', encoding='utf-8') as stream:
            site_rows = [
                json.loads(line) for line in stream
                if line.strip() and json.loads(line).get('site_id') == SITE
            ]
        expected_ids = {row['image_id'] for row in site_rows}

        completed_ids = set()
        if TAGS_PATH.is_file():
            with TAGS_PATH.open('r', encoding='utf-8') as stream:
                for line in stream:
                    try:
                        completed_ids.add(json.loads(line)['image_id'])
                    except (json.JSONDecodeError, KeyError):
                        pass

        completed_count = len(expected_ids & completed_ids)
        if expected_ids <= completed_ids:
            print(f'Tags đã đủ: {completed_count}/{len(site_rows)} — bỏ qua.')
        else:
            print(f'Tiếp tục tạo tags: {completed_count}/{len(site_rows)} đã có.')
            tag_command = [
                sys.executable, '-u', 'tools/precompute_ram_tags.py',
                '--data-root', str(LOCAL_IMAGES),
                '--checkpoint', str(RAM_CKPT),
                '--site-id', SITE,
                '--output', str(TAGS_PATH),
                '--batch-size', '8',
                '--num-workers', '4',
            ]
            subprocess.run(tag_command, cwd=REPO, check=True)

        print('File tags:', TAGS_PATH)
        """
    ),
    markdown(
        """
        ## Bước 7 — Smoke test training 20 step

        Chỉ chạy sau khi Bước 1–6 đều thành công. Log training có tiến trình của Lightning.
        Nếu đã có checkpoint trong thư mục run trên Drive, cấu hình sẽ tự resume.
        """
    ),
    code(
        """
        RUN_DIR = DRIVE_ROOT / 'runs' / 'h1' / SITE.split(':')[-1]
        env = os.environ.copy()
        env.update(
            WILD_DATA_ROOT=str(LOCAL_IMAGES),
            KGA_SITE_ID=SITE,
            KGA_TAGS=str(TAGS_PATH),
            BPP_WEIGHT=str(BPP_WEIGHT),
            WILD_RUN_DIR=str(RUN_DIR),
            PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
        )
        command = [
            sys.executable, '-u', 'train.py',
            '--config', 'configs/train_kgalagadi_colab.yaml',
            '--init-checkpoint', str(AUTHOR_CKPT),
            'lightning.trainer.max_steps=20',
            'lightning.trainer.val_check_interval=10',
            'lightning.trainer.check_val_every_n_epoch=1',
            'lightning.trainer.limit_val_batches=2',
        ]
        print('Lệnh chạy:', ' '.join(command))
        subprocess.run(command, cwd=REPO, env=env, check=True)
        """
    ),
    markdown(
        """
        ## Sau khi smoke test thành công

        Không sửa các cell chuẩn bị phía trên. Khi chạy thí nghiệm dài, đổi riêng cell training
        theo H1/H2/H3. Checkpoint nằm trong `MyDrive/wild_diff_icmh/runs/`; runtime mới vẫn
        chạy lại Bước 1–6 trước, sau đó training tự resume checkpoint mới nhất.
        """
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
