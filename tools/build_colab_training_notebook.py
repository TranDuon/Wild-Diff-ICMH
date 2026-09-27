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
        - Bước 7: chạy smoke test 20 step và kiểm tra checkpoint có optimizer/global step.
        - Bước 8: chạy thêm đúng 1 step từ checkpoint để xác nhận resume toàn trạng thái.
        - Bước 9: decode 2 ảnh test, tính metric và ghi `results/results.jsonl` trên Drive.
        - Bước 10: audit metadata 100 ảnh và ghi báo cáo khép Phase 1; không chạy thêm GPU job.

        Trong **cùng một runtime**, nếu một cell đã có dấu tích xanh thì không cần chạy lại,
        trừ khi cell đó vừa báo lỗi.

        **Khi có bản sửa mới trên GitHub:** chỉ chạy cell **Bước 2A**, rồi chạy lại đúng
        cell vừa lỗi. Không cần copy cell pull tạm và không cần chạy lại các bước đã thành công.
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
        ### Bước 2A — Cập nhật bản sửa mới nhất (chỉ chạy khi được báo đã push)

        Cell cố định này thay cho mọi cell pull tạm. Trong cùng runtime, chạy cell này sau
        khi có bản sửa trên GitHub, rồi chạy lại **đúng cell vừa lỗi**. Cell tự in commit
        trước/sau và xác nhận thư mục code đã khớp hoàn toàn với `origin/main`.
        """
    ),
    code(
        """
        assert (REPO / '.git').is_dir(), 'Hãy chạy Bước 2 trước để clone repository.'

        local_changes = subprocess.check_output(
            ['git', 'status', '--porcelain'], cwd=REPO, text=True
        ).strip()
        if local_changes:
            raise RuntimeError(
                'Thư mục code Colab có sửa cục bộ nên chưa thể pull an toàn:\\n'
                f'{local_changes}\\n'
                'Không sửa file trong /content/Wild-Diff-ICMH; hãy gửi phần này để xử lý.'
            )

        before = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'], cwd=REPO, text=True
        ).strip()
        subprocess.run(
            ['git', 'pull', '--ff-only', 'origin', BRANCH], cwd=REPO, check=True
        )
        after = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'], cwd=REPO, text=True
        ).strip()
        remote = subprocess.check_output(
            ['git', 'rev-parse', '--short', f'origin/{BRANCH}'], cwd=REPO, text=True
        ).strip()

        print('Commit trước:', before)
        print('Commit sau:  ', after)
        print('GitHub main: ', remote)
        assert after == remote, f'Code local {after} chưa khớp GitHub {remote}'
        if before == after:
            print('ĐÃ Ở BẢN MỚI NHẤT — chạy lại đúng cell vừa lỗi.')
        else:
            print('CẬP NHẬT THÀNH CÔNG — chạy lại đúng cell vừa lỗi.')
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
            '--force-reinstall', 'src/recognize-anything',
        ], cwd=REPO, check=True)

        import importlib
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

        # A pip subprocess can add files after this Jupyter kernel initialized.
        # Refresh import caches so the freshly installed vendored package is
        # visible immediately, without requiring a runtime restart.
        importlib.invalidate_caches()
        import ram
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
        print('RAM package:', ram.__file__)
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
            # L4 is the project's operator-selected Colab tier. Keep one reviewed
            # setting instead of silently changing the experiment on another GPU.
            ram_batch_size = 2
            ram_num_workers = min(2, os.cpu_count() or 1)
            tag_log_path = DRIVE_ROOT / 'logs' / f'ram_tags_{SITE.replace(":", "_")}.log'
            tag_log_path.parent.mkdir(parents=True, exist_ok=True)
            print(
                f'RAM++: GPU={torch.cuda.get_device_name(0)}, batch={ram_batch_size}, '
                f'workers={ram_num_workers}'
            )
            print('Log chi tiết:', tag_log_path)
            tag_command = [
                sys.executable, '-u', 'tools/precompute_ram_tags.py',
                '--data-root', str(LOCAL_IMAGES),
                '--checkpoint', str(RAM_CKPT),
                '--site-id', SITE,
                '--output', str(TAGS_PATH),
                '--batch-size', str(ram_batch_size),
                '--num-workers', str(ram_num_workers),
            ]
            with tag_log_path.open('a', encoding='utf-8') as log_stream:
                process = subprocess.Popen(
                    tag_command,
                    cwd=REPO,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                assert process.stdout is not None
                for line in process.stdout:
                    print(line, end='')
                    log_stream.write(line)
                    log_stream.flush()
                return_code = process.wait()
            if return_code:
                raise RuntimeError(
                    f'RAM++ tagging dừng với mã {return_code}. '
                    f'Traceback đầy đủ đã lưu tại {tag_log_path}'
                )

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
        # h1_v2 is intentionally fresh: h1/A01 checkpoints created before the
        # entropy migration fix are unsafe to resume.
        RUN_DIR = DRIVE_ROOT / 'runs' / 'h1_v2' / SITE.split(':')[-1]
        import time

        COLAB_CU_PER_HOUR = 1.54  # sửa nếu bảng Tài nguyên hiển thị mức khác
        env = os.environ.copy()
        env.update(
            WILD_DATA_ROOT=str(LOCAL_IMAGES),
            KGA_SITE_ID=SITE,
            KGA_TAGS=str(TAGS_PATH),
            BPP_WEIGHT=str(BPP_WEIGHT),
            WILD_RUN_DIR=str(RUN_DIR),
            PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
            PYTHONUNBUFFERED='1',
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
        train_log_path = DRIVE_ROOT / 'logs' / f"train_smoke_{SITE.replace(':', '_')}.log"
        train_log_path.parent.mkdir(parents=True, exist_ok=True)
        print('Log chi tiết:', train_log_path)

        smoke_started = time.monotonic()
        with train_log_path.open('a', encoding='utf-8') as log_stream:
            process = subprocess.Popen(
                command,
                cwd=REPO,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end='')
                log_stream.write(line)
                log_stream.flush()
            return_code = process.wait()

        if return_code:
            raise RuntimeError(
                f'Smoke test training dừng với mã {return_code}. '
                f'Traceback đầy đủ đã lưu tại {train_log_path}'
            )
        smoke_hours = (time.monotonic() - smoke_started) / 3600
        smoke_cu_estimate = smoke_hours * COLAB_CU_PER_HOUR

        def latest_full_checkpoint(run_dir):
            checkpoint_dir = Path(run_dir) / 'checkpoints'
            last_checkpoint = checkpoint_dir / 'last.ckpt'
            if last_checkpoint.is_file():
                return last_checkpoint
            candidates = list(checkpoint_dir.glob('*.ckpt'))
            if not candidates:
                raise FileNotFoundError(f'Không tìm thấy checkpoint trong {checkpoint_dir}')
            return max(candidates, key=lambda path: path.stat().st_mtime)

        def inspect_full_checkpoint(path):
            checkpoint = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
            global_step = int(checkpoint.get('global_step', -1))
            optimizer_states = checkpoint.get('optimizer_states', [])
            if global_step < 20:
                raise RuntimeError(f'Checkpoint mới chỉ ở global_step={global_step}, cần ít nhất 20')
            if not optimizer_states:
                raise RuntimeError('Checkpoint không có optimizer_states nên chưa thể chứng minh resume đúng')
            print('Checkpoint:', path)
            print('global_step:', global_step)
            print('optimizer_states:', len(optimizer_states))
            print('Kích thước:', f'{path.stat().st_size / 2**30:.2f} GiB')
            return global_step

        PROJECT_CKPT = latest_full_checkpoint(RUN_DIR)
        SMOKE_GLOBAL_STEP = inspect_full_checkpoint(PROJECT_CKPT)
        print(f'Smoke test hoàn tất trong {smoke_hours:.2f} giờ (~{smoke_cu_estimate:.2f} CU).')
        """
    ),
    markdown(
        """
        ## Bước 8 — Xác nhận resume checkpoint

        Cell này đặt đích bằng `global_step hiện tại + 1`. Nếu resume đúng, log phải có dòng
        `Restoring states from ...ckpt` và checkpoint mới phải tăng step trong khi vẫn có
        `optimizer_states`. Muốn kiểm tra đúng tình huống Colab bị ngắt, hãy **khởi động lại
        runtime**, chạy lại Bước 1–7 rồi mới chạy cell này; chạy ngay trong cùng runtime vẫn là
        phép kiểm tra full-state resume hợp lệ. Nếu checkpoint cũ được báo là hỏng hoặc không
        đọc được, chạy lại **Bước 7 đúng một lần** để bỏ qua file hỏng và tạo `last.ckpt` mới
        theo cơ chế ghi an toàn, rồi mới chạy lại Bước 8.
        """
    ),
    code(
        """
        RESUME_TARGET_STEP = SMOKE_GLOBAL_STEP + 1
        resume_command = [
            sys.executable, '-u', 'train.py',
            '--config', 'configs/train_kgalagadi_colab.yaml',
            '--init-checkpoint', str(AUTHOR_CKPT),
            '--resume', str(PROJECT_CKPT),
            f'lightning.trainer.max_steps={RESUME_TARGET_STEP}',
            'lightning.trainer.limit_val_batches=0',
        ]
        resume_log_path = DRIVE_ROOT / 'logs' / f"train_resume_{SITE.replace(':', '_')}.log"
        print('Lệnh resume:', ' '.join(resume_command))
        resume_started = time.monotonic()
        with resume_log_path.open('a', encoding='utf-8') as log_stream:
            process = subprocess.Popen(
                resume_command,
                cwd=REPO,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end='')
                log_stream.write(line)
                log_stream.flush()
            return_code = process.wait()
        if return_code:
            raise RuntimeError(
                f'Resume test dừng với mã {return_code}. Traceback đầy đủ: {resume_log_path}'
            )

        PROJECT_CKPT = latest_full_checkpoint(RUN_DIR)
        resumed_step = inspect_full_checkpoint(PROJECT_CKPT)
        if resumed_step < RESUME_TARGET_STEP:
            raise RuntimeError(
                f'Resume không tăng global step: trước={SMOKE_GLOBAL_STEP}, sau={resumed_step}'
            )
        resume_hours = (time.monotonic() - resume_started) / 3600
        print(
            f'RESUME THÀNH CÔNG: {SMOKE_GLOBAL_STEP} -> {resumed_step}; '
            f'~{resume_hours * COLAB_CU_PER_HOUR:.3f} CU'
        )
        """
    ),
    markdown(
        """
        ## Bước 9 — Decode và đánh giá 2 ảnh test

        Đây là kiểm tra end-to-end, chưa phải số liệu dùng trong báo cáo. Cell chỉ decode 2 ảnh
        bằng 5 bước DDIM để tiết kiệm CU, sau đó ghi metric tổng hợp vào
        `MyDrive/wild_diff_icmh/results/results.jsonl`. Khi chạy thí nghiệm chính phải bỏ
        `--limit 2`, tăng lên 50 DDIM step, giữ center crop 256×256 giống validation
        và đánh giá toàn bộ test split.
        """
    ),
    code(
        """
        SMOKE_DECODE_DIR = Path('/content/results/phase1_h1_smoke')
        PER_IMAGE_RESULTS = DRIVE_ROOT / 'results' / f'phase1_h1_smoke_{SITE.replace(":", "_")}.jsonl'
        RESULTS_REGISTRY = DRIVE_ROOT / 'results' / 'results.jsonl'
        SD_CKPT = CKPT_ROOT / 'sd2p1' / 'v2-1_512-ema-pruned.ckpt'
        decode_steps = 5
        decode_started = time.monotonic()

        decode_command = [
            sys.executable, '-u', 'inference_partition.py',
            '--ckpt_sd', str(SD_CKPT),
            '--ckpt_lc', str(PROJECT_CKPT),
            '--config', str(RUN_DIR / 'config_model.yaml'),
            '--input', str(LOCAL_IMAGES),
            '--output', str(SMOKE_DECODE_DIR),
            '--manifest', 'data/manifests/kgalagadi_site_split.jsonl',
            '--tag-cache', str(TAGS_PATH),
            '--split', 'test', '--site-id', SITE,
            '--sampler', 'ddim', '--steps', str(decode_steps),
            '--device', 'cuda', '--limit', '2', '--crop-size', '256',
            'params.c_cfg_scale=3.0',
        ]
        def run_and_log(command, log_path, label):
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open('a', encoding='utf-8') as log_stream:
                process = subprocess.Popen(
                    command,
                    cwd=REPO,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                assert process.stdout is not None
                for line in process.stdout:
                    print(line, end='')
                    log_stream.write(line)
                    log_stream.flush()
                return_code = process.wait()
            if return_code:
                raise RuntimeError(
                    f'{label} dừng với mã {return_code}. Traceback đầy đủ: {log_path}'
                )

        decode_log_path = DRIVE_ROOT / 'logs' / f"decode_smoke_{SITE.replace(':', '_')}.log"
        print('Decode:', ' '.join(decode_command))
        run_and_log(decode_command, decode_log_path, 'Decode smoke test')

        decode_hours = (time.monotonic() - decode_started) / 3600
        decode_cu_estimate = decode_hours * COLAB_CU_PER_HOUR
        evaluate_command = [
            sys.executable, '-u', 'tools/evaluate_kgalagadi.py',
            '--manifest', 'data/manifests/kgalagadi_site_split.jsonl',
            '--data-root', str(LOCAL_IMAGES),
            '--reconstruction-root', str(SMOKE_DECODE_DIR),
            '--split', 'test', '--site-id', SITE,
            '--method', 'H1', '--limit', '2', '--crop-size', '256', '--lpips',
            '--output', str(PER_IMAGE_RESULTS),
            '--results-registry', str(RESULTS_REGISTRY),
            '--exp-id', f'phase1_h1_smoke_{SITE.replace(":", "_")}',
            '--lambda-rate', str(BPP_WEIGHT),
            '--ddim-steps', str(decode_steps),
            '--cu-estimate', str(decode_cu_estimate),
        ]
        print('Đánh giá:', ' '.join(evaluate_command))
        evaluate_log_path = DRIVE_ROOT / 'logs' / f"evaluate_smoke_{SITE.replace(':', '_')}.log"
        run_and_log(evaluate_command, evaluate_log_path, 'Evaluate smoke test')
        print('END-TO-END THÀNH CÔNG')
        print('Kết quả từng ảnh:', PER_IMAGE_RESULTS)
        print('Registry:', RESULTS_REGISTRY)
        """
    ),
    markdown(
        """
        ## Bước 10 — Khép Phase 1 và dự báo chi phí 2K step

        Chạy **một lần sau khi Bước 7–9 đã thành công trong cùng runtime**. Cell này chỉ
        đọc artifact đã có, audit metadata 100 ảnh và ghi hai báo cáo JSON lên Drive;
        **không train, không decode và không tự chạy 2K step**. Báo cáo sẽ nói rõ có nên
        chạy calibration 2K hay dừng vì dự báo vượt phần ngân sách còn lại.
        """
    ),
    code(
        """
        METADATA_REPORT = DRIVE_ROOT / 'results' / f"phase1_metadata_{SITE.replace(':', '_')}.json"
        PHASE1_CLOSEOUT = DRIVE_ROOT / 'results' / f"phase1_closeout_{SITE.replace(':', '_')}.json"
        EXP_ID = f'phase1_h1_smoke_{SITE.replace(":", "_")}'

        metadata_command = [
            sys.executable, '-u', 'tools/data/audit_metadata.py',
            '--manifest', 'data/manifests/kgalagadi_site_split.jsonl',
            '--site-id', SITE,
            '--sample-size', '100',
            '--data-root', str(LOCAL_IMAGES),
            '--output', str(METADATA_REPORT),
        ]
        print('Audit metadata:', ' '.join(metadata_command))
        subprocess.run(metadata_command, cwd=REPO, check=True)

        git_commit = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'], cwd=REPO, text=True
        ).strip()
        consumed_cu_estimate = (
            smoke_cu_estimate
            + resume_hours * COLAB_CU_PER_HOUR
            + decode_cu_estimate
        )
        closeout_command = [
            sys.executable, '-u', 'tools/phase1_closeout.py',
            '--manifest', 'data/manifests/kgalagadi_site_split.jsonl',
            '--metadata-report', str(METADATA_REPORT),
            '--results-registry', str(RESULTS_REGISTRY),
            '--exp-id', EXP_ID,
            '--checkpoint', str(PROJECT_CKPT),
            '--resume-before', str(SMOKE_GLOBAL_STEP),
            '--resume-after', str(resumed_step),
            '--smoke-steps', '20',
            '--smoke-elapsed-seconds', str(smoke_hours * 3600),
            '--consumed-cu-estimate', str(consumed_cu_estimate),
            '--cu-per-hour', str(COLAB_CU_PER_HOUR),
            '--phase-budget-cu', '8',
            '--calibration-target-steps', '2000',
            '--gpu-name', torch.cuda.get_device_name(0),
            '--git-commit', git_commit,
            '--output', str(PHASE1_CLOSEOUT),
        ]
        print('Closeout:', ' '.join(closeout_command))
        subprocess.run(closeout_command, cwd=REPO, check=True)

        closeout = json.loads(PHASE1_CLOSEOUT.read_text(encoding='utf-8'))
        calibration = closeout['calibration']
        print('PHASE 1 CLOSEOUT THÀNH CÔNG')
        print('Metadata:', METADATA_REPORT)
        print('Closeout:', PHASE1_CLOSEOUT)
        print(
            f"Dự báo 2K: {calibration['projected_elapsed_seconds'] / 3600:.2f} giờ, "
            f"~{calibration['projected_cu_estimate']:.2f} CU"
        )
        print('Quyết định:', calibration['recommendation'])
        """
    ),
    markdown(
        """
        ## Sau khi Phase 1 smoke test thành công

        Sau Bước 10, dùng quyết định trong `phase1_closeout_*.json` để cập nhật ngân sách.
        Không tự chạy 2K nếu báo cáo ghi `do_not_run_2k`. Checkpoint nằm trong
        `MyDrive/wild_diff_icmh/runs/`; runtime mới vẫn chạy lại Bước 1–6 trước, sau đó
        training tự resume checkpoint mới nhất.
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
