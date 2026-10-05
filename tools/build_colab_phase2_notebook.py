"""Generate the Phase 2 Colab notebook (corpus labels, dev set, detections, B0, baselines).

Steps 1-5 are reused verbatim from ``build_colab_training_notebook.py`` (Drive,
code, dependencies, images, checkpoints); the Phase 2 cells follow.  Every
Phase 2 step is resumable, so a new runtime re-runs steps 1-5 and then the
Phase 2 cells skip finished work.
"""
from __future__ import annotations

import json
from pathlib import Path

from build_colab_training_notebook import cells as TRAINING_CELLS, code, markdown

BRANCH_LINE_TRAINING = "BRANCH = 'phase2'  # test branch; set back to 'main' when merging"
BRANCH_LINE_PHASE2 = "BRANCH = 'phase2'  # Phase 2 branch; set back to 'main' when merging"


def _reused(prefixes):
    """Markdown heading cell + its code cell for every heading prefix, in order."""
    picked = []
    for prefix in prefixes:
        for index, cell in enumerate(TRAINING_CELLS):
            source = cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])
            if cell["cell_type"] == "markdown" and source.lstrip("#").strip().startswith(prefix):
                pair = [dict(cell), dict(TRAINING_CELLS[index + 1])]
                pair[1]["source"] = pair[1]["source"].replace(BRANCH_LINE_TRAINING, BRANCH_LINE_PHASE2)
                picked.extend(pair)
                break
        else:
            raise KeyError(f"training notebook has no cell headed {prefix!r}")
    return picked


cells = [
    markdown(
        """
        # Wild-Diff-ICMH — Phase 2 trên Google Colab

        Notebook này dựng hạ tầng đánh giá và các hàng đối chứng, **không train**. Kết quả nằm trong
        `MyDrive/wild_diff_icmh/phase2/` và `results/results.jsonl`.

        | Bước | Việc | GPU? |
        |---|---|---|
        | 1–5 | Drive, code (nhánh `phase2`), cài đặt, chép ảnh, checkpoint (giống notebook training) | có runtime GPU |
        | P2-0 | Checkpoint tác giả λ = 2/8/32 (tải từ Hugging Face về Drive một lần) | không |
        | P2-1 | Nhãn ngày/đêm theo nguồn sáng cho 10.222 ảnh | không |
        | P2-2 | Đóng băng tập dev 300 ảnh | không |
        | P2-3 | MegaDetector trên ảnh gốc (env riêng) | có |
        | P2-4 | Thống kê miền | không |
        | P2-5 | RAM++ tags cho cả 20 site | có |
        | P2-6 | Đo thời gian decode B0 ở 1024 và 512 → dự báo CU | có |
        | P2-7 | **Decode B0 trên tập dev** — chỉ chạy khi đặt `RUN_B0 = True` | có, tốn nhất |
        | P2-8 | Baseline JPEG/WebP | không |
        | P2-9 | Baseline CompressAI | có, nhẹ |
        | P2-10 | Chấm điểm mọi kho lưu trữ (ảnh + MegaDetector) | có |
        | P2-11 | Đồ thị RD chung | không |
        | P2-12 | Ghi CU đo thật của phiên | không |

        Mọi bước đều **chạy tiếp được**: runtime mới thì chạy lại 1–5 rồi chạy lại các bước P2; phần đã xong
        tự bỏ qua. Ngân sách Phase 2: ≤ 8 CU — xem dự báo ở P2-6 trước khi bật P2-7.
        """
    ),
    *_reused(["Bước 1 ", "Bước 2 ", "Bước 2A", "Bước 3 ", "Bước 4 ", "Bước 5 "]),
    markdown(
        """
        ## P2-0 — Đường dẫn chung và checkpoint B0 (λ = 2/8/32)

        Checkpoint λ = 8 và 32 chưa có trên Drive thì được tải từ `RuoyuFeng/Diff-ICMH` về Drive một
        lần (vài GB), rồi chép sang ổ cục bộ.
        """
    ),
    code(
        """
        import json
        import shutil
        from concurrent.futures import ThreadPoolExecutor
        from huggingface_hub import hf_hub_download

        P2 = DRIVE_ROOT / 'phase2'
        P2.mkdir(parents=True, exist_ok=True)
        (DRIVE_ROOT / 'logs').mkdir(parents=True, exist_ok=True)
        MANIFEST = 'data/manifests/kgalagadi_site_split.jsonl'
        ILLUMINATION = P2 / 'kgalagadi_illumination.jsonl'
        DEV_LIST = P2 / 'kgalagadi_dev.txt'
        DETECTIONS = P2 / 'detections' / 'originals.jsonl'
        TAGS_ALL = DRIVE_ROOT / 'tags' / 'KGA_all.jsonl'
        ARCHIVE = P2 / 'archive'
        EVAL_DIR = P2 / 'eval'
        RESULTS_REGISTRY = DRIVE_ROOT / 'results' / 'results.jsonl'
        SD_CKPT = CKPT_ROOT / 'sd2p1' / 'v2-1_512-ema-pruned.ckpt'
        LAMBDAS = [2, 8, 32]
        DDIM_STEPS = 50
        B0_OVERRIDES = [
            'params.control_stage_config.params.control_model_ratio=1.0',
            'params.c_cfg_scale=3.0',
        ]
        for folder in (ARCHIVE, EVAL_DIR, DETECTIONS.parent):
            folder.mkdir(parents=True, exist_ok=True)


        def run_logged(command, log_name, echo=True):
            log_path = DRIVE_ROOT / 'logs' / log_name
            command = [str(part) for part in command]
            print('$', ' '.join(command))
            with log_path.open('a', encoding='utf-8') as log_stream:
                process = subprocess.Popen(
                    command, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1,
                )
                for line in process.stdout:
                    if echo:
                        print(line, end='')
                    log_stream.write(line)
                return_code = process.wait()
            if return_code:
                raise RuntimeError(f'{log_name}: dừng với mã {return_code}; log đầy đủ: {log_path}')


        def author_checkpoint(lam):
            folder = f'CNscale1.0_1_1_{lam}_2_WTagGCM_bs16x1_lr0.00005_cfg7.0'
            relative = Path('difficmh_models') / folder / 'model.ckpt'
            drive_copy = DRIVE_CKPT_ROOT / relative
            if not drive_copy.is_file():
                print(f'Tải checkpoint tác giả λ={lam} từ Hugging Face về Drive...')
                hf_hub_download('RuoyuFeng/Diff-ICMH', relative.as_posix(), local_dir=DRIVE_CKPT_ROOT)
            local = CKPT_ROOT / relative
            if not local.is_file() or local.stat().st_size != drive_copy.stat().st_size:
                local.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(drive_copy, local)
            return local


        B0_CKPT = {lam: author_checkpoint(lam) for lam in LAMBDAS}
        for lam, path in B0_CKPT.items():
            print(f'B0 λ={lam}: {path} ({path.stat().st_size / 2**30:.2f} GiB)')
        """
    ),
    markdown(
        """
        ### P2-0b — Kiểm tra đủ 10.222 ảnh (tải bù nếu thiếu)

        Phase 1 chỉ cần site A01. Nếu Drive chưa có đủ ảnh Kgalagadi, cell tải phần thiếu từ LILA thẳng
        vào Drive (chạy tiếp được), rồi chép sang ổ cục bộ. Tải ~9 GB mất khoảng 15–30 phút; nếu thiếu
        nhiều, nên chạy riêng cell này trên runtime **CPU** (rẻ hơn GPU nhiều) rồi mới đổi sang L4.
        """
    ),
    code(
        """
        manifest_rows = [
            json.loads(line) for line in (REPO / MANIFEST).read_text(encoding='utf-8').splitlines() if line.strip()
        ]
        missing = [row for row in manifest_rows if not (LOCAL_IMAGES / row['relative_path']).is_file()]
        print(f'Ảnh cục bộ: {len(manifest_rows) - len(missing)}/{len(manifest_rows)}')
        if missing:
            run_logged([
                sys.executable, '-u', 'tools/data/download_images.py', '--manifest', MANIFEST,
                '--root', DRIVE_ROOT / 'images', '--workers', 16,
            ], 'p2_download_images.log')
            for row in missing:
                source = DRIVE_ROOT / 'images' / row['relative_path']
                target = LOCAL_IMAGES / row['relative_path']
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
            still_missing = [row for row in manifest_rows if not (LOCAL_IMAGES / row['relative_path']).is_file()]
            assert not still_missing, f'Vẫn thiếu {len(still_missing)} ảnh, xem logs/p2_download_images.log'
        print('Đủ ảnh cho Phase 2.')
        """
    ),
    markdown(
        """
        ## P2-1 — Nhãn ngày/đêm theo nguồn sáng (CPU, vài phút)

        Đêm = ảnh do camera tự chiếu sáng (flash hoặc IR), xác định từ chính file ảnh:
        EXIF Flash → ảnh xám IR → độ sáng pixel. Không dùng giờ chụp. Kết quả in ra số ảnh theo
        từng nguồn tín hiệu và số ảnh mà nhãn theo giờ cũ bị sai.
        """
    ),
    code(
        """
        run_logged([
            sys.executable, '-u', 'tools/data/label_illumination.py',
            '--manifest', MANIFEST, '--data-root', LOCAL_IMAGES, '--output', ILLUMINATION,
        ], 'p2_illumination.log')
        """
    ),
    markdown(
        """
        ## P2-2 — Đóng băng tập dev (CPU)

        Lần đầu: tạo `phase2/kgalagadi_dev.txt` từ validation, phân tầng theo nhãn ngày/đêm mới.
        Các lần sau: tạo lại vào file tạm và **chỉ kiểm tra** là trùng khớp — không bao giờ ghi đè.
        """
    ),
    code(
        """
        def build_dev(output):
            run_logged([
                sys.executable, '-u', 'tools/data/build_dev_set.py', '--manifest', MANIFEST,
                '--illumination-sidecar', ILLUMINATION, '--output', output,
            ], 'p2_dev_set.log')

        def dev_ids(path):
            return [line for line in Path(path).read_text().splitlines() if line and not line.startswith('#')]

        if DEV_LIST.is_file():
            check = Path('/content/kgalagadi_dev_check.txt')
            build_dev(check)
            assert dev_ids(check) == dev_ids(DEV_LIST), 'Tập dev tạo lại khác bản đã đóng băng!'
            print('Tập dev đã đóng băng và tái lập đúng:', DEV_LIST)
        else:
            build_dev(DEV_LIST)
        print(len(dev_ids(DEV_LIST)), 'ảnh dev')
        """
    ),
    markdown(
        """
        ## P2-3 — MegaDetector trên 10.222 ảnh gốc (GPU, env riêng)

        MegaDetector V6 cài trong venv riêng `/content/envs/detect` (thư viện ultralytics của nó xung đột
        với môi trường codec). Kết quả là pseudo ground truth: một dòng cho **mọi** ảnh, kể cả ảnh không có
        con vật. Chạy tiếp được nếu runtime ngắt.
        """
    ),
    code(
        """
        DETECT_PY = Path('/content/envs/detect/bin/python')
        if not DETECT_PY.exists():
            subprocess.run([sys.executable, '-m', 'venv', '--system-site-packages', '/content/envs/detect'], check=True)
            subprocess.run([str(DETECT_PY), '-m', 'pip', 'install', '-q', 'PytorchWildlife'], check=True)

        run_logged([
            DETECT_PY, '-u', 'tools/detect/run_megadetector.py', '--manifest', MANIFEST,
            '--image-root', LOCAL_IMAGES, '--output', DETECTIONS,
        ], 'p2_megadetector_originals.log')
        """
    ),
    markdown("## P2-4 — Thống kê miền (CPU)"),
    code(
        """
        run_logged([
            sys.executable, '-u', 'tools/data/domain_stats.py', '--manifest', MANIFEST,
            '--illumination-sidecar', ILLUMINATION, '--detections', DETECTIONS,
            '--output', P2 / 'domain_stats.json',
        ], 'p2_domain_stats.log')
        """
    ),
    markdown(
        """
        ## P2-5 — RAM++ tags cho cả 20 site (GPU)

        Một file cho cả corpus (`tags/KGA_all.jsonl`), dùng cho decode B0 và cho H1 chung ở Phase 3.
        """
    ),
    code(
        """
        manifest_ids = {
            json.loads(line)['image_id']
            for line in (REPO / MANIFEST).read_text(encoding='utf-8').splitlines() if line.strip()
        }
        tagged = set()
        if TAGS_ALL.is_file():
            for line in TAGS_ALL.read_text(encoding='utf-8').splitlines():
                try:
                    tagged.add(json.loads(line)['image_id'])
                except (json.JSONDecodeError, KeyError):
                    pass
        if manifest_ids <= tagged:
            print(f'Tags đã đủ {len(manifest_ids)} ảnh — bỏ qua.')
        else:
            print(f'Còn {len(manifest_ids - tagged)} ảnh chưa có tag.')
            run_logged([
                sys.executable, '-u', 'tools/precompute_ram_tags.py', '--data-root', LOCAL_IMAGES,
                '--checkpoint', RAM_CKPT, '--output', TAGS_ALL,
                '--batch-size', '2', '--num-workers', str(min(2, os.cpu_count() or 1)),
            ], 'p2_ram_tags_all.log')
        """
    ),
    markdown(
        """
        ## P2-6 — Đo thời gian decode B0 và dự báo chi phí P2-7

        Decode 3 ảnh dev bằng B0 λ=2 ở cạnh dài 1024 và 512, đọc `decode_log.jsonl`, rồi dự báo giờ GPU
        và CU cho P2-7. Chọn `B0_SIDE`, `B0_LIMIT` ở P2-7 theo bảng in ra. Kết quả đo lưu vào
        `phase2/probe_timing.json`; các phiên sau chỉ in lại, không decode thử nữa.
        """
    ),
    code(
        """
        COLAB_CU_PER_HOUR = 1.54

        def b0_command(lam, side, output, limit=None):
            command = [
                sys.executable, '-u', 'inference_partition.py',
                '--ckpt_sd', SD_CKPT, '--ckpt_lc', B0_CKPT[lam],
                '--config', 'configs/model/diffeic.yaml',
                '--input', LOCAL_IMAGES, '--output', output,
                '--manifest', MANIFEST, '--split', 'val', '--dev-list', DEV_LIST,
                '--tag-cache', TAGS_ALL, '--sampler', 'ddim', '--steps', DDIM_STEPS,
                '--device', 'cuda', '--processing-long-side', side, '--skip-existing',
            ]
            if limit:
                command += ['--limit', limit]
            return command + B0_OVERRIDES

        n_dev = len(dev_ids(DEV_LIST))
        PROBE_FILE = P2 / 'probe_timing.json'
        probe_seconds = json.loads(PROBE_FILE.read_text()) if PROBE_FILE.is_file() else {}
        for side in (1024, 512):
            if str(side) not in probe_seconds:
                probe = Path('/content/p2_probe') / f'ls{side}'
                run_logged(b0_command(2, side, probe, limit=3), f'p2_probe_ls{side}.log', echo=False)
                records = [json.loads(line) for line in (probe / 'decode_log.jsonl').read_text().splitlines()]
                timed = records[1:] or records  # the first image includes CUDA warm-up
                probe_seconds[str(side)] = sum(r['encode_seconds'] + r['decode_seconds'] for r in timed) / len(timed)
                PROBE_FILE.write_text(json.dumps(probe_seconds, indent=2))
            seconds = probe_seconds[str(side)]
            hours = seconds * n_dev * len(LAMBDAS) / 3600
            print(f'cạnh dài {side}: {seconds:.1f} s/ảnh '
                  f'→ {n_dev} ảnh × {len(LAMBDAS)} λ ≈ {hours:.2f} giờ ≈ {hours * COLAB_CU_PER_HOUR:.2f} CU')
        """
    ),
    markdown(
        """
        ## P2-7 — Decode B0 trên tập dev (tốn GPU nhất)

        Đọc dự báo ở P2-6, chỉnh `B0_SIDE` / `B0_LIMIT` cho vừa ngân sách, rồi đặt `RUN_B0 = True`.
        Kho lưu trữ: `phase2/archive/B0_ls<side>_ddim<steps>/lambda_<λ>/`. Ngắt giữa chừng thì chạy lại
        cell — ảnh đã decode được bỏ qua (`--skip-existing`), mỗi ảnh có seed riêng.
        """
    ),
    code(
        """
        B0_SIDE = 1024
        B0_LIMIT = None      # None = cả tập dev; ví dụ 100 nếu dự báo vượt ngân sách
        RUN_B0 = False       # đặt True sau khi xem dự báo ở P2-6

        if not RUN_B0:
            print('P2-7 chưa chạy: đặt RUN_B0 = True sau khi xem dự báo ở P2-6.')
        else:
            for lam in LAMBDAS:
                output = ARCHIVE / f'B0_ls{B0_SIDE}_ddim{DDIM_STEPS}' / f'lambda_{lam}'
                run_logged(b0_command(lam, B0_SIDE, output, limit=B0_LIMIT), f'p2_b0_lambda{lam}.log', echo=False)
                print('Xong B0 λ =', lam, '→', output)
        """
    ),
    markdown(
        """
        ## P2-8 — Baseline JPEG / WebP (CPU, chạy song song 4 tiến trình)

        JPEG ở độ phân giải gốc và ở cạnh dài 1024; WebP ở 1024. Cùng giao thức: ảnh gốc vào, ảnh gốc ra.
        """
    ),
    code(
        """
        CLASSICAL = (
            [('jpeg', q, None) for q in (5, 15, 40)]
            + [('jpeg', q, 1024) for q in (5, 15, 40)]
            + [('webp', q, 1024) for q in (5, 15, 40)]
        )

        def run_classical(job):
            codec, quality, side = job
            curve = f"{codec}_{'full' if side is None else f'ls{side}'}"
            command = [
                sys.executable, '-u', 'tools/baselines/run_classical.py', '--manifest', MANIFEST,
                '--data-root', LOCAL_IMAGES, '--split', 'val', '--dev-list', DEV_LIST,
                '--codec', codec, '--quality', quality, '--output', ARCHIVE / curve / f'q{quality}',
                '--skip-existing',
            ]
            if side:
                command += ['--processing-long-side', side]
            run_logged(command, f'p2_{curve}_q{quality}.log', echo=False)
            return curve, quality

        with ThreadPoolExecutor(max_workers=4) as pool:
            for curve, quality in pool.map(run_classical, CLASSICAL):
                print('Xong', curve, 'q', quality)
        """
    ),
    markdown(
        """
        ## P2-9 — Baseline CompressAI (GPU, nhẹ)

        `bmshj2018-hyperprior` (họ codec của bài Xie 2025), `mbt2018`, `cheng2020-attn`, chất lượng 1–3,
        ở cạnh dài 1024; entropy coding thật, bitstream ghi ra file.
        """
    ),
    code(
        """
        for model in ('bmshj2018-hyperprior', 'mbt2018', 'cheng2020-attn'):
            for quality in (1, 2, 3):
                curve = f'compressai-{model}_ls1024'
                run_logged([
                    sys.executable, '-u', 'tools/baselines/run_compressai_zoo.py', '--manifest', MANIFEST,
                    '--data-root', LOCAL_IMAGES, '--split', 'val', '--dev-list', DEV_LIST,
                    '--model', model, '--quality', quality, '--processing-long-side', 1024,
                    '--output', ARCHIVE / curve / f'q{quality}', '--skip-existing',
                ], f'p2_{curve}_q{quality}.log', echo=False)
                print('Xong', curve, 'q', quality)
        """
    ),
    markdown(
        """
        ## P2-10 — Chấm điểm mọi kho lưu trữ

        Với mỗi `phase2/archive/<đường>/<điểm>/`: chỉ số ảnh (PSNR, SSIM ×2, MS-SSIM, LPIPS, DISTS, bpp,
        compression ratio, thời gian) + MegaDetector trên ảnh tái tạo → mAP, ảnh rỗng báo nhầm, ảo giác,
        mất con vật. Tất cả ghi vào `results/results.jsonl` với `exp_id = p2dev_<đường>__<điểm>`.
        Chỉ chấm đúng các ảnh đã có trong kho (`decode_log.jsonl`); kho đã chấm thì bỏ qua, trừ khi kho
        có thêm ảnh từ sau lần chấm trước (ví dụ P2-7 chạy tiếp sau khi bị ngắt) — khi đó chấm lại.
        """
    ),
    code(
        """
        try:
            import pycocotools  # noqa: F401
        except ImportError:
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'pycocotools'], check=True)

        for run_info in sorted(ARCHIVE.glob('*/*/run_info.json')):
            root = run_info.parent
            curve, point = root.parent.name, root.name
            name = f'{curve}__{point}'
            image_eval = EVAL_DIR / f'{name}.jsonl'
            machine_eval = EVAL_DIR / f'{name}.machine.json'
            archived = sum(1 for line in (root / 'decode_log.jsonl').read_text().splitlines() if line.strip())
            done_marker = EVAL_DIR / f'{name}.done.json'
            if done_marker.is_file() and json.loads(done_marker.read_text())['images'] >= archived:
                print('Đã chấm:', name, f'({archived} ảnh)')
                continue
            info = json.loads(run_info.read_text())
            common = ['--manifest', MANIFEST, '--split', 'val', '--dev-list', DEV_LIST,
                      '--illumination-sidecar', ILLUMINATION]
            run_logged([
                sys.executable, '-u', 'tools/evaluate_kgalagadi.py', *common,
                '--data-root', LOCAL_IMAGES, '--reconstruction-root', root,
                '--detections', DETECTIONS, '--method', curve, '--lpips', '--dists', '--archived-only',
                '--output', image_eval, '--results-registry', RESULTS_REGISTRY,
                '--exp-id', f'p2dev_{name}', '--lambda-rate', point,
                '--ddim-steps', info.get('steps', 0),
            ], f'p2_eval_{name}.log', echo=False)
            predictions = DETECTIONS.parent / f'{name}.jsonl'
            run_logged([
                DETECT_PY, '-u', 'tools/detect/run_megadetector.py', '--manifest', MANIFEST,
                '--split', 'val', '--dev-list', DEV_LIST, '--image-root', root, '--suffix', '.png', '--archived-only',
                '--output', predictions,
            ], f'p2_megadetector_{name}.log', echo=False)
            run_logged([
                sys.executable, '-u', 'tools/eval_machine.py', *common,
                '--gt', DETECTIONS, '--pred', predictions, '--output', machine_eval, '--archived-only',
                '--results-registry', RESULTS_REGISTRY, '--exp-id', f'p2dev_{name}',
                '--method', curve, '--lambda-rate', point, '--ddim-steps', info.get('steps', 0),
            ], f'p2_machine_{name}.log', echo=False)
            done_marker.write_text(json.dumps({'images': archived}))
            print('Đã chấm:', name, f'({archived} ảnh)')
        """
    ),
    markdown(
        """
        ## P2-11 — Đồ thị RD chung (tiêu chí 7 của Phase 2)

        Mỗi đường là một phương pháp; trục x là bpp tính trên pixel ảnh gốc (log). Dùng để thấy dải bitrate
        chồng lấn giữa B0 và các baseline trước khi chọn λ cho H1.
        """
    ),
    code(
        """
        import matplotlib.pyplot as plt

        points = {}
        for line in RESULTS_REGISTRY.read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            if not str(row['exp_id']).startswith('p2dev_'):
                continue
            if row['illumination'] != 'all' or row.get('subset', 'all') != 'all':
                continue
            points.setdefault(row['exp_id'], {})[row['metric']] = row['value']

        curves = {}
        for exp_id, values in points.items():
            curve = exp_id[len('p2dev_'):].split('__')[0]
            curves.setdefault(curve, []).append(values)

        panels = ['psnr', 'ms_ssim', 'lpips', 'dists', 'map', 'hallucination_rate']
        figure, axes = plt.subplots(2, 3, figsize=(16, 9))
        for axis, metric in zip(axes.flat, panels):
            for curve, values in sorted(curves.items()):
                series = sorted((v['bpp'], v[metric]) for v in values if 'bpp' in v and v.get(metric) is not None)
                if series:
                    axis.plot(*zip(*series), marker='o', label=curve)
            axis.set_xscale('log')
            axis.set_xlabel('bpp (pixel ảnh gốc)')
            axis.set_title(metric)
            axis.grid(alpha=0.3)
        axes.flat[0].legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(P2 / 'rd_dev.png', dpi=120)
        plt.show()
        print('Đồ thị:', P2 / 'rd_dev.png')
        """
    ),
    markdown(
        """
        ## P2-12 — Ghi CU đo thật của phiên

        Mở **Runtime → View resources**, chép số "Available" vào `CU_AVAILABLE_NOW`, chạy cell, rồi ngắt runtime.
        """
    ),
    code(
        """
        from datetime import datetime, timezone

        CU_AVAILABLE_NOW = None  # số "Available" ngay lúc này

        if CU_AVAILABLE_AT_START is None or CU_AVAILABLE_NOW is None:
            raise ValueError('Điền CU_AVAILABLE_AT_START (Bước 1) và CU_AVAILABLE_NOW rồi chạy lại cell.')
        session = {
            'date': datetime.now(timezone.utc).isoformat(),
            'gpu': torch.cuda.get_device_name(0),
            'cu_start': float(CU_AVAILABLE_AT_START),
            'cu_end': float(CU_AVAILABLE_NOW),
            'cu_consumed': float(CU_AVAILABLE_AT_START) - float(CU_AVAILABLE_NOW),
            'git_commit': subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=REPO, text=True).strip(),
        }
        with (P2 / 'sessions.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(session) + chr(10))
        print(session)
        print('Ngắt runtime ngay (Runtime → Disconnect and delete runtime) để ngừng tiêu CU.')
        """
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"name": "Wild_Diff_ICMH_Phase2_Eval.ipynb", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

if __name__ == "__main__":
    output = Path(__file__).resolve().parents[1] / "Wild_Diff_ICMH_Phase2_Eval.ipynb"
    output.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(output)
