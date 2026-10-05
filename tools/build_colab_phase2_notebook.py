"""Generate the two Phase 2 Colab notebooks.

* ``Wild_Diff_ICMH_Phase2_Prepare.ipynb`` -- light-source day/night labels,
  frozen dev set, MegaDetector on the originals, domain statistics, RAM++ tags
  and the B0 decode (needs all images and the SD/B0 checkpoints).  Done once.
* ``Wild_Diff_ICMH_Phase2_Eval.ipynb`` -- one top-to-bottom workflow on the
  300 dev images: baselines, scoring of every archive, RD plot, summary.

Steps 1-5 come verbatim from ``build_colab_training_notebook.py``.  Every step
is resumable, so a new runtime re-runs the notebook from step 1.
"""
from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

from build_colab_training_notebook import cells as TRAINING_CELLS, code, markdown

BRANCH_LINE_TRAINING = "BRANCH = 'phase2'  # test branch; set back to 'main' when merging"
BRANCH_LINE_PHASE2 = "BRANCH = 'phase2'  # Phase 2 branch; set back to 'main' when merging"



# Code shared by several cells (inserted where a cell holds the placeholder).
SNIPPETS = {
    "__RUN_LOGGED__": dedent("""
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
    """).strip(),
    "__DETECT_ENV__": dedent("""
        # Colab's Python has no ensurepip, so ``python -m venv`` cannot install pip;
        # virtualenv ships its own.  --system-site-packages reuses Colab's torch.
        DETECT_ENV = Path('/content/envs/detect')
        DETECT_PY = DETECT_ENV / 'bin' / 'python'
        DETECT_READY = DETECT_ENV / '.ready'
        if not DETECT_READY.exists():
            shutil.rmtree(DETECT_ENV, ignore_errors=True)  # leftovers of a failed attempt
            run_logged([sys.executable, '-m', 'pip', 'install', '-q', 'virtualenv'], 'p2_detect_env.log')
            run_logged([sys.executable, '-m', 'virtualenv', '--system-site-packages', DETECT_ENV], 'p2_detect_env.log')
            # The env also sees Colab's transformers, which needs Colab's
            # huggingface-hub (<1.0); PytorchWildlife's deps would otherwise pull 1.x.
            import importlib.metadata
            hub_pin = f"huggingface-hub=={importlib.metadata.version('huggingface_hub')}"
            run_logged([DETECT_PY, '-m', 'pip', 'install', '-q', 'PytorchWildlife', hub_pin], 'p2_detect_env.log')
            run_logged([DETECT_PY, '-c', 'from PytorchWildlife.models import detection; print("PytorchWildlife OK")'],
                       'p2_detect_env.log')
            DETECT_READY.touch()
    """).strip(),
}

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


PREPARE_CELLS = [
    markdown(
        """
        # Wild-Diff-ICMH — Phase 2 (1/2): chuẩn bị dữ liệu và decode B0

        **Đã chạy xong ngày 05–06/10/2026 — không cần chạy lại.** Giữ lại để tái lập: nhãn ngày/đêm, tập dev,
        MegaDetector trên ảnh gốc, thống kê miền, RAM++ tags, decode B0. Phần còn lại của Phase 2 (baseline,
        chấm điểm, đồ thị) nằm ở `Wild_Diff_ICMH_Phase2_Eval.ipynb`.

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
        ARCHIVE = P2 / 'archive'                        # B0 (expensive to redo): full archive on Drive
        BASELINE_ROOT = Path('/content/p2_baselines')   # JPEG/WebP/CompressAI: local SSD only
        BITSTREAMS = P2 / 'bitstreams'                  # one tar of baseline bitstreams per point
        EVAL_DIR = P2 / 'eval'
        RESULTS_REGISTRY = DRIVE_ROOT / 'results' / 'results.jsonl'
        SD_CKPT = CKPT_ROOT / 'sd2p1' / 'v2-1_512-ema-pruned.ckpt'
        LAMBDAS = [2, 8, 32]
        DDIM_STEPS = 50
        B0_OVERRIDES = [
            'params.control_stage_config.params.control_model_ratio=1.0',
            'params.c_cfg_scale=3.0',
        ]
        for folder in (ARCHIVE, EVAL_DIR, DETECTIONS.parent, BITSTREAMS, BASELINE_ROOT):
            folder.mkdir(parents=True, exist_ok=True)


        __RUN_LOGGED__


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
        __DETECT_ENV__

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

        def b0_command(lam, side, output, limit=None, dev_list=None):
            command = [
                sys.executable, '-u', 'inference_partition.py',
                '--ckpt_sd', SD_CKPT, '--ckpt_lc', B0_CKPT[lam],
                '--config', 'configs/model/diffeic.yaml',
                '--input', LOCAL_IMAGES, '--output', output,
                '--manifest', MANIFEST, '--split', 'val', '--dev-list', dev_list or DEV_LIST,
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

        Đọc dự báo ở P2-6, chỉnh `B0_RUNS` cho vừa ngân sách, rồi đặt `RUN_B0 = True`. Mỗi dòng của
        `B0_RUNS` là (cạnh dài, danh sách λ, số ảnh). Số ảnh < cả tập dev thì lấy **tập con ngẫu nhiên có
        seed cố định** của tập dev (trải đều site, các tập con lồng nhau: 30 ảnh nằm trong 100 ảnh), không
        lấy N ảnh đầu theo tên. Mặc định: 512 trên 100 ảnh × 3 λ, và 1024 trên 30 ảnh với λ=2 để so chất
        lượng 512/1024 trên cùng ảnh. Kho: `phase2/archive/B0_ls<side>_ddim<steps>/lambda_<λ>/`.
        Ngắt giữa chừng thì chạy lại cell — ảnh đã decode được bỏ qua, mỗi ảnh có seed riêng.
        """
    ),
    code(
        """
        import hashlib

        B0_RUNS = [
            (512, LAMBDAS, 100),   # (cạnh dài, các λ, số ảnh dev)
            (1024, [2], 30),
        ]
        RUN_B0 = False       # đặt True sau khi xem dự báo bên dưới

        def dev_subset(size):
            ids = dev_ids(DEV_LIST)
            if size >= len(ids):
                return DEV_LIST
            # Seeded hash order: a reproducible random sample; smaller subsets nest in larger ones.
            chosen = sorted(ids, key=lambda i: hashlib.sha256(f'20261005:{i}'.encode()).hexdigest())[:size]
            path = P2 / f'kgalagadi_dev_sub{size}.txt'
            path.write_text('# random subset of kgalagadi_dev.txt, seed 20261005' + chr(10)
                            + chr(10).join(sorted(chosen)) + chr(10))
            return path

        probe_seconds = json.loads((P2 / 'probe_timing.json').read_text())
        total_hours = 0.0
        for side, lambdas, size in B0_RUNS:
            hours = probe_seconds[str(side)] * size * len(lambdas) / 3600
            total_hours += hours
            print(f'B0 cạnh {side}: {size} ảnh × λ {lambdas} ≈ {hours:.2f} giờ')
        print(f'Tổng ≈ {total_hours:.2f} giờ ≈ {total_hours * COLAB_CU_PER_HOUR:.2f} CU (chưa trừ phần đã decode)')

        if not RUN_B0:
            print('P2-7 chưa chạy: đặt RUN_B0 = True nếu dự báo trên vừa ngân sách.')
        else:
            for side, lambdas, size in B0_RUNS:
                subset = dev_subset(size)
                for lam in lambdas:
                    output = ARCHIVE / f'B0_ls{side}_ddim{DDIM_STEPS}' / f'lambda_{lam}'
                    run_logged(b0_command(lam, side, output, dev_list=subset),
                               f'p2_b0_ls{side}_lambda{lam}.log', echo=False)
                    print('Xong B0 cạnh', side, 'λ =', lam, '→', output)
        """
    ),
]

EVAL_CELLS = [
    markdown(
        """
        # Wild-Diff-ICMH — Phase 2 (2/2): baseline, chấm điểm, đồ thị RD

        **Chạy lần lượt từ trên xuống, không bỏ cell nào.** Notebook `Wild_Diff_ICMH_Phase2_Prepare.ipynb`
        (nhãn ngày/đêm, tập dev, MegaDetector trên ảnh gốc, decode B0) đã chạy xong; ở đây chỉ cần 300 ảnh dev.

        | Bước | Việc | Thời gian |
        |---|---|---|
        | 1–3 | Drive (+ điền CU), code nhánh `phase2`, cài đặt | ~10 phút |
        | 4 | Chép 300 ảnh dev, kiểm tra kho B0, dựng MegaDetector | ~5 phút |
        | 5 | Thống kê miền | vài giây |
        | 6 | Baseline JPEG/WebP (CPU) | ~10 phút |
        | 7 | Baseline CompressAI (GPU) | ~10 phút |
        | 8 | Chấm điểm mọi kho: chỉ số ảnh + MegaDetector | ~1,5–2 giờ |
        | 9 | Đồ thị RD | vài giây |
        | 10 | Tóm tắt để gửi lại | vài giây |
        | 11 | Ghi CU đo thật, rồi ngắt runtime | — |

        Runtime bị ngắt giữa chừng: mở lại notebook, chạy lại **từ Bước 1** — phần đã xong tự bỏ qua
        (điểm đã chấm không nén/chấm lại). Kết quả: `MyDrive/wild_diff_icmh/phase2/` và `results/results.jsonl`.
        """
    ),
    *_reused(["Bước 1 ", "Bước 2 ", "Bước 3 "]),
    markdown(
        """
        ## Bước 4 — Chuẩn bị (ảnh dev, đường dẫn, kiểm tra B0, MegaDetector)

        Chỉ chép **300 ảnh dev** (~250 MB) từ Drive — không cần 10.222 ảnh hay checkpoint. Kiểm tra các kết quả của
        notebook Prepare (nhãn ngày/đêm, tập dev, MegaDetector trên ảnh gốc, kho B0) và in số ảnh B0 đã decode.
        Dựng môi trường MegaDetector riêng cho Bước 8.
        """
    ),
    code(
        """
        import json
        import shutil
        from concurrent.futures import ThreadPoolExecutor

        import torch

        DRIVE_ROOT = Path('/content/drive/MyDrive/wild_diff_icmh')
        LOCAL_IMAGES = Path('/content/data/wild_diff_icmh/images')
        P2 = DRIVE_ROOT / 'phase2'
        MANIFEST = 'data/manifests/kgalagadi_site_split.jsonl'
        ILLUMINATION = P2 / 'kgalagadi_illumination.jsonl'
        DEV_LIST = P2 / 'kgalagadi_dev.txt'
        DETECTIONS = P2 / 'detections' / 'originals.jsonl'
        ARCHIVE = P2 / 'archive'
        BASELINE_ROOT = Path('/content/p2_baselines')
        BITSTREAMS = P2 / 'bitstreams'
        EVAL_DIR = P2 / 'eval'
        RESULTS_REGISTRY = DRIVE_ROOT / 'results' / 'results.jsonl'
        for folder in (EVAL_DIR, BITSTREAMS, BASELINE_ROOT, DRIVE_ROOT / 'logs'):
            folder.mkdir(parents=True, exist_ok=True)
        for needed, step in ((ILLUMINATION, 'P2-1'), (DEV_LIST, 'P2-2'), (DETECTIONS, 'P2-3')):
            assert needed.is_file(), f'Notebook Prepare chưa xong {step}: chưa có {needed}'

        __RUN_LOGGED__


        def dev_ids(path):
            return [line for line in Path(path).read_text().splitlines() if line and not line.startswith('#')]


        def scored(curve, point):
            return (EVAL_DIR / f'{curve}__{point}.done.json').is_file()


        wanted = set(dev_ids(DEV_LIST))
        rows = [json.loads(line) for line in (REPO / MANIFEST).read_text(encoding='utf-8').splitlines() if line.strip()]
        to_copy = [row['relative_path'] for row in rows
                   if row['image_id'] in wanted and not (LOCAL_IMAGES / row['relative_path']).is_file()]

        def copy_dev_image(relative):
            target = LOCAL_IMAGES / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(DRIVE_ROOT / 'images' / relative, target)

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(copy_dev_image, to_copy))
        print(f'Ảnh dev cục bộ: {len(wanted)} (vừa chép {len(to_copy)})')

        print('Kho B0 (ảnh đã decode / ảnh yêu cầu):')
        b0_runs = sorted(ARCHIVE.glob('B0_*/*/run_info.json'))
        assert b0_runs, 'Chưa có kho B0 nào trong phase2/archive (notebook Prepare, P2-7)'
        for run_info in b0_runs:
            requested = dev_ids(json.loads(run_info.read_text()).get('dev_list') or DEV_LIST)
            logged = {json.loads(line)['image_id'] for line in (run_info.parent / 'decode_log.jsonl').read_text().splitlines() if line.strip()}
            done = len(set(requested) & logged)
            print(f'  {run_info.parent.parent.name}/{run_info.parent.name}: {done}/{len(requested)}'
                  + ('' if done == len(requested) else '  ← CHƯA ĐỦ'))

        __DETECT_ENV__
        print('Sẵn sàng.')
        """
    ),
    markdown("## Bước 5 — Thống kê miền (CPU, vài giây)"),
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
        ## Bước 6 — Baseline JPEG / WebP (CPU, chạy song song 4 tiến trình)

        JPEG ở độ phân giải gốc, ở cạnh dài 1024 và 512; WebP ở 1024 và 512. Cùng giao thức: ảnh gốc vào, ảnh gốc ra.

        Ảnh tái tạo của baseline (~10 MB/ảnh PNG) ghi vào ổ cục bộ `/content/p2_baselines`, **không ghi lên
        Drive** — ghi hàng chục GB lên Drive làm Colab bị chặn ("Google Drive quota exceeded"). Chúng tính lại
        được từ bitstream trong vài giây; sau khi chấm (Bước 8), chỉ bitstream được gói thành một file `.tar`
        lên Drive. Điểm nào đã chấm thì bỏ qua cả bước nén.
        """
    ),
    code(
        """
        CLASSICAL = (
            [('jpeg', q, None) for q in (5, 15, 40)]
            + [('jpeg', q, side) for q in (5, 15, 40) for side in (1024, 512)]
            + [('webp', q, side) for q in (5, 15, 40) for side in (1024, 512)]
        )

        def run_classical(job):
            codec, quality, side = job
            curve = f"{codec}_{'full' if side is None else f'ls{side}'}"
            if scored(curve, f'q{quality}'):
                return curve, quality
            command = [
                sys.executable, '-u', 'tools/baselines/run_classical.py', '--manifest', MANIFEST,
                '--data-root', LOCAL_IMAGES, '--split', 'val', '--dev-list', DEV_LIST,
                '--codec', codec, '--quality', quality, '--output', BASELINE_ROOT / curve / f'q{quality}',
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
        ## Bước 7 — Baseline CompressAI (GPU, nhẹ)

        `bmshj2018-hyperprior` (họ codec của bài Xie 2025), `mbt2018`, `cheng2020-attn`, chất lượng 1–3,
        ở cạnh dài 1024 và 512; entropy coding thật, bitstream ghi ra file.
        """
    ),
    code(
        """
        for model in ('bmshj2018-hyperprior', 'mbt2018', 'cheng2020-attn'):
            for quality, side in [(q, s) for q in (1, 2, 3) for s in (1024, 512)]:
                curve = f'compressai-{model}_ls{side}'
                if scored(curve, f'q{quality}'):
                    continue
                run_logged([
                    sys.executable, '-u', 'tools/baselines/run_compressai_zoo.py', '--manifest', MANIFEST,
                    '--data-root', LOCAL_IMAGES, '--split', 'val', '--dev-list', DEV_LIST,
                    '--model', model, '--quality', quality, '--processing-long-side', side,
                    '--output', BASELINE_ROOT / curve / f'q{quality}', '--skip-existing',
                ], f'p2_{curve}_q{quality}.log', echo=False)
                print('Xong', curve, 'q', quality)
        """
    ),
    markdown(
        """
        ## Bước 8 — Chấm điểm mọi kho lưu trữ

        Với mỗi kho B0 trên Drive (`phase2/archive/B0_*/<điểm>/`) và mỗi kho baseline trên ổ cục bộ: chỉ số ảnh (PSNR, SSIM ×2, MS-SSIM, LPIPS, DISTS, bpp,
        compression ratio, thời gian) + MegaDetector trên ảnh tái tạo → mAP, ảnh rỗng báo nhầm, ảo giác,
        mất con vật. Tất cả ghi vào `results/results.jsonl` với `exp_id = p2dev_<đường>__<điểm>`.
        Chỉ chấm đúng các ảnh đã có trong kho (`decode_log.jsonl`); kho đã chấm thì bỏ qua, trừ khi kho
        có thêm ảnh từ sau lần chấm trước (ví dụ B0 được decode thêm) — khi đó chấm lại.
        """
    ),
    code(
        """
        try:
            import pycocotools  # noqa: F401
        except ImportError:
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'pycocotools'], check=True)

        import tarfile

        def pack_bitstreams(root, curve, point):
            # One file on Drive per point instead of hundreds of small writes.
            target = BITSTREAMS / curve / f'{point}.tar'
            target.parent.mkdir(parents=True, exist_ok=True)
            with tarfile.open(target, 'w') as bundle:
                for path in sorted(root.rglob('*')):
                    if path.is_file() and (path.parent.name == 'data' or path.name in ('decode_log.jsonl', 'run_info.json')):
                        bundle.add(path, arcname=path.relative_to(root).as_posix())

        archives = sorted(ARCHIVE.glob('B0_*/*/run_info.json')) + sorted(BASELINE_ROOT.glob('*/*/run_info.json'))
        for run_info in archives:
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
            # Score the image set this archive was asked for (a B0 subset), never
            # leftovers of an earlier, differently sized run in the same folder.
            archive_dev = info.get('dev_list') or DEV_LIST
            common = ['--manifest', MANIFEST, '--split', 'val', '--dev-list', archive_dev,
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
                '--split', 'val', '--dev-list', archive_dev, '--image-root', root, '--suffix', '.png', '--archived-only',
                '--output', predictions,
            ], f'p2_megadetector_{name}.log', echo=False)
            run_logged([
                sys.executable, '-u', 'tools/eval_machine.py', *common,
                '--gt', DETECTIONS, '--pred', predictions, '--output', machine_eval, '--archived-only',
                '--results-registry', RESULTS_REGISTRY, '--exp-id', f'p2dev_{name}',
                '--method', curve, '--lambda-rate', point, '--ddim-steps', info.get('steps', 0),
            ], f'p2_machine_{name}.log', echo=False)
            done_marker.write_text(json.dumps({'images': archived}))
            if BASELINE_ROOT in root.parents:
                pack_bitstreams(root, curve, point)
                shutil.rmtree(root)  # ~3 GB of PNGs per point; reproducible from the bitstream tar
            print('Đã chấm:', name, f'({archived} ảnh)')
        """
    ),
    markdown(
        """
        ## Bước 9 — Đồ thị RD chung (tiêu chí 7 của Phase 2)

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
        ## Bước 10 — Tóm tắt để gửi lại

        In nhãn ngày/đêm theo nguồn, vài con số thống kê miền và bảng kết quả từng điểm (tất cả ảnh). Chụp lại
        output cell này cùng `phase2/rd_dev.png`.
        """
    ),
    code(
        """
        import collections

        labels = [json.loads(line) for line in ILLUMINATION.read_text(encoding='utf-8').splitlines() if line.strip()]
        print('Ngày/đêm:', dict(collections.Counter(r['illumination'] for r in labels)),
              '| nguồn:', dict(collections.Counter(r['illumination_source'] for r in labels)),
              '| giờ chụp → nhãn mới:', dict(collections.Counter(f"{r.get('illumination_hour_proxy')}->{r['illumination']}" for r in labels)))
        stats_all = json.loads((P2 / 'domain_stats.json').read_text())['splits']['all']
        print('Ảnh rỗng (nhãn người):', round(stats_all['empty_ratio_human_labels'], 3),
              '| bbox con vật:', stats_all['animal_boxes'],
              '| cỡ COCO gốc:', stats_all['coco_size_original'],
              '| ở 1024:', stats_all.get('coco_size_at_long_side_1024'))

        columns = ['bpp', 'psnr', 'ms_ssim', 'lpips', 'dists', 'map', 'ap_small', 'hallucination_rate', 'missed_animal_rate']
        print(f"{'điểm':52s}" + ''.join(f'{c[:10]:>11s}' for c in columns))
        for exp_id, values in sorted(points.items()):
            print(f'{exp_id[len("p2dev_"):]:52s}' + ''.join(
                f'{values[c]:11.4f}' if values.get(c) is not None else f"{'-':>11s}" for c in columns))
        """
    ),
    markdown(
        """
        ## Bước 11 — Ghi CU đo thật của phiên

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
    ),]

for notebook_cells in (PREPARE_CELLS, EVAL_CELLS):
    for cell in notebook_cells:
        if cell["cell_type"] == "code":
            for placeholder, snippet in SNIPPETS.items():
                if placeholder in cell["source"]:
                    cell["source"] = cell["source"].replace(placeholder, snippet)


def _notebook(cells, name):
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"name": name, "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


NOTEBOOKS = {
    "Wild_Diff_ICMH_Phase2_Prepare.ipynb": PREPARE_CELLS,
    "Wild_Diff_ICMH_Phase2_Eval.ipynb": EVAL_CELLS,
}

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    for name, notebook_cells in NOTEBOOKS.items():
        output = root / name
        output.write_text(json.dumps(_notebook(notebook_cells, name), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(output)
