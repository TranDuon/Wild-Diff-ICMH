"""Generate the CPU-first, explicitly gated 200-step H1 recovery notebook."""
from __future__ import annotations

import json
from pathlib import Path
from textwrap import indent

from build_colab_training_notebook import code, markdown
from build_colab_phase3_notebook import reused
from build_colab_phase2_notebook import SNIPPETS


def gated(source: str, condition: str = "RUN_TRIAL or RUN_EVALUATION"):
    """No expensive imports, environment setup or copying outside this gate."""
    return code(f"if {condition}:\n" + indent(source.strip(), "    ") +
                "\nelse:\n    print('Bỏ qua: chưa bật hành động GPU này.')\n")


dependency_setup = reused("Bước 3 ")["source"].replace("subprocess.run(", "run_checked(")
h2_dependency_setup = dependency_setup.replace(
    "/content/requirements-colab-safe.txt", "/content/requirements-colab-safe-h2.txt",
).replace(
    "/content/colab-core-constraints.txt", "/content/colab-core-constraints-h2.txt",
)
image_setup = reused("Bước 4 ")["source"].replace(
    "DRIVE_ROOT = Path('/content/drive/MyDrive/wild_diff_icmh')",
    "# DRIVE_ROOT đã chọn ở Bước 1; không đổi đường dẫn giữa chừng.",
).replace("subprocess.run(", "run_checked(")
checkpoint_setup = reused("Bước 5 ")["source"].replace(
    "    Path('ram/ram_plus_swin_large_14m.pth'),\n", ""
).replace("RAM_CKPT = CKPT_ROOT / 'ram' / 'ram_plus_swin_large_14m.pth'\n", "").replace(
    "print('RAM++:', RAM_CKPT)\n", ""
)
detector_setup = SNIPPETS["__DETECT_ENV__"].replace(
    "Path('/content/envs/detect')", "Path('/content/envs/h1-recovery-detect')"
).replace(
    "shutil.rmtree(DETECT_ENV, ignore_errors=True)  # leftovers of a failed attempt",
    "DETECT_ENV.mkdir(parents=True, exist_ok=True)  # no deletion of another environment",
).replace("p2_detect_env.log", "h1_recovery_detect_env.log")


cells = [
    markdown("""
    # Wild-Diff-ICMH — H1 Recovery và H2 ROI pilot

    **Runtime L4 mới sau khi phiên cũ đã bị xóa:** `/content` cũ không còn, nhưng
    đối chứng H1 đã hoàn tất vẫn nằm trên Drive. Trong chính notebook này, chạy
    **1 → 2 → H2-A → H2-B → H2-C → H2-D** ở cuối. H2-A chỉ đặt cờ TẮT; bạn phải chủ động
    bật lần lượt hai cell setup H2 trước khi hiện bảng điều khiển. Không chạy
    lại các cell H1 3–9, đặc biệt **không chạy H1 4–8**. H2 preflight/duyệt
    overlay/train200/eval30/status đều là thao tác riêng trong bảng nút.
    **Không dùng Run All**: các cell H1 lịch sử có tác dụng riêng nếu bật cờ.

    **Phần H1 bên dưới chỉ giữ để tái lập lịch sử, không chạy cho H2:** H1 cũ đã
    tới 1.000 bước và Recovery200 đã hoàn tất. Giữ nguyên run/checkpoint/kết quả.
    Trước đây có thể chạy 1 → 2 → 3 bằng CPU để đọc báo cáo H1;
    Không cần GPU, chép toàn bộ ảnh, mở checkpoint nhiều gigabyte hay cài detector để đọc báo cáo.

    Trong quy trình H1 lịch sử, chỉ khi báo cáo cho phép và bạn đồng ý mới bật Bước 4, chuyển runtime
    sang **L4**, rồi chạy lại **1 → 4 → 5 → 6 → 7**. Run mới có tối đa **200 bước tổng**,
    khởi đầu từ trọng số tác giả, optimizer mới; tốc độ học chính 0,00001,
    phụ 0,001. Đây là giả thuyết cần kiểm tra, không phải lỗi tốc độ học đã được xác nhận.
    Sau đó bật riêng Bước 8 để đánh giá đúng 30 ảnh; Bước 9 đọc kết quả và quyết định.

    **Run all mặc định không train hoặc sinh ảnh.** Không chạy lại B0, Xie-SGC, 3.000 bước,
    toàn bộ validation hay test. Phase 3 chưa hoàn thành chỉ vì notebook chạy hết.
    Sửa script sau đó dùng **2A** rồi chạy lại cell lỗi nếu repo sạch. Pull không thay cấu trúc
    notebook đang mở. Hướng dẫn H2 cho phiên mới: `docs/H2_ROI_PILOT.md`.
    """),
    markdown("## 1 — Gắn Drive, chọn đường dẫn (CPU được)"),
    code("""
    from google.colab import drive
    from pathlib import Path
    import hashlib
    import json
    import os
    import shutil
    import subprocess
    import sys

    drive.mount('/content/drive')
    DRIVE_ROOT = Path('/content/drive/MyDrive/wild_diff_icmh')
    CU_RATE = 1.54  # dùng số CU/giờ Colab đang báo, chỉ là ước tính
    CU_AVAILABLE_AT_START = None  # điền Available để đo chi phí thật cả phiên
    REVIEW_PATH = DRIVE_ROOT / 'phase3/h1_low_lr_recovery/review.json'
    REVIEW_MD = REVIEW_PATH.with_suffix('.md')
    RUN_DIR = DRIVE_ROOT / 'runs/h1_low_lr_recovery_ls512/lambda_2'
    RECOVERY_ARTIFACTS = DRIVE_ROOT / 'phase3/h1_low_lr_recovery'
    """),
    markdown("## 2 — Lấy main, không yêu cầu GPU"),
    code("""
    REPO_URL = 'https://github.com/TranDuon/Wild-Diff-ICMH.git'
    BRANCH = 'main'
    REPO = Path('/content/Wild-Diff-ICMH')

    def run_checked(command, cwd=None, check=True, **kwargs):
        command = list(map(str, command))
        with subprocess.Popen(
            command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, **kwargs,
        ) as process:
            for line in process.stdout:
                print(line, end='', flush=True)
            return_code = process.wait()
        if check and return_code:
            raise subprocess.CalledProcessError(return_code, command)
        return return_code

    def safe_pull():
        current_branch = subprocess.check_output(
            ['git', 'branch', '--show-current'], cwd=REPO, text=True,
        ).strip()
        assert current_branch == BRANCH, f'Đang ở nhánh {current_branch}, không tự đổi nhánh hay ghi đè code.'
        local_changes = subprocess.check_output(
            ['git', 'status', '--porcelain'], cwd=REPO, text=True,
        ).strip()
        if local_changes:
            raise RuntimeError('Code có thay đổi cục bộ; giữ nguyên và gửi log để xử lý:\\n' + local_changes)
        before = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
        run_checked(['git', 'pull', '--ff-only', 'origin', BRANCH], cwd=REPO)
        after = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
        remote = subprocess.check_output(['git', 'rev-parse', f'origin/{BRANCH}'], cwd=REPO, text=True).strip()
        assert after == remote, f'Commit local {after} chưa khớp GitHub {remote}'
        print('Commit trước:', before[:12], 'sau:', after[:12])
        print('PULL THÀNH CÔNG — chạy lại đúng cell vừa lỗi.', flush=True)

    if (REPO / '.git').is_dir():
        safe_pull()
    else:
        run_checked(['git', 'clone', '--branch', BRANCH, REPO_URL, str(REPO)])
    os.chdir(REPO)
    print('Mã nguồn:', REPO)

    def recovery(action, *options):
        # A new child uses the latest pulled script, not a stale imported module.
        command = [sys.executable, '-u', 'tools/phase3_h1_recovery.py', action,
                   '--drive-root', str(DRIVE_ROOT), '--cu-rate', str(CU_RATE), *map(str, options)]
        print('$', ' '.join(command), flush=True)
        with subprocess.Popen(
            command, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        ) as process:
            for line in process.stdout:
                print(line, end='', flush=True)
            return_code = process.wait()
        if return_code:
            raise subprocess.CalledProcessError(return_code, command)
    """),
    markdown("## 2A — Pull cố định sau khi có bản sửa đã push"),
    code("""
    assert (REPO / '.git').is_dir(), 'Chạy Bước 2 trước.'
    safe_pull()
    """),
    markdown("""
    ## 3 — Đọc và tính lại kết quả đã lưu trên đúng 30 ảnh (CPU)

    Chỉ cần gói chấm điểm nhỏ `pycocotools`; không cài mô hình học sâu hay tạo lại ảnh.
    Báo cáo so B0 ở ba mức nén với H1 500/1.000 bước trên cùng danh sách ảnh.
    Dữ liệu/protocol thiếu hoặc khác nhau sẽ dừng, không lặng lẽ ghép các kết quả.
    Nếu thiếu ảnh để nhìn hoặc log thành phần loss, báo cáo ghi rõ chưa kiểm tra được.
    """),
    code("""
    import importlib.util
    if importlib.util.find_spec('pycocotools') is None:
        run_checked([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', 'pycocotools==2.0.11'])
    recovery('review')
    recovery('status')
    print('Báo cáo:', REVIEW_PATH)
    if REVIEW_MD.is_file():
        from IPython.display import Markdown, display
        display(Markdown(REVIEW_MD.read_text(encoding='utf-8')))
    """),
    markdown("""
    ## 4 — Quyết định và ngân sách: tất cả hành động nặng đang TẮT

    Đọc báo cáo trước. Nếu được chọn thử, đổi `RUN_TRIAL = True` và
    `ACKNOWLEDGE_REVIEW = REVIEW_SHA256` (không cần copy đoạn code khác).
    Không bật đánh giá cùng lúc nếu chưa train đủ 200 bước. Mỗi runtime mới chạy 1–4 lại.
    Nếu hash thay đổi, phải đọc báo cáo mới và xác nhận lại, không dùng quyết định cũ.

    Ước lượng **riêng training**: 200 × 5,65 giây ≈ 18,8 phút, khoảng 0,48 CU
    nếu mức tiêu thụ là 1,54 CU/giờ; **chưa gồm setup, nạp mô hình, checkpoint hay đánh giá**.
    Có thể dừng theo thời gian rồi chạy lại Bước 7; chỉ resume run mới, không học lại từ đầu.
    """),
    code("""
    assert REVIEW_PATH.is_file(), 'Chạy Bước 3 để tạo báo cáo trước.'
    REVIEW_SHA256 = hashlib.sha256(REVIEW_PATH.read_bytes()).hexdigest()
    print('Hash báo cáo cần xác nhận:', REVIEW_SHA256)
    RUN_TRIAL = False
    RUN_EVALUATION = False
    ACKNOWLEDGE_REVIEW = ''  # sau khi đọc báo cáo: đổi thành REVIEW_SHA256
    SESSION_HOURS = 0.5  # 0 < số giờ <= 2; mỗi lượt train, không gồm setup

    def require_review_acknowledgement():
        current_hash = hashlib.sha256(REVIEW_PATH.read_bytes()).hexdigest()
        assert current_hash == REVIEW_SHA256, 'Báo cáo đã đổi; đọc lại Bước 3 → 4.'
        assert ACKNOWLEDGE_REVIEW == current_hash, 'Chưa xác nhận đã đọc đúng báo cáo hiện tại.'
        review = json.loads(REVIEW_PATH.read_text(encoding='utf-8'))
        assert review.get('training_allowed') is True, 'Báo cáo chưa cho phép thử; xem lý do trước.'
        assert review.get('protocol_verified') is True, 'Chưa xác nhận cùng protocol; không mở hành động nặng.'
        assert review.get('recommendation') == 'conditional_low_lr_trial', 'Quyết định không phải lượt thử tốc độ học thấp.'
        assert 0 < SESSION_HOURS <= 2, 'Giới hạn phiên phải lớn hơn 0 và không quá 2 giờ.'

    print('Run mới:', RUN_DIR)
    print('Artifacts riêng:', RECOVERY_ARTIFACTS)
    print('Chỉ đọc kết quả: dừng tại đây, không cần GPU.')
    """),
    markdown("""
    ## 5 — Chỉ khi đã bật thử/đánh giá: kiểm tra GPU và môi trường đã dùng

    Chọn L4 rồi **Lưu** cấu hình runtime và kết nối phiên mới. Nếu chưa có GPU thực,
    cell dừng trước khi cài môi trường; không chỉ bỏ qua lỗi rồi train trên CPU.
    Giữ Torch/NumPy/SciPy của Colab, dùng quy trình cài đặt đã kiểm tra của dự án.
    """),
    gated("""
require_review_acknowledgement()
assert shutil.which('nvidia-smi'), 'Phiên này chưa có GPU. Chọn L4, Lưu, kết nối lại rồi chạy 1–4.'
run_checked(['nvidia-smi'])
""" + dependency_setup),
    markdown("""
    ## 6 — Chỉ sau khi đồng ý: chép tài sản training đã có (không tạo tags/B0)

    Chép ảnh và checkpoint từ Drive vào `/content` có thể mất thời gian.
    Tự bỏ qua file đã đủ; chỉ chép checkpoint Stable Diffusion và tác giả λ = 2.
    Không tải/chép RAM++ weights, vì dùng bộ tags đã lưu. Không chạm run H1 cũ.
    """),
    gated("require_review_acknowledgement()\n" + image_setup + "\n\n" + checkpoint_setup),
    markdown("""
    ## 7 — Thử H1 tốc độ học thấp: tối đa 200 bước tổng (TẮT mặc định)

    Lần đầu nạp trọng số tác giả, không nạp optimizer H1 cũ. Lần sau chỉ tiếp tục
    checkpoint có hợp đồng và tốc độ học đúng của run mới. Đủ 200 sẽ không train thêm.
    Trong training chỉ tính loss, không sinh ảnh; ghi thành phần loss và tốc độ học
    để lần này có dữ liệu kiểm tra. Không đổi λ, crop, dữ liệu hay trọng số loss cùng lúc.
    """),
    code("""
    if RUN_TRIAL:
        require_review_acknowledgement()
        recovery('train', '--enable', '--acknowledge-review', ACKNOWLEDGE_REVIEW,
                 '--session-hours', SESSION_HOURS)
        recovery('status')
    else:
        print('Bỏ qua train: RUN_TRIAL vẫn False.')
    """),
    markdown("""
    ## 8 — Đánh giá riêng checkpoint mới trên 30 ảnh (TẮT mặc định)

    Sau khi đủ 200 bước, đổi `RUN_TRIAL = False`, `RUN_EVALUATION = True` ở Bước 4
    rồi chạy lại 4 → 5 → 6 → 8 → 9. Detector chỉ cài tại đây; không decode/chạy lại B0.
    Cùng cạnh dài 512, 50 bước lấy mẫu, seed 231, mức hướng dẫn 3, chấm ảnh ở kích thước gốc.
    Chi phí đánh giá ngoài ước tính training 0,48 CU phía trên.
    """),
    gated("""
require_review_acknowledgement()
def run_logged(command, log_name, echo=True):
    log_path = RECOVERY_ARTIFACTS / 'logs' / log_name
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = list(map(str, command))
    with log_path.open('a', encoding='utf-8') as log_stream:
        with subprocess.Popen(command, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, bufsize=1) as process:
            for line in process.stdout:
                if echo:
                    print(line, end='', flush=True)
                log_stream.write(line)
                log_stream.flush()
            return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)
""" + detector_setup + """

recovery('evaluate', '--enable', '--acknowledge-review', ACKNOWLEDGE_REVIEW,
         '--detect-python', DETECT_PY)
""", "RUN_EVALUATION"),
    markdown("""
    ## 9 — Đọc status/kết quả mới và gửi lại để quyết định (CPU)

    Gửi `review.json`, kết quả comparison mới, `throughput.json` và `loss_components.csv`
    trong run mới nếu đã train. Không kết luận mô hình tốt hơn chỉ từ 30 ảnh hoặc từ
    điểm có dung lượng khác nhau. Không tự kéo dài training, đánh giá rộng hoặc bật H2.
    """),
    code("""
    recovery('status')
    comparisons = sorted((RECOVERY_ARTIFACTS / 'eval').glob('**/comparison.json'))
    for path in comparisons:
        print('Kết quả mới:', path)
        comparison = json.loads(path.read_text(encoding='utf-8'))
        for method, values in comparison.get('points', {}).items():
            quality = values.get('quality', {})
            machine = values.get('machine', {})
            print(method, 'số ảnh:', quality.get('n'), 'bit/pixel:', quality.get('bpp'))
            print('  chất lượng gần gốc:', quality.get('psnr'), 'sai khác cảm nhận:', quality.get('lpips'))
            print('  điểm phát hiện:', machine.get('map'), 'ảnh bỏ sót:', machine.get('missed_animal_rate'),
                  'phát hiện giả:', machine.get('hallucination_rate'))
        print('Thông số đã xác minh:', comparison.get('protocol_verified'))
        print('Quyết định tham khảo:', comparison.get('recommendation'))
        loss_review = comparison.get('loss_review', {})
        print('Log thành phần loss:', loss_review.get('status'))
        for missing in loss_review.get('missing', []):
            print('  Chưa có bằng chứng:', missing)
    print('Báo cáo CPU:', REVIEW_PATH)
    print('Throughput nếu đã train:', RUN_DIR / 'throughput.json')
    print('Loss/tốc độ học nếu đã train:', RUN_DIR / 'loss_components.csv')
    print('Phase 3 chưa tự đánh dấu hoàn thành; H2 là bước thiết kế tiếp theo.')
    """),
    markdown("""
    ## 10 — Kết thúc phiên; chỉ chủ động bật khi đã xong

    Nếu đang giữ L4 mà không làm gì nữa, ngắt runtime để không tiêu thêm CU.
    Checkpoint và báo cáo đã ghi trên Drive còn; file `/content` có thể mất.
    """),
    code("""
    CU_AVAILABLE_AT_END = None
    if CU_AVAILABLE_AT_START is not None and CU_AVAILABLE_AT_END is not None:
        print('CU thật cả phiên:', CU_AVAILABLE_AT_START - CU_AVAILABLE_AT_END)
    DISCONNECT_NOW = False
    if DISCONNECT_NOW:
        from google.colab import runtime
        drive.flush_and_unmount()
        runtime.unassign()
    """),
]

# Keep the old H1 cells/controls intact. H2 setup is separate from H1 gates.
cells.extend([
    markdown("""
    ## H2-A — Cờ setup cho phiên L4 mới (TẮT mặc định)

    Sau 1 (Drive) và 2 (main) trong notebook này, chọn L4, rồi bật riêng
    `RUN_H2_DEPENDENCIES` để chạy H2-B. Sau đó bật riêng `RUN_H2_DETECTOR` để chạy
    H2-C. Cả hai cờ mặc định False; chạy cell chỉ đặt cờ, không cài gì. Không
    chạy H1 4–8; chúng là lịch sử của lượt H1 đã hoàn tất trên Drive.
    """),
    code("""
    RUN_H2_DEPENDENCIES = False  # đổi True rồi chỉ chạy H2-B; sau đó trả False
    RUN_H2_DETECTOR = False  # đổi True rồi chỉ chạy H2-C; sau đó trả False
    """),
    markdown("""
    ## H2-B — Cài dependency training H2, chuẩn bị đường dẫn checkpoint

    Chỉ chạy khi đã bật `RUN_H2_DEPENDENCIES` ở H2-A. Giữ nguyên NumPy/SciPy/
    Torch/Torchvision của Colab; kiểm tra import training và cài pycocotools.
    Chỉ tạo link `./checkpoints` tới ổ local; **không chép checkpoint/ảnh**.
    Runner H2 tự stage đúng hai checkpoint và asset theo từng thao tác sau.
    """),
    gated("""
assert (REPO / '.git').is_dir(), 'Chạy Bước 2 lấy main trước H2 setup.'
assert DRIVE_ROOT.is_dir(), 'Chạy Bước 1 gắn Drive trước H2 setup.'
assert shutil.which('nvidia-smi'), 'Chọn runtime L4 trước H2 setup.'
gpu_names = subprocess.check_output(
    ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], text=True,
).splitlines()
assert len(gpu_names) == 1 and gpu_names[0].strip() in ('NVIDIA L4', 'L4'), (
    f'H2 pilot yêu cầu đúng một L4; GPU hiện tại: {gpu_names}'
)
""" + h2_dependency_setup + """
import importlib.util
if importlib.util.find_spec('pycocotools') is None:
    run_checked([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', 'pycocotools==2.0.11'])
import pycocotools
CKPT_ROOT = Path('/content/data/wild_diff_icmh/checkpoints')
CKPT_ROOT.mkdir(parents=True, exist_ok=True)
repo_checkpoints = REPO / 'checkpoints'
if repo_checkpoints.is_symlink():
    assert repo_checkpoints.resolve() == CKPT_ROOT.resolve(), (
        f'{repo_checkpoints} không trỏ tới {CKPT_ROOT}'
    )
elif repo_checkpoints.exists():
    raise RuntimeError(f'{repo_checkpoints} đã tồn tại và không phải link checkpoint được duyệt')
else:
    repo_checkpoints.symlink_to(CKPT_ROOT, target_is_directory=True)
assert repo_checkpoints.is_symlink() and repo_checkpoints.resolve() == CKPT_ROOT.resolve()
print('H2 training imports, pycocotools và link checkpoint đã sẵn sàng; chưa chép tài sản.')
""", "RUN_H2_DEPENDENCIES"),
    markdown("""
    ## H2-C — Dựng detector riêng cho H2, không chấm H1

    Chỉ chạy khi đã bật `RUN_H2_DETECTOR` ở H2-A. Môi trường riêng tái sử dụng
    Torch của Colab và ghi log dưới H2 trên Drive. Cell này chỉ kiểm tra import
    PytorchWildlife và `.ready`; **không chạy detector trên ảnh nào**.
    """),
    gated("""
assert (REPO / '.git').is_dir(), 'Chạy Bước 2 lấy main trước H2 setup.'
assert DRIVE_ROOT.is_dir(), 'Chạy Bước 1 gắn Drive trước H2 setup.'
assert shutil.which('nvidia-smi'), 'Chọn runtime L4 trước H2 setup.'
gpu_names = subprocess.check_output(
    ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], text=True,
).splitlines()
assert len(gpu_names) == 1 and gpu_names[0].strip() in ('NVIDIA L4', 'L4'), (
    f'H2 pilot yêu cầu đúng một L4; GPU hiện tại: {gpu_names}'
)

def h2_run_logged(command):
    log_path = DRIVE_ROOT / 'phase3/h2_roi_v1_alpha2/logs/h2_detect_env.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = list(map(str, command))
    with log_path.open('a', encoding='utf-8') as log_stream:
        with subprocess.Popen(command, cwd=REPO, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
            for line in process.stdout:
                print(line, end='', flush=True)
                log_stream.write(line)
                log_stream.flush()
            return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)

DETECT_ENV = Path('/content/envs/h1-recovery-detect')
DETECT_PY = DETECT_ENV / 'bin/python'
DETECT_READY = DETECT_ENV / '.ready'
if not DETECT_READY.is_file():
    h2_run_logged([sys.executable, '-m', 'pip', 'install', '-q', 'virtualenv'])
    h2_run_logged([sys.executable, '-m', 'virtualenv', '--system-site-packages', DETECT_ENV])
    from importlib import metadata as importlib_metadata
    hub_pin = f"huggingface-hub=={importlib_metadata.version('huggingface_hub')}"
    h2_run_logged([DETECT_PY, '-m', 'pip', 'install', '-q', 'PytorchWildlife', hub_pin])
    h2_run_logged([DETECT_PY, '-c', 'from PytorchWildlife.models import detection; print("PytorchWildlife OK")'])
    DETECT_READY.touch()
assert DETECT_PY.is_file() and DETECT_READY.is_file(), 'H2 detector chưa sẵn sàng.'
h2_run_logged([DETECT_PY, '-c', 'from PytorchWildlife.models import detection; print("PytorchWildlife OK")'])
print('H2 detector đã sẵn sàng:', DETECT_PY)
""", "RUN_H2_DETECTOR"),
    markdown("""
    ## H2-D — Điều khiển pilot, mỗi việc là một nút riêng

    Chỉ dùng khi H1 Recovery đã train **200/200** và chấm xong dev30 trên Drive.
    Không chạy lại cell H1 4–8. Cell dưới chỉ hiển thị bảng nút; **không tự cài,
    chép ảnh, train, decode hay chạy detector**. Sau H2-B/C, lần lượt preflight
    ≤10 ảnh → xem ROI thật → xác nhận hash → train H2 200 → eval H2 dev30 → status.

    Run H2 riêng, khởi đầu cùng trọng số tác giả/optimizer mới như H1; chỉ đổi
    trọng số vùng con vật từ 1 lên 2. H1/B0 đã chấm được giữ nguyên và dùng lại.
    Ảnh local đủ dung lượng được bỏ qua; đánh giá chỉ chuẩn bị đúng 30 ảnh.
    Mỗi lần sửa: pull **2A** rồi chạy lại cell này để dùng giao diện/script mới.
    """),
    code("""
    # Dùng đúng REPO/DRIVE_ROOT của Bước 1/2 trong runtime hiện tại.
    RUN_TRIAL = False
    RUN_EVALUATION = False
    assert (REPO / '.git').is_dir(), 'Chạy Bước 2 lấy main trước H2-D.'
    assert DRIVE_ROOT.is_dir(), 'Chạy Bước 1 gắn Drive trước H2-D.'
    import runpy
    H2_CONTINUATION = runpy.run_path(
        str(REPO / 'tools/colab_h2_continuation.py'), run_name='__main__',
        init_globals={'REPO': REPO, 'DRIVE_ROOT': DRIVE_ROOT, 'CU_RATE': CU_RATE},
    )
    """),
])


def build():
    return {"cells": cells, "metadata": {
        "colab": {"name": "Wild_Diff_ICMH_H1_Recovery.ipynb", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }, "nbformat": 4, "nbformat_minor": 0}


if __name__ == "__main__":
    output = Path(__file__).resolve().parents[1] / "Wild_Diff_ICMH_H1_Recovery.ipynb"
    output.write_text(json.dumps(build(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)
