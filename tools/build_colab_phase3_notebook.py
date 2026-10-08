"""Build the thin, ordered Phase 3 H1 notebook; operations live in phase3_h1.py."""
from __future__ import annotations

import json
from pathlib import Path

from build_colab_training_notebook import cells as SETUP, code, markdown
from build_colab_phase2_notebook import SNIPPETS


def reused(prefix):
    for index, cell in enumerate(SETUP):
        if cell["cell_type"] == "markdown" and cell["source"].lstrip("#").strip().startswith(prefix):
            copied = dict(SETUP[index + 1])
            copied["source"] = copied["source"].replace(
                "BRANCH = 'phase2'  # test branch; set back to 'main' when merging", "BRANCH = 'main'")
            return copied
    raise KeyError(prefix)


cells = [
    markdown("""
    # Wild-Diff-ICMH — Phase 3: H1 pooled pilot (λ = 2)

    Một mô hình học từ **train của cả 20 site**, không phải một mô hình/site.
    Resize cạnh dài 512, crop ngẫu nhiên 256; không dùng ROI loss hoặc H3.
    **L4** là GPU đã chạy dự án. Chưa có kết quả Phase 3 trước khi bạn chạy notebook này.

    Runtime mới: chạy **1 → 6**, rồi **7 (smoke một lần) → 8 → 9 → 10**.
    Bước 8 đạt **500 optimizer steps tổng cộng**, không phải thêm 500 mỗi lần ấn.
    Phiên hết thời gian: chạy lại Bước 8 để tiếp tục checkpoint, không tạo folder mới.
    Bước 9 dùng 30 ảnh validation cố định, cùng protocol B0; không dùng test.
    **11 và 12 tắt mặc định**: chỉ bật sau khi đã xem kết quả và ngân sách.

    Khi có bản sửa: chạy **2A**, rồi đúng cell vừa lỗi. Pull cập nhật script/config
    trong repo, không tự cập nhật các cell notebook đang mở. Nếu chính notebook
    đổi cấu trúc, mở lại link này; không copy lẻ cell.
    Code /content mất khi runtime bị thu hồi; checkpoint, log, kết quả trên Drive còn.
    Không chạy lại notebook Phase 2, không tạo lại baseline/tags, không chạy Xie-SGC.
    """),
    markdown("## 1 — Gắn Drive và ghi ngân sách phiên"),
    code("""
    from google.colab import drive
    drive.mount('/content/drive')
    CU_AVAILABLE_AT_START = None  # điền số Available trong bảng Tài nguyên
    CU_RATE = 1.54  # thay bằng tốc độ CU/giờ Colab đang báo; chỉ dùng để ước tính
    SESSION_HOURS = 1.5  # giới hạn mỗi lần chạy training, có checkpoint cuối phiên
    """),
    markdown("## 2 — Lấy code nhánh main / xem GPU (mỗi runtime mới)"),
    reused("Bước 2 "),
    markdown("## 2A — Cell pull cố định: chạy sau khi được báo đã push bản sửa"),
    reused("Bước 2A"),
    markdown("## 3 — Cài môi trường đã kiểm tra cho Colab (giữ NumPy/SciPy/Torch)"),
    reused("Bước 3 "),
    markdown("## 4 — Chép ảnh đã có trên Drive, kiểm tra split (không tải lại dataset)"),
    reused("Bước 4 "),
    markdown("## 5 — Chép checkpoint gốc λ = 2 và Stable Diffusion (không tải lại)"),
    reused("Bước 5 "),
    markdown("""
    ## 6 — Kiểm tra tài sản Phase 2 / đóng băng 30 ảnh validation (CPU)

    Không tạo RAM tags mới. Cần KGA_all.jsonl, detections/originals.jsonl,
    kgalagadi_illumination.jsonl và kgalagadi_dev_b0eval.txt trên Drive.
    Tập 30 được phân tầng ngày/đêm, ảnh trống và kích thước con vật; đây là
    tập kiểm tra chọn mô hình, không đại diện tỷ lệ tự nhiên của toàn corpus.
    """),
    code("""
    import json
    import shutil

    def p3(action, *options):
        # Each call launches the currently pulled helper, not cached imports.
        subprocess.run([
            sys.executable, '-u', 'tools/phase3_h1.py', action,
            '--drive-root', str(DRIVE_ROOT), '--cu-rate', str(CU_RATE), *map(str, options),
        ], cwd=REPO, check=True)

    p3('prepare')
    print('Run H1:', DRIVE_ROOT / 'runs/h1_pooled_pilot_ls512/lambda_2')
    """),
    markdown("""
    ## 7 — Smoke cấu hình mới: train 20 step, decode 4 ảnh × 5 bước (một lần)

    Run smoke riêng, không dùng checkpoint smoke làm kết quả pilot.
    Chạy lại tự bỏ qua train đã đủ / ảnh đã decode. Không so DDIM5 với B0 DDIM50.
    """),
    code("""
    p3('train', '--target-steps', 20, '--session-hours', SESSION_HOURS)
    p3('evaluate', '--scope', 'smoke')
    """),
    markdown("""
    ## 8 — Train pilot đến 500 optimizer steps (GPU, việc chính)

    Không sinh ảnh trong training; validation chỉ tính loss trên 2 batch.
    Checkpoint rolling mỗi 100 optimizer steps, lưu đầy đủ trạng thái cuối phiên.
    Nếu runtime bị ngắt đột ngột, có thể mất phần sau checkpoint gần nhất.
    Giới hạn mỗi phiên SESSION_HOURS; chưa đủ 500 thì chạy lại cell này.
    """),
    code("""
    p3('train', '--target-steps', 500, '--session-hours', SESSION_HOURS)
    p3('status')
    """),
    markdown("""
    ## 9 — Đánh giá nhanh 30 ảnh và đối chiếu B0 đã lưu

    Snapshot checkpoint cố định trước khi decode. Cả B0 và H1: cạnh dài 512,
    DDIM50, seed231, guidance3, ảnh tái tạo đánh giá ở kích thước gốc.
    Đo dung lượng, chất lượng ảnh (gồm LPIPS), giữ con vật/mất con vật/phát hiện
    giả bằng MegaDetector. Không chạy SpeciesNet/DISTS tại pilot.
    MegaDetector gốc là nhãn tự động, không phải hộp chuẩn do người gán.
    Có ảnh/kết quả đủ rồi thì bỏ qua; chỉ chấm lại B0 trên tập nhỏ, không decode B0.
    """),
    code(SNIPPETS["__RUN_LOGGED__"] + "\n\n" + SNIPPETS["__DETECT_ENV__"] + "\n\n" +
         "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'pycocotools'], check=True)\n"
         "p3('evaluate', '--scope', 'quick', '--detect-python', DETECT_PY)\n"),
    markdown("## 10 — Xem chi phí / gửi kết quả để quyết định bước tiếp theo"),
    code("""
    p3('status')
    comparisons = sorted((DRIVE_ROOT / 'phase3/eval').glob('step_*_quick/comparison.json'))
    for path in comparisons:
        report = json.loads(path.read_text())
        print('Kết quả:', path)
        for method, values in report['methods'].items():
            quality = values['quality']['illumination=all|content=all']
            machine = values['machine']['illumination=all|content=all']
            print(method, {key: quality.get(key) for key in ('n', 'bpp', 'psnr', 'lpips')})
            print(method, {key: machine.get(key) for key in ('map', 'missed_animal_rate', 'hallucination_rate')})
    print('Gửi comparison.json + throughput.json. Không tự kết luận H1 tốt hơn từ 30 ảnh.')
    """),
    markdown("""
    ## 11 — Tùy chọn: kéo dài đến 1.000 hoặc 3.000 step (TẮT)

    Chỉ bật khi đã review pilot, đủ ngân sách và được chọn tiếp tục.
    Đây là tổng step, resume đúng run; không train lại từ đầu. Sau đó chạy 9 → 10.
    """),
    code("""
    RUN_EXTENSION = False
    EXTEND_TO = 1000  # chỉ 1000 hoặc 3000
    if RUN_EXTENSION:
        assert EXTEND_TO in (1000, 3000)
        p3('train', '--target-steps', EXTEND_TO, '--session-hours', SESSION_HOURS, '--allow-extension')
    else:
        print('Bỏ qua kéo dài training.')
    """),
    markdown("""
    ## 12 — Tùy chọn: đánh giá toàn bộ dev202 của mô hình được chọn (TẮT)

    Chỉ bật sau khi chọn checkpoint cuối. Không chạy lại baseline, không dùng test.
    Tốn hơn dev30; xem ngân sách trước. Không tự bật qua Run all.
    """),
    code("""
    RUN_FULL_DEV = False
    if RUN_FULL_DEV:
        p3('evaluate', '--scope', 'full', '--allow-full', '--detect-python', DETECT_PY)
    else:
        print('Bỏ qua full dev; Phase 3 chưa được confirm hoàn thành.')
    """),
    markdown("## 13 — Kết thúc phiên: ghi CU thực và ngắt GPU (chủ động bật)"),
    code("""
    CU_AVAILABLE_AT_END = None  # ghi số Available mới
    if CU_AVAILABLE_AT_START is not None and CU_AVAILABLE_AT_END is not None:
        print('CU thực tiêu cả phiên:', CU_AVAILABLE_AT_START - CU_AVAILABLE_AT_END)
    DISCONNECT_NOW = False  # bật khi các việc cần chạy đã xong để không giữ GPU nhàn rỗi
    if DISCONNECT_NOW:
        from google.colab import runtime
        drive.flush_and_unmount()
        runtime.unassign()
    """),
]


def build():
    return {"cells": cells, "metadata": {"colab": {"name": "Wild_Diff_ICMH_Phase3_H1.ipynb"},
             "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
             "accelerator": "GPU"}, "nbformat": 4, "nbformat_minor": 0}


if __name__ == "__main__":
    path = Path(__file__).resolve().parents[1] / "Wild_Diff_ICMH_Phase3_H1.ipynb"
    path.write_text(json.dumps(build(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(path)
