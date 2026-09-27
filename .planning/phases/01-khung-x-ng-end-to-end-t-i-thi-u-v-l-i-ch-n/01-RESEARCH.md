# Phase 1 Research — Khép vòng lặp Colab tối thiểu

**Ngày:** 2026-09-27

## Kết luận

Lát cắt end-to-end đã chạy thật: dependency, train 20 step, resume thêm một step,
decode hai ảnh và upsert metric vào `results.jsonl`. Không có lý do kỹ thuật để chạy
lại toàn bộ đường GPU. Phần còn thiếu là biến bằng chứng đang rải trong log thành
artifact máy đọc được và đo/fallback metadata trên 100 ảnh.

## Tài sản có thể tái sử dụng

- `tools/data/split_check.py` đã chặn leakage theo `sequence_id`; Kgalagadi cho phép
  cùng site xuất hiện ở ba split.
- `tools/data/download_images.py` đã ghi `exif_datetime`, `exif_status`, kích thước và
  grayscale vào `_meta/checksums.jsonl`.
- `utils/checkpoint_contract.py` đã xác thực contract version, `global_step` và
  `optimizer_states`.
- `utils/results_registry.py` đã cố định schema và upsert idempotent.
- `tools/build_colab_training_notebook.py` là nguồn sinh notebook; không sửa notebook
  bằng tay.

## Khoảng trống còn lại

1. Báo cáo EXIF hiện chỉ in ra stdout và không lọc theo site; cần JSON artifact cho
   đúng 100 ảnh KGA:A01 cùng tỷ lệ fallback sang datetime/site trong manifest.
2. Cần một closeout artifact gom split gate, negative leakage probe, checkpoint,
   resume, registry và dự báo chi phí 2K step.
3. Với tốc độ smoke quan sát được, chạy 2K mù có nguy cơ vượt ngân sách. Closeout phải
   ngoại suy trước và chỉ khuyến nghị chạy khi phần ngân sách còn lại đủ.

## Hướng triển khai

- Thêm `tools/data/audit_metadata.py` thuần CPU, JSON vào/ra, có unit test.
- Thêm `tools/phase1_closeout.py` thuần CPU ngoại trừ việc đọc header checkpoint qua
  `torch.load(..., mmap=True)`; tạo một JSON quyết định rõ `run_2k` hay
  `do_not_run_2k`.
- Sinh Bước 10 trong notebook để gọi hai tool và lưu artifact trên Drive.
- Không thêm logic cứu T4; người vận hành chọn L4 thủ công theo D-04.

## Rủi ro

- Checkpoint PyTorch dùng pickle: chỉ đọc checkpoint do chính run dự án tạo, không
  nhận đường dẫn checkpoint tùy ý từ nguồn không tin cậy.
- Manifest chứa đường dẫn: tool audit không mở đường dẫn ảnh, chỉ nối bằng `image_id`.
- Artifact Drive phải ghi nguyên tử để tránh file nửa chừng khi runtime ngắt.

