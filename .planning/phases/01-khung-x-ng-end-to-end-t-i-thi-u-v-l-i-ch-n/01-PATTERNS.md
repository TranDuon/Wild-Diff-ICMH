# Phase 1 Patterns

| File mới/sửa | Mẫu gần nhất | Quy ước giữ lại |
|---|---|---|
| `tools/data/audit_metadata.py` | `tools/data/download_images.py` | JSONL, `Path`, hàm CLI test được, không tải lại ảnh |
| `tools/phase1_closeout.py` | `tools/evaluate_kgalagadi.py` | CLI rõ ràng, JSON artifact, lỗi fail-closed |
| `tools/build_colab_training_notebook.py` | các Bước 7–9 | stream log, Drive là nơi lưu bền vững, builder là source of truth |
| `tests/data/test_audit_metadata.py` | `tests/data/test_download_images.py` | fixture nhỏ, không network |
| `tests/test_phase1_closeout.py` | `tests/test_phase1_completion_contract.py` | kiểm tra contract và quyết định ngân sách |

