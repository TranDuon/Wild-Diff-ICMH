---
quick_id: 261008-inp
slug: phase2-results-summary
mode: quick
date: 2026-10-08
---

# Quick 261008-inp — Đưa kết quả Phase 2 vào repo cho cả nhóm

Kết quả Phase 2 chỉ nằm trên Drive của một người. Đưa bản tóm tắt vào repo để thành viên khác đọc được.

## Tasks

1. `tools/summarize_phase2.py` + `tests/test_summarize_phase2.py`: đọc `results.jsonl`, giữ dòng `p2dev_*`
   (`subset == all`, dòng mới nhất thắng), ghi CSV (all/day/night + CI 95%), bảng Markdown và `rd_dev.png`.
   Verify: `python -m pytest -q tests/test_summarize_phase2.py`.
2. Chạy script trên `tmp/results.jsonl` (bản tải từ Drive, 06/10) → `phases/02-*/results/`.
3. `02-RESULTS.md`: giao thức, bảng chính, nhận xét, hệ quả cho Phase 3, giới hạn. Không commit
   `results.jsonl` (7 MB, có số từng ảnh).
