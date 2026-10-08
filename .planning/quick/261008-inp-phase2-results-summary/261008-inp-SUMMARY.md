---
quick_id: 261008-inp
status: complete
date: 2026-10-08
---

# Summary — Kết quả Phase 2 vào repo

- `tools/summarize_phase2.py` (+ 3 test): `results.jsonl` → `results/phase2_points.csv`,
  `results/phase2_table.md`, `results/rd_dev.png`. Mỗi đường một màu (tab20).
- Chạy trên bản `results.jsonl` tải từ Drive (lượt Eval 06/10): 37 điểm, 13 đường, 202 ảnh dev.
- `02-RESULTS.md`: giao thức, bảng chính, nhận xét, hệ quả cho Phase 3 (pilot λ = 2, cạnh 512), giới hạn.
- `.gitignore`: thêm `tmp/*.jsonl`; `results.jsonl` không vào git.
- `python -m pytest -q`: 166 passed, 3 skipped.
