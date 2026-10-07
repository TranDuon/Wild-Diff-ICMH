---
quick_id: 261007-l4e
status: complete
date: 2026-10-07
commits: [9b17234, 3c2fdb2, 7df5b4e]
---

# Summary — Thêm baseline tái hiện Xie-SGC vào kế hoạch

Chỉ sửa tài liệu kế hoạch; chưa có code, notebook hay test nào thay đổi.

## Thay đổi

- `REQUIREMENTS.md` (9b17234): requirement mới **EVAL-17** (Xie-SGC tái hiện); EVAL-14 nhắc EVAL-17;
  **ANLS-09** ghi đã chốt Xie et al. 2025, nguyên tắc "không train lại" chỉ áp cho model Diff-ICMH,
  Xie-SGC là ngoại lệ có giới hạn (≤2 CU); traceability + Coverage 70 → 71.
- `ROADMAP.md` (3c2fdb2): mục "Cập nhật 07/10"; G-5; Phase 2 (requirement, ngân sách, tiêu chí 8,
  việc code, 7 plans); Phase 4 tiêu chí 6 (khác biệt H2 vs Xie-SGC, cặp Xie-FT → Xie-SGC tương ứng
  H1-control → H2); Phase 6 tiêu chí 6; bảng ngân sách và bảng phân việc.
- `02-CONTEXT.md`, `02-PLAN.md`, `PROJECT.md` (7df5b4e): giải thích Xie-SGC ở A2; **Plan 02-07** mô tả
  `tools/baselines/train_xie_sgc.py`, test local, bước notebook, đồ thị SSIM–compression ratio;
  thêm một dòng Key Decisions.
- `STATE.md`: Current Position, Decisions, Quick Tasks.

## Các lựa chọn diễn giải (bài gốc không nói rõ)

- "Encoder" = `g_a` + `h_a`; `g_s`, `h_s` và entropy model giữ nguyên để bên nhận dùng decoder pretrained.
- Trọng số nền 0,001 nhân vào MSE theo pixel, lấy trung bình trên mọi pixel (không chuẩn hoá theo tổng W).
- Ảnh rỗng vẫn dùng khi fine-tune (toàn bộ là nền).
- Chỉ q = 1/2/3 ở cạnh dài 512 (dải chồng với B0), không LoRA, không blur nền.

## Việc tiếp theo

Code Plan 02-07 (`train_xie_sgc.py` + test với model giả) rồi thêm bước vào notebook Eval; pilot một
site để đo thời gian trước khi chạy 20 site.
