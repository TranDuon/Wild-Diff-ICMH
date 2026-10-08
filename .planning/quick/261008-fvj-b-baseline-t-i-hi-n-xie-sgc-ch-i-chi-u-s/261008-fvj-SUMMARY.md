---
quick_id: 261008-fvj
slug: remove-xie-sgc-reproduction
status: complete
date: 2026-10-08
implementation_commit: e1fbe92
---

# Summary — Bỏ yêu cầu tái hiện Xie-SGC

## Đã thay đổi

- Xoá requirement EVAL-17 và file đặc tả triển khai `XIE-SGC-SPEC.md`.
- Gỡ Xie-SGC/Xie-FT khỏi baseline phổ quát, Phase 3, Phase 4, Phase 6 và ngân sách GPU.
- Phase 3 trở lại đúng phạm vi H1 + viết Method/Setup; không có lượt train nào cho bài Xie.
- Giữ Xie et al. 2025 trong ANLS-09 dưới dạng literature comparison: chỉ trích số công bố, ghi rõ
  khác dữ liệu/split/giao thức và không tuyên bố thắng trực tiếp.
- Tổng requirement trở lại 70 và đều được map vào phase.
- Giữ quick task cũ như lịch sử; `STATE.md` đánh dấu quyết định tái hiện ngày 07/10 đã superseded.

## Kiểm tra

- `git diff --check`: đạt.
- Requirement definitions: 70; traceability rows: 70.
- `EVAL-17` không còn trong `REQUIREMENTS.md` hoặc `ROADMAP.md` hiện hành.
- `XIE-SGC-SPEC.md` không còn tồn tại.
- Phase 3 chỉ liệt kê H1-01..06 và REPT-02.

