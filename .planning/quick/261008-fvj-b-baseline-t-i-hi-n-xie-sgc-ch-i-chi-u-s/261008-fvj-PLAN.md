---
quick_id: 261008-fvj
slug: remove-xie-sgc-reproduction
status: complete
date: 2026-10-08
---

# Quick 261008-fvj — Bỏ yêu cầu tái hiện Xie-SGC

Người dùng chốt rằng bài Xie et al. 2025 chỉ được dùng để đối chiếu các số liệu đã công bố; dự án
không viết lại code, không fine-tune Ballé và không chạy baseline Xie-SGC/Xie-FT.

## Task 1 — Sửa kế hoạch hiện hành

- Xoá requirement EVAL-17 và đặc tả triển khai `XIE-SGC-SPEC.md`.
- Gỡ Xie-SGC khỏi baseline phổ quát, Phase 3, Phase 4, Phase 6 và ngân sách GPU.
- Giữ ANLS-09 nhưng đổi thành so sánh tài liệu: dùng số công bố của Xie, ghi rõ khác split/giao thức,
  không tuyên bố thắng trực tiếp.
- Giữ các quick task cũ như lịch sử, nhưng đánh dấu quyết định tái hiện ngày 07/10 là superseded.

## Task 2 — Kiểm tra tính nhất quán

- Xác nhận không còn requirement/plan/spec hiện hành yêu cầu code hoặc chạy Xie-SGC.
- Xác nhận tổng requirement trở lại 70 và Phase 3 chỉ còn H1 + viết Method/Setup.
- Cập nhật `STATE.md` và tạo summary cho quick task.

## Kết quả

Hoàn thành trong commit `e1fbe92`.

