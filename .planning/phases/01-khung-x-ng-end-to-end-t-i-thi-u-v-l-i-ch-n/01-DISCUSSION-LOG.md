# Phase 1: Khung xương end-to-end tối thiểu + vá lỗi chặn - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-09-27
**Phase:** 1-Khung xương end-to-end tối thiểu + vá lỗi chặn
**Areas discussed:** Cổng hoàn tất Phase 1, Chính sách GPU và ngân sách, Hợp đồng resume và checkpoint, Dữ liệu mẫu và kết quả

---

## Cổng hoàn tất Phase 1

| Option | Description | Selected |
|--------|-------------|----------|
| Dùng bằng chứng hiện có và đóng khoảng trống | Giữ smoke/resume/decode đã thành công; chỉ bổ sung gate còn thiếu | ✓ |
| Chạy lại toàn bộ từ đầu | Tái chạy mọi cell dù không có thay đổi hành vi | |
| Chuyển thẳng sang H1 dài | Bỏ qua metadata, leakage và compute calibration | |

**User's choice:** Agent chọn phương án tốt nhất; chọn dùng bằng chứng hiện có và đóng khoảng trống.
**Notes:** Cell 9 đã decode/evaluate thành công hai ảnh; kết quả chỉ là smoke, không phải số báo cáo.

---

## Chính sách GPU và ngân sách

| Option | Description | Selected |
|--------|-------------|----------|
| L4 thủ công, benchmark có giới hạn | Giữ một cấu hình chuẩn, tăng dần workload và dừng theo ngân sách | ✓ |
| Tự thích ứng mọi tier | Thêm logic T4/A100 và nhiều nhánh cấu hình | |
| Bắt buộc A100 | Tốn CU cao hơn và không phù hợp ngân sách mặc định | |

**User's choice:** Trước đó người dùng đã yêu cầu bỏ các thay đổi T4 và sẽ tự chọn L4.
**Notes:** Không tái đưa T4 auto-tuning vào plan.

---

## Hợp đồng resume và checkpoint

| Option | Description | Selected |
|--------|-------------|----------|
| Xác minh toàn trạng thái | Kiểm optimizer state, global_step tăng và checkpoint contract | ✓ |
| Chỉ kiểm tra load weights | Có thể warm-start lại từ đầu mà không phát hiện | |
| Chỉ nhìn log không lỗi | Không chứng minh optimizer/scheduler được khôi phục | |

**User's choice:** Agent chọn phương án tốt nhất; chọn xác minh toàn trạng thái.
**Notes:** Warm-start checkpoint tác giả và resume checkpoint dự án là hai semantics riêng.

---

## Dữ liệu mẫu và kết quả

| Option | Description | Selected |
|--------|-------------|----------|
| Lát cắt có kiểm chứng | KGA:A01, negative leakage fixture, metadata audit, registry idempotent | ✓ |
| Mở rộng ngay toàn corpus | Thuộc Phase 2, tăng I/O và phạm vi trước khi preflight đóng | |
| Chỉ giữ ảnh đã chạy thành công | Không chứng minh data contract hoặc metadata fallback | |

**User's choice:** Agent chọn phương án tốt nhất; chọn lát cắt có kiểm chứng.
**Notes:** Site có thể lặp giữa split; sequence/burst không được lặp.

## the agent's Discretion

- Độ dài calibration cụ thể và format artifact máy đọc được.
- Chia plan sao cho mọi gate local/CPU chạy trước GPU.

## Deferred Ideas

- Full-corpus/ROI/eval machine-task sang Phase 2.
- H1/H2/H3 và ablation giữ đúng roadmap hiện tại.
