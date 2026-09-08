---
gsd_state_version: '1.0'
status: planning
progress:
  total_phases: 6
  completed_phases: 0
  total_plans: 0
  completed_plans: 0
  percent: 0
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-09-08)

**Core value:** Chứng minh được rằng cơ chế chuyên biệt hoá (H2 ROI-weighted loss hoặc H3 domain-aware TGM) đóng góp vượt trên fine-tuning thuần (H1).
**Current focus:** Phase 1 — Khung xương end-to-end tối thiểu + vá lỗi chặn

## Current Position

Phase: 1 of 6 (Khung xương end-to-end tối thiểu + vá lỗi chặn)
Plan: TBD (chưa lập plan)
Status: Ready to plan
Last activity: 2026-09-08 — ROADMAP.md và STATE.md được tạo từ REQUIREMENTS.md + research/SUMMARY.md

Progress: [░░░░░░░░░░] 0%

## Performance Metrics

**Velocity:**
- Total plans completed: 0
- Average duration: - min
- Total execution time: 0h

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| - | - | - | - |

**Recent Trend:**
- Last 5 plans: -
- Trend: -

*Updated after each plan completion*

## Accumulated Context

### Decisions

Đầy đủ ở PROJECT.md Key Decisions. Các quyết định định hình roadmap gần đây nhất:

- [Roadmap]: Phase 1 dựng dưới dạng MVP theo chiều dọc (walking skeleton) — mẫu nhỏ chạy đầu-cuối trước khi mở rộng toàn corpus ở Phase 2, thay vì xây trọn tầng dữ liệu rồi mới xây hạ tầng.
- [Roadmap]: H3 (Phase 5) lên lịch chạy **song song** với H1/H2 (Phase 3/4) vì tầng L1/L2 không cần training — đòn bẩy ngân sách lớn nhất dự án.
- [Roadmap]: Thứ tự hy sinh pre-committed — cắt điểm bitrate thừa trước, ablation cell sau, phân tích tuỳ chọn cuối cùng; không bao giờ cắt baseline, gate split_check.py, bảng ablation, bảng cái giá chuyên biệt hoá, hay Limitations.
- [Roadmap]: Sửa số liệu Coverage trong REQUIREMENTS.md — file có 59 requirement v1 có ID cụ thể, không phải 52 như dòng tổng ghi lúc định nghĩa requirements.

### Pending Todos

None yet.

### Blockers/Concerns

- **Rủi ro sinh tử #1 — rò rỉ dữ liệu theo site/burst**: phải được chặn bằng gate `split_check.py` tự động (DATA-03) ngay từ Phase 1, không phải script chạy tay. Theo dõi tới khi Phase 1 verify xong.
- **Rủi ro sinh tử #2 — cạn ngân sách compute-unit**: ngân sách trong ROADMAP.md là ước tính TRƯỚC đo lường; phải hiệu chỉnh lại bằng số đo thật ngay sau smoke-test Phase 1 (INFRA-03) và ở mọi ranh giới phase sau đó.
- Số liệu Coverage gốc trong REQUIREMENTS.md (52 requirement) không khớp nội dung thật (59 requirement có ID) — đã sửa khi tạo roadmap này; không phải lỗi mới phát sinh, chỉ ghi nhận để không gây nhầm lẫn khi đọc lại lịch sử.

## Deferred Items

Items acknowledged and deferred at milestone close, most recent first:

| Category | Item | Status | Deferred At | Milestone |
|----------|------|--------|-------------|-----------|
| *(none)* | | | | |

## Session Continuity

Last session: 2026-09-08
Stopped at: Roadmap tạo xong, chờ user duyệt (approval gate do orchestrator quản lý)
Resume file: None
