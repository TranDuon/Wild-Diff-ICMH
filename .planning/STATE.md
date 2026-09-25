---
gsd_state_version: "1.0"
current_phase: 1
current_phase_name: Khung xương end-to-end tối thiểu + vá lỗi chặn
status: planning
stopped_at: Roadmap tạo xong, chờ user duyệt (approval gate do orchestrator quản lý)
last_updated: "2026-09-25T05:46:45.381Z"
last_activity: 2026-09-16
last_activity_desc: "Completed quick task 260916-sxk: Bước 1 fine-tune camera trap (manifest dữ liệu + downloader + split_check)"
state_head: 1b2012461fa4eb79a7bd527264046254cc857b7b
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
Last activity: 2026-09-16 - Completed quick task 260916-sxk: Bước 1 fine-tune camera trap (manifest dữ liệu + downloader + split_check)

Kế hoạch triển khai liên phase: `CAMERA_TRAP_FINE_TUNING_PLAN.md` (16/09/2026). Đây là kế hoạch thực hiện, chưa phải bằng chứng Phase 1 đã hoàn tất.

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
- [2026-09-24]: Protocol chính = toàn bộ 10.222 ảnh Snapshot Kgalagadi không có người, chia 70/15/15 theo sequence trong từng site và fine-tune một model/site giống bài so sánh. Bài không công bố split nên đây là protocol tái lập của dự án. Snapshot Serengeti chỉ đánh giá bổ sung; fine-tune trên Colab L4.
- [Roadmap]: Sửa số liệu Coverage trong REQUIREMENTS.md — file có 59 requirement v1 có ID cụ thể, không phải 52 như dòng tổng ghi lúc định nghĩa requirements.

### Pending Todos

None yet.

### Blockers/Concerns

- **Rủi ro sinh tử #1 — rò rỉ dữ liệu theo site/burst**: phải được chặn bằng gate `split_check.py` tự động (DATA-03) ngay từ Phase 1, không phải script chạy tay. Theo dõi tới khi Phase 1 verify xong.
  (260916-sxk: `tools/data/split_check.py` đã có và chạy trong build manifest + trước khi ghi .list; CHƯA được gọi tự động trong `train.py`/eval.)
- **Rủi ro sinh tử #2 — cạn ngân sách compute-unit**: ngân sách trong ROADMAP.md là ước tính TRƯỚC đo lường; phải hiệu chỉnh lại bằng số đo thật ngay sau smoke-test Phase 1 (INFRA-03) và ở mọi ranh giới phase sau đó.
- Số liệu Coverage gốc trong REQUIREMENTS.md (52 requirement) không khớp nội dung thật (59 requirement có ID) — đã sửa khi tạo roadmap này; không phải lỗi mới phát sinh, chỉ ghi nhận để không gây nhầm lẫn khi đọc lại lịch sử.

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 260916-sxk | Bước 1 fine-tune camera trap: manifest Serengeti train/val + Kgalagadi test, downloader, split_check | 2026-09-16 | 3d9b82b | [260916-sxk-buoc-1-fine-tune-camera-trap-dung-manife](./quick/260916-sxk-buoc-1-fine-tune-camera-trap-dung-manife/) |
| 2 | Sắp xếp notebook Colab theo thứ tự và sửa dependency/progress | 2026-09-25 | 1b20124 | — |

## Deferred Items

Items acknowledged and deferred at milestone close, most recent first:

| Category | Item | Status | Deferred At | Milestone |
|----------|------|--------|-------------|-----------|
| *(none)* | | | | |

## Session Continuity

Last session: 2026-09-08
Stopped at: Roadmap tạo xong, chờ user duyệt (approval gate do orchestrator quản lý)
Resume file: None
