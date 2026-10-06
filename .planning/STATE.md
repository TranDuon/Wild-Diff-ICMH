---
gsd_state_version: "1.0"
current_phase: 2
current_phase_name: Mở rộng dữ liệu toàn corpus + hạ tầng eval + baseline
status: executing
stopped_at: Phase 2 code complete; awaiting Colab run of Wild_Diff_ICMH_Phase2_Eval.ipynb
last_updated: "2026-09-27T16:22:57.295Z"
last_activity: 2026-09-27
last_activity_desc: "Implemented Phase 1 metadata audit and Colab Bước 10 closeout; awaiting live artifact"
state_head: 5afb7b2f63ef0ec06398867d71689de138d1ff83
progress:
  total_phases: 6
  completed_phases: 1
  total_plans: 2
  completed_plans: 2
  percent: 0
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-09-08)

**Core value:** Chứng minh được rằng cơ chế chuyên biệt hoá (H2 ROI-weighted loss hoặc H3 domain-aware TGM) đóng góp vượt trên fine-tuning thuần (H1).
**Current focus:** Phase 2 — Mở rộng dữ liệu toàn corpus + hạ tầng eval + baseline

## Current Position

Phase: 2 (Mở rộng dữ liệu toàn corpus + hạ tầng eval + baseline) — EXECUTING
Plan: 6/6 code complete (05/10); chờ chạy `Wild_Diff_ICMH_Phase2_Eval.ipynb` trên Colab (kế hoạch: `.planning/phases/02-mo-rong-corpus-eval-baseline/02-PLAN.md`)
Status: Phase 1 hoàn thành 04/10/2026 (14/14, closeout thật trên L4). Còn 73,00 CU.
Nhánh: `phase2` (đổi tên từ `ver2` ngày 05/10) chứa bằng chứng Phase 1 và toàn bộ Phase 2; chưa merge vào `main`. Hai notebook trên nhánh này clone `phase2`; khi merge phải đổi `BRANCH` về `main`.
Last activity: 2026-10-06 - Completed quick task 261006-hxn: Phase 2 Eval notebook step 7 fix (estimated rate for autoregressive CompressAI, per-point scoring, auto-disconnect)

Kế hoạch duy nhất: khung GSD (`PROJECT.md`, `REQUIREMENTS.md`, `ROADMAP.md`, `phases/`). `CAMERA_TRAP_FINE_TUNING_PLAN.md` đã xoá ngày 04/10/2026; `COLAB_TRAINING.md` chỉ giữ lưu ý vận hành.

Progress: [██░░░░░░░░] 17%

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
- [2026-10-04]: Chỉ dùng một kế hoạch là khung GSD; xoá `CAMERA_TRAP_FINE_TUNING_PLAN.md`; `COLAB_TRAINING.md` chỉ giữ lưu ý vận hành.
- [2026-10-04]: Bài so sánh ngoài chưa chốt ⇒ đánh giá độc lập với bài so sánh (ROADMAP G-1..G-5): H1 chính là một model chung cho 20 site (thay cho model riêng từng site của quyết định 24/09); đo end-to-end ở độ phân giải gốc với độ phân giải xử lý mặc định cạnh dài 1024; λ = 2/8/32 + núm giải mã; tập dev từ validation, test chạy một lần; lưu toàn bộ bitstream + ảnh tái tạo; đa chỉ số; baseline phổ quát. Requirement mới: EVAL-11..16, H1-05, H1-06, ANLS-09.
- [2026-10-04]: Thêm hai cải tiến chỉ ở encode/decode, không train: ép xám ảnh đêm khi giải mã (H3-09) và chọn bitrate theo nội dung bằng MegaDetector ở encoder (H3-10), gộp thành cấu hình B5. Ảnh nền tham chiếu theo site (V2-09) và fine-tune riêng từng site lớn (V2-10) đưa vào v2. Tổng 70 requirement v1.
- [2026-10-04]: Phase 1 đóng với số đo thật: 5,53 s/optimizer step, ~1 CU chi phí cố định mỗi phiên Colab, 1,37 CU cho phiên closeout. Ngày/đêm theo giờ lệch ảnh xám IR ở 9/100 ảnh ⇒ Phase 2 xác định đêm theo ảnh xám.
- [2026-10-05]: Ngày/đêm định nghĩa theo **nguồn sáng** (đêm = camera tự chiếu sáng bằng flash hoặc IR), xác định từ chính file ảnh để dùng được cho mọi dataset: EXIF Flash → ảnh xám IR → độ cao mặt trời (chỉ khi có toạ độ) → độ sáng pixel. Kgalagadi dùng flash trắng ⇒ ảnh đêm là ảnh màu; nhãn theo giờ cũ sai 32/78 ảnh "đêm" ở mẫu 450 ảnh. H3-05/06/07/09 sửa theo `is_grayscale`.
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
| 3 | Thêm cell pull cố định vào notebook Colab | 2026-09-27 | db097b4 | — |
| 4 | Bước 7–10: run dir mới mỗi lần, ThroughputMonitor, CU đo thật vào closeout | 2026-10-04 | — | — |
| 261006-hxn | Eval Phase 2: CompressAI tự hồi quy dùng rate ước lượng, chấm từng điểm ngay khi nén, Bước 11 tự ngắt runtime | 2026-10-06 | 4288862 | [261006-hxn-fix-phase2-eval-step7-compressai](./quick/261006-hxn-fix-phase2-eval-step7-compressai/) |

## Deferred Items

Items acknowledged and deferred at milestone close, most recent first:

| Category | Item | Status | Deferred At | Milestone |
|----------|------|--------|-------------|-----------|
| *(none)* | | | | |

## Session Continuity

Last session: 2026-09-27T16:06:57.418Z
Stopped at: Phase 1 context gathered
Resume file: .planning/phases/01-khung-x-ng-end-to-end-t-i-thi-u-v-l-i-ch-n/01-CONTEXT.md
