---
gsd_state_version: "1.0"
current_phase: 3
current_phase_name: H1 — Fine-tuning thích ứng miền
status: planning
stopped_at: Phase 2 complete, ready to plan Phase 3
last_updated: "2026-10-08T06:49:51.823Z"
last_activity: 2026-10-08
last_activity_desc: Phase 2 complete, transitioned to Phase 3
state_head: 64d4a04788b75fa3a4c7cf6142e70042a4266f78
progress:
  total_phases: 6
  completed_phases: 2
  total_plans: 3
  completed_plans: 3
  percent: 33
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-10-08)

**Core value:** Chứng minh được rằng cơ chế chuyên biệt hoá (H2 ROI-weighted loss hoặc H3 domain-aware TGM) đóng góp vượt trên fine-tuning thuần (H1).
**Current focus:** Phase 3 — H1 — Fine-tuning thích ứng miền

## Current Position

Phase: 3 — H1 — Fine-tuning thích ứng miền
Plan: Not started
Status: Ready to plan
Nhánh: làm việc trên `main` (đã merge `phase2` ngày 07/10). Các notebook hiện vẫn đặt `BRANCH = 'phase2'` — `origin/phase2` đang chậm hơn `main`; notebook Phase 3 phải clone `main` (hoặc fast-forward `phase2`).
Last activity: 2026-10-08 — Phase 2 complete, transitioned to Phase 3

Kế hoạch duy nhất: khung GSD (`PROJECT.md`, `REQUIREMENTS.md`, `ROADMAP.md`, `phases/`). `CAMERA_TRAP_FINE_TUNING_PLAN.md` đã xoá ngày 04/10/2026; `COLAB_TRAINING.md` chỉ giữ lưu ý vận hành.

Progress: [███░░░░░░░] 33%

## Performance Metrics

**Velocity:**

- Total plans completed: 1
- Average duration: - min
- Total execution time: 0h

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 2 | 1 | - | - |

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
- [2026-10-07, **superseded 08/10**]: Từng dự kiến tự tái hiện Xie-SGC và chuyển việc đó từ Phase 2 sang đầu Phase 3. Quyết định này không còn hiệu lực; không triển khai EVAL-17 hay spec Xie-SGC.
- [2026-10-08]: Chốt Xie et al. 2025 chỉ là **literature comparison** (ANLS-09): dùng số công bố với cảnh báo khác dữ liệu/split/giao thức; không viết lại code, không chạy Xie-SGC/Xie-FT, không thêm lượt train hay CU. Tổng trở lại 70 requirement v1.
- [2026-10-08]: Phase 2 đóng (UAT 5/5, `02-VERIFICATION.md` passed). INFRA-02 (overlay mask) dời sang Phase 4, EVAL-04 (SpeciesNet) dời sang Phase 6 — nên làm sớm, song song Phase 3. Kết quả: `phases/02-*/02-RESULTS.md`; λ pilot H1 đề xuất = 2.
- [Roadmap]: Sửa số liệu Coverage trong REQUIREMENTS.md — file có 59 requirement v1 có ID cụ thể, không phải 52 như dòng tổng ghi lúc định nghĩa requirements.

### Pending Todos

None yet.

### Blockers/Concerns

- [Phase 2→4] INFRA-02: ảnh overlay mask phải có trước lượt train H2 đầu tiên.
- [Phase 2→6] EVAL-04: chưa có SpeciesNet; baseline phải decode lại để chấm định loài.
- [Phase 2] Tỉ lệ con vật "ảo" 52–85% của codec CompressAI ở bitrate thấp chưa kiểm bằng mắt / ở ngưỡng 0,5.
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
| 261007-l4e | Thêm baseline tái hiện Xie-SGC (Xie et al. 2025) vào kế hoạch: EVAL-17, Plan 02-07, ANLS-09 đã chốt, ghi chú H2 vs Xie-SGC ở Phase 4 | 2026-10-07 | 3c2fdb2 | [261007-l4e-add-xie-sgc-baseline-to-plan](./quick/261007-l4e-add-xie-sgc-baseline-to-plan/) |
| 261007-lm6 | Chuyển Xie-SGC (EVAL-17) từ Plan 02-07 sang đầu Phase 3 (spec `phases/03-…/XIE-SGC-SPEC.md`) | 2026-10-07 | — | [261007-lm6-move-xie-sgc-plan-to-phase3](./quick/261007-lm6-move-xie-sgc-plan-to-phase3/) |
| 261008-fvj | Bỏ baseline tái hiện Xie-SGC; chỉ đối chiếu số liệu công bố của bài Xie với ghi chú khác giao thức | 2026-10-08 | e1fbe92 | [261008-fvj-b-baseline-t-i-hi-n-xie-sgc-ch-i-chi-u-s](./quick/261008-fvj-b-baseline-t-i-hi-n-xie-sgc-ch-i-chi-u-s/) |
| 261008-inp | Đưa kết quả Phase 2 vào repo: `tools/summarize_phase2.py`, `02-RESULTS.md`, `results/` (CSV, bảng, RD) | 2026-10-08 | — | [261008-inp-phase2-results-summary](./quick/261008-inp-phase2-results-summary/) |
| 261008-j6m | Đóng Phase 2: dời INFRA-02 → Phase 4, EVAL-04 → Phase 6; `02-SUMMARY.md`, `02-VERIFICATION.md`, UAT 5/5 | 2026-10-08 | — | [261008-j6m-defer-infra02-eval04-close-phase2](./quick/261008-j6m-defer-infra02-eval04-close-phase2/) |

## Deferred Items

Items acknowledged and deferred at milestone close, most recent first:

| Category | Item | Status | Deferred At | Milestone |
|----------|------|--------|-------------|-----------|
| *(none)* | | | | |

## Session Continuity

Last session: 2026-09-27T16:06:57.418Z
Stopped at: Phase 2 complete, ready to plan Phase 3
Resume file: .planning/phases/01-khung-x-ng-end-to-end-t-i-thi-u-v-l-i-ch-n/01-CONTEXT.md
