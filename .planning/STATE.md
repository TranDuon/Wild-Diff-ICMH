---
gsd_state_version: "1.0"
current_phase: 3
current_phase_name: H1 — Fine-tuning thích ứng miền
status: executing
stopped_at: Fresh-L4 Recovery notebook H2 setup complete; live H2 overlay approval, train200 and eval30 pending
last_updated: "2026-10-14T15:06:43Z"
last_activity: 2026-10-14
last_activity_desc: Added explicit fresh-L4 H2 setup to the existing Recovery notebook; no H1 rerun or live H2 GPU run
state_head: 70a2956
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
Plan: H1 recovery200 completed; bounded H2 ROI-V1 alpha2/background1 pilot implemented via quick tasks 261014-otr and 261014-twb; full Phase 3 experimental completion pending
Status: H1 recovery200 checkpoint, loss/LR log and frozen-dev30 comparison confirmed from Drive. The existing Recovery notebook now includes fresh-L4 H2 setup, but H2 has NOT been run on GPU and Phase 3/4 are not complete.
Nhánh: làm việc trên `main` (đã merge `phase2` ngày 07/10). Notebook Phase 3 và recovery clone/pull `main`; không dùng notebook legacy còn trỏ nhánh `phase2` để lấy bản sửa mới.
Last activity: 2026-10-15 — Completed quick task 261015-h2d on branch phase3+4: H2 train now copies train/val originals from Drive on demand (checkpoints-only pre-staging); configs unchanged so the frozen H2 contract stays valid. Live H2 train200/eval30 still await user action.

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
- [2026-10-13]: Đã xác nhận H1 cũ tới 1.000 bước. Đối chiếu CPU B0 λ2/8/32 và H1 500/1000 trên đúng dev30 với protocol thực cho thấy đánh đổi dung lượng/chất lượng, không chứng minh fine-tuning thất bại toàn diện. Chuẩn bị một lượt thử mới từ trọng số tác giả λ2, optimizer mới, main LR1e-5/aux LR1e-3, tối đa 200 bước, đồng ý theo hash báo cáo; chưa chạy. Giữ nguyên run/config/ảnh cũ, không tự mở rộng hay train H2. Cặp H1/H2 tiếp theo phải cùng nguồn/LR/crop/số bước và chỉ khác ROI weighting.
- [2026-10-14]: H1 low-LR recovery200 đã chạy xong trên L4, lưu checkpoint/throughput/loss CSV và chấm đúng dev30; cùng 30 ảnh, H1 dùng 0,0053525 bit/pixel và mAP 0,40835 so với B0 λ2 0,0059902 và 0,39439. Đây chỉ là tín hiệu pilot, không phải chứng minh thắng ở cùng bitrate hay kết luận bài báo. H2 ROI-V1 alpha2/background1 dùng cùng nguồn tác giả λ2, optimizer mới, LR/crop/seed/200 bước, chỉ khác trọng số loss; giữ notebook Recovery và tài sản `/content` đang có. Chỉ chấm H2 trên dev30, lấy H1 cache làm đối chứng; không chạy Xie, quét alpha, mở rộng 3.000 bước hay tự đánh dấu phase hoàn tất.
- [Roadmap]: Sửa số liệu Coverage trong REQUIREMENTS.md — file có 59 requirement v1 có ID cụ thể, không phải 52 như dòng tổng ghi lúc định nghĩa requirements.

### Pending Todos

None yet.

### Blockers/Concerns

- [H1/H2] 30 ảnh là tín hiệu chọn hướng, không chứng minh thắng đường cong nén hay bài workshop. H1 recovery200 có cùng 6/20 khung dương tính bị bỏ sót như B0 và đã có log loss/LR; chưa xác định riêng tốc độ học là nguyên nhân. So H2 phải dùng đúng H1 recovery200, không lấy H1 cũ 500/1000 làm cặp đối chứng.
- [H2 GPU gate] Overlay mask từ transform thật, nguồn dữ liệu và hash xác nhận phải được xem/duyệt trong Colab trước lượt train H2 đầu tiên; kiểm thử CPU không thay xác nhận trực quan hay GPU. INFRA-02 chỉ hoàn tất khi gate này được xác nhận trên dữ liệu thật.
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
| 261008-m6w | Notebook Phase3 H1 pooled λ2 pilot500, dev30, pull cố định, resume/snapshot/cache; chưa confirm GPU | 2026-10-08 | 6c04451 | [261008-m6w-build-resource-efficient-phase3-h1-noteb](./quick/261008-m6w-build-resource-efficient-phase3-h1-noteb/) |
| 261013-fzr | Fix Timer mỗi phiên H1 resume, giữ trạng thái học; notebook stream log, 191 CPU tests +32 real-Lightning targeted tests; chưa chạy GPU bản sửa | 2026-10-13 | c05f6ad | [261013-fzr-fix-h1-resume-with-a-fresh-per-session-t](./quick/261013-fzr-fix-h1-resume-with-a-fresh-per-session-t/) |
| 261013-i7u | Audit B0/H1 cùng dev30 từ cache Drive; notebook CPU-first + run low-LR 200 bước riêng, hash consent, loss CSV; 253 CPU tests passed, chưa chạy trial GPU | 2026-10-13 | 687e660 | [261013-i7u-audit-cached-h1-rate-quality-tradeoff-an](./quick/261013-i7u-audit-cached-h1-rate-quality-tradeoff-an/) |
| 261014-otr | H2 ROI-V1 alpha2 pilot200 ghép cặp với H1 recovery200 đã xong; sửa notebook Recovery tại chỗ, preflight mask, H2-only dev30; GPU H2 chưa chạy | 2026-10-14 | c4a0373 | [261014-otr-build-one-paired-h2-roi-v1-alpha2-pilot2](./quick/261014-otr-build-one-paired-h2-roi-v1-alpha2-pilot2/) |
| 261014-twb | Chuẩn bị riêng H2 cho runtime L4 mới trong notebook Recovery; không chạy lại H1, H2 GPU vẫn chờ | 2026-10-14 | 70a2956 | [261014-twb-fix-existing-recovery-notebook-for-a-new](./quick/261014-twb-fix-existing-recovery-notebook-for-a-new/) |
| 261014-w2j | Lọc hộp MegaDetector có diện tích bằng 0 ở H2 ROI preflight; giữ kiểm tra dữ liệu hỏng | 2026-10-15 | dee980c | [261014-w2j-handle-zero-area-megadetector-boxes-in-h](./quick/261014-w2j-handle-zero-area-megadetector-boxes-in-h/) |
| 261015-h2d | H2 train chép ảnh khi cần (copy-on-demand) thay vì chép trước 8.690 ảnh train/val; nhánh phase3+4 | 2026-10-15 | ba3ecc6 | [261015-h2d-h2-train-copy-on-demand-image-staging](./quick/261015-h2d-h2-train-copy-on-demand-image-staging/) |

## Deferred Items

Items acknowledged and deferred at milestone close, most recent first:

| Category | Item | Status | Deferred At | Milestone |
|----------|------|--------|-------------|-----------|
| *(none)* | | | | |

## Session Continuity

Last session: 2026-10-14T15:06:43Z
Stopped at: Fresh-L4 H2 setup implemented in the same notebook; live H2 pilot still pending
Resume file: .planning/quick/261014-twb-fix-existing-recovery-notebook-for-a-new/261014-twb-SUMMARY.md
