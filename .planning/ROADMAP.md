# Roadmap: Wild-Diff-ICMH

**Đây là kế hoạch duy nhất của dự án** (khung GSD: `PROJECT.md` → `REQUIREMENTS.md` → `ROADMAP.md` → `STATE.md` → `phases/`). Từ 04/10/2026 đã xoá `CAMERA_TRAP_FINE_TUNING_PLAN.md` (16/09, giao thức Serengeti/CCT cũ); các phần còn giá trị của nó (ma trận cấu hình B0–B4) đã chuyển vào roadmap này. `COLAB_TRAINING.md` chỉ là hướng dẫn vận hành, không định nghĩa giao thức.

## Cập nhật 04/10/2026

- **Giao thức dữ liệu đã đổi từ 24/09:** dữ liệu chính là toàn bộ **10.222 ảnh Snapshot Kgalagadi** không có người, chia 70/15/15 theo sequence trong từng site (20 site). Serengeti chỉ dùng đánh giá bổ sung. ROI lấy từ bbox **MegaDetector** (pseudo-label), không dùng SAM 2.1. Các phase bên dưới đã được sửa theo giao thức này.
- **Phase 1: HOÀN THÀNH 04/10/2026** (nhánh `ver2` — nay đổi tên thành `phase2` —, commit `140830a`, L4). Một phiên Colab chạy liền Bước 1→10: train 20 step → resume 20→21 → decode 2 ảnh → `results.jsonl` → closeout. Số đo thật: **5,53 s/optimizer step** (0,69 s/batch × 8), cả phiên tốn **1,37 CU** (trong đó ~1 CU là cài đặt + chép ảnh). Còn lại **73,00 CU**. Chi tiết ở "Trạng thái" trong Phase 1.
- **Compute:** Colab Resources ngày 04/10/2026 hiển thị **còn 74,37 CU** (không có session đang chạy). Tổng trần gốc của Phase 2–6 là 87 CU > 74,37 CU ⇒ bảng ngân sách đã có thêm cột trần đề xuất tạm thời (xem mục ngân sách). Chưa xác nhận được 25,63 CU đã dùng có hoàn toàn thuộc dự án hay không, và gói Colab Pro có được gia hạn thêm CU hằng tháng trong thời gian dự án hay không.
- **Tiến độ:** hôm nay là tuần 5/~10 (hạn chót ~16/11/2026); dự án chậm khoảng 1–2 tuần so với lịch theo tuần ngày 16/09 (H1 lẽ ra bắt đầu 07/10).
- **Bài so sánh ngoài chưa chốt** (bài Xie 2025 là workshop, ít thông tin). Vì vậy từ 04/10 dự án theo nguyên tắc **"đánh giá độc lập với bài so sánh"**: train và giải mã một lần, lưu lại tất cả, chọn bài so sánh sau mà không train lại (ANLS-09).

## Cập nhật 08/10/2026

- **Bài so sánh ngoài đã chốt: Xie et al. 2025** (*Saliency-guided deployment-adaptive compression for wildlife camera traps*, CCAI@NeurIPS 2025, cùng Snapshot Kgalagadi). Dự án chỉ dùng **số liệu đã công bố** để đối chiếu trong phần Related Work/Discussion; không tự tái hiện code, không fine-tune Ballé và không chạy Xie-SGC/Xie-FT. Vì bài không công bố code/split, các số của hai bên được đặt trong bảng literature comparison có ghi rõ khác giao thức, không dùng để tuyên bố thắng trực tiếp.
- **Hệ quả cho H2 (Phase 4):** vẫn phải giải thích rõ điểm giống và khác về ý tưởng ưu tiên vùng động vật, nhưng bằng chứng thực nghiệm chịu lực của H2 là phép so sánh cùng giao thức với H1-control và B0/H1; bài Xie chỉ cung cấp bối cảnh ngoài.

### Quyết định đã chốt ngày 04/10/2026 (theo hướng tổng quát)

| # | Quyết định | Lý do | Requirement |
|---|---|---|---|
| G-1 | **H1 chính là một model chung** fine-tune trên train split của cả 20 site; model riêng từng site chỉ là mở rộng nếu còn CU | Rẻ hơn ~20 lần; đánh giá được trên tập test của bất kỳ bài nào; vẫn giữ split per-site đã đóng băng | H1-05, DATA-02 |
| G-2 | **Đo end-to-end ở độ phân giải gốc**; độ phân giải xử lý bên trong là tham số (mặc định cạnh dài 1024) chọn trên tập dev; train crop từ ảnh ở cùng tỉ lệ đó | Từ ảnh gốc tính lại được mọi kiểu đánh giá; tránh hết VRAM (~26 GiB ở ảnh gốc); tránh lệch tỉ lệ con vật giữa train và test | EVAL-11, INFRA-01 |
| G-3 | **Phủ rộng dải bitrate**: λ = 2, 8, 32 + núm chỉnh phía giải mã không tốn bit | Bài so sánh nào ở dải bitrate nào cũng có điểm chồng lên | ANLS-01, H1-06 |
| G-4 | **Split và tập test giữ nguyên**; mọi lựa chọn siêu tham số làm trên tập dev lấy từ validation; test chạy một lần | Đổi split hay chọn trên test sẽ làm hỏng mọi kết quả | DATA-02, EVAL-16 |
| G-5 | **Giải mã một lần, lưu hết** (bitstream, ảnh tái tạo gốc, thời gian); **đa chỉ số**; **baseline phổ quát** (JPEG, WebP, CompressAI zoo, B0, VTM/BPG); **ghi đủ thông tin giao thức** mỗi dòng kết quả | Có thể phân tích lại kết quả nội bộ mà không train/giải mã lại; bài Xie được đối chiếu riêng bằng số công bố | EVAL-12..15, ANLS-09 |

## Overview

Dự án đi từ một codebase Diff-ICMH đã chạy được nhưng **chưa có một byte dữ liệu bẫy ảnh nào**, tới một báo cáo kỹ thuật chứng minh H2 (ROI-weighted loss) hoặc H3 (domain-aware Tag Guidance Module) đóng góp vượt trên fine-tuning thuần (H1). Vì ngân sách compute thật (~50h L4-equivalent trên Colab Pro) nhỏ hơn giả định kế hoạch gốc 5-6 lần, roadmap này áp dụng **MVP theo chiều dọc (vertical slice)**: Phase 1 không xây "toàn bộ tầng dữ liệu rồi toàn bộ hạ tầng rồi mới thí nghiệm", mà chứng minh **một lát cắt mỏng chạy được đầu-cuối** — một tập ảnh mẫu nhỏ đi từ tải về, qua split an toàn theo site, qua một lượt train ngắn, tới decode và ghi một dòng `results.jsonl` — trước khi tiêu bất kỳ giờ GPU nghiêm túc nào. Các phase sau đó **mở rộng** lát cắt này về chiều rộng (toàn bộ 10.222 ảnh Kgalagadi, harness eval đầy đủ) rồi chiều sâu (H1 → H2 → H3), tận dụng việc H3 tầng L1/L2 **không cần training** để chạy song song với các phase tốn GPU thay vì xếp hàng sau chúng. Hai rủi ro có thể làm dừng dự án — rò rỉ dữ liệu theo site/burst và cạn ngân sách compute-unit giữa chừng — được gắn thành cổng chặn cứng (hard gate) ngay từ Phase 1, không phải điều khoản ghi chú.

**Lưu ý về số lượng requirement:** `REQUIREMENTS.md` hiện liệt kê và map đủ **70 requirement v1** có ID cụ thể. Con số 59 là bộ requirement trước cập nhật 04/10; yêu cầu tái hiện bài Xie đã được bỏ ngày 08/10 và không nằm trong tổng này.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

- [x] **Phase 1: Khung xương end-to-end tối thiểu + vá lỗi chặn** *(hoàn thành 04/10/2026)* - Vá mọi lỗi chặn (dependency, resume, checkpoint) và chứng minh vòng lặp tải mẫu nhỏ → split an toàn → train ngắn → decode → ghi `results.jsonl` chạy được thật trên Colab, trước khi tiêu GPU nghiêm túc.
- [ ] **Phase 2: Mở rộng dữ liệu toàn corpus + hạ tầng eval + baseline** - Mở rộng sang toàn bộ 10.222 ảnh Kgalagadi, hoàn thiện dataset (log tỉ lệ crop chứa động vật), dựng eval harness đầy đủ và chấm điểm baseline Diff-ICMH gốc — hàng đối chứng chịu lực của cả bảng ablation.
- [ ] **Phase 3: H1 — Fine-tuning thích ứng miền** - Fine-tune codec + control module trên dữ liệu bẫy ảnh, theo dõi rate collapse và catastrophic forgetting, bắt đầu viết Method/Setup của báo cáo.
- [ ] **Phase 4: H2 — ROI-weighted loss** - Gắn trọng số ROI vào `L_dist`/`L_sem` đúng vị trí không gian, quét α (V1/V2), theo dõi nghịch lý bỏ đói nền.
- [ ] **Phase 5: H3 — Domain-aware Tag Guidance Module (song song, không cần training)** - Vocab RAM++ thu gọn + metadata thật nối vào prompt lúc decode, lấp ô ablation gần như miễn phí, chạy song song với Phase 3/4.
- [ ] **Phase 6: Tổng hợp, ablation, cái giá chuyên biệt hoá & báo cáo** - RD curve, bảng ablation cộng dồn, bảng cái giá chuyên biệt hoá, báo cáo kỹ thuật hoàn chỉnh và gói tái lập.

## Ma trận cấu hình thí nghiệm

Chuyển từ kế hoạch theo tuần ngày 16/09 (đã xoá). Mọi bảng ablation và so sánh dùng đúng các mã này.

| Mã | Cấu hình | Phase | Config hiện có |
|---|---|---|---|
| B0 | Checkpoint Diff-ICMH gốc của tác giả, chưa fine-tune | 2 | `configs/model/diffeic.yaml` + checkpoint `CNscale1.0_*` |
| B1 | H1: fine-tune codec + control, loss gốc | 3 | `configs/train_kgalagadi_colab.yaml` |
| B1c | B1 train tiếp đúng số bước/batch/seed như B2, loss đều (α = 1) — đối chứng hiệu ứng train thêm | 4 | `configs/train_kgalagadi_h1_control.yaml` |
| B2 | B1 rồi H2 (ROI loss), cùng điểm xuất phát và ngân sách bước với B1c | 4 | `configs/train_kgalagadi_h2.yaml` |
| B3 | B1 + H3 ở đường encode/decode (không train) | 5 | `inference_partition.py --tag-vocabulary --domain-metadata` |
| B4 | B2 + H3 | 5 | như B3, trên checkpoint B2 |
| B5 | Cấu hình tốt nhất trong B1–B4 + chọn bitrate theo nội dung (H3-10) + ép xám ảnh đêm (H3-09) — chỉ thay đổi ở encode/decode, không train | 5 | cần thêm code (Phase 5) |

So sánh chính: **B2 với B1c** và **B3 với B1**, trên cùng ảnh, cùng bpp. B4 cho biết hiệu ứng cộng dồn. Chỉ tuyên bố cải thiện khi chênh lệch có khoảng tin cậy bootstrap theo site và không đánh đổi quá mức chất lượng ảnh rỗng/ảnh đêm.

## Phase Details

### Phase 1: Khung xương end-to-end tối thiểu + vá lỗi chặn
**Goal**: Toàn bộ vòng lặp — tải một tập ảnh mẫu nhỏ, split an toàn theo site/burst, một lượt train ngắn có thể resume an toàn, decode, và ghi một dòng kết quả — chạy được thật trên môi trường Colab hiện hành, với mọi lỗi chặn code đã được vá trước khi bất kỳ giờ GPU nghiêm túc nào bị tiêu.
**Mode:** mvp
**Depends on**: Nothing (first phase)
**Requirements**: PRE-01, PRE-02, PRE-03, PRE-04, PRE-05, PRE-06, DATA-02, DATA-03, DATA-05, INFRA-03, INFRA-04, INFRA-05, EVAL-01, EVAL-08
**Compute budget**: ≤8 CU (~4.5h tương đương L4) — smoke-test ~2K iterations + kill-and-resume test + vài lượt decode pyiqa. Đây là con số ước tính TRƯỚC đo lường; INFRA-03 trong chính phase này sẽ đo throughput thật và hiệu chỉnh lại toàn bộ ngân sách các phase sau.
**Success Criteria** (what must be TRUE):
  1. Cài đặt sạch dependencies (`numpy`, `lightning>=2.6`, `pyiqa`; `xformers` nếu có wheel khớp ABI torch hiện hành) và `train.py` chạy được trên tier L4 do người vận hành chọn thủ công, không tự đổi cấu hình thí nghiệm theo T4/A100.
  2. Kill-and-resume test vượt qua: giết tiến trình training giữa chừng, resume qua `trainer.fit(ckpt_path=...)`, `global_step` và optimizer state tiếp tục đúng chỗ — không phải warm-start lại từ 0.
  3. Checkpoint ghi ra chỉ chứa trọng số trainable (đã loại SD 2.1/VAE/RAM++ đông cứng) với tần suất 500–1000 step, không phình đĩa.
  4. Trên một tập ảnh mẫu nhỏ đã tải, split theo site+burst được dựng và `split_check.py` chạy tự động trước job training/eval, chặn được một trường hợp rò rỉ site/burst cố ý tạo ra để kiểm thử; EXIF/location trên 100 ảnh đầu được xác minh còn dùng được (hoặc fallback JSON của LILA).
  5. Một dòng kết quả (bpp + PSNR/LPIPS từ `pyiqa` trên ảnh decode của checkpoint vừa smoke-train) được ghi vào `results.jsonl` đúng schema cố định, kèm compute-unit đã tiêu **đo được thật** — con số này dùng để hiệu chỉnh ngân sách mọi phase còn lại.
**Plans**: 2 plans — metadata/leakage evidence; Phase 1 closeout and bounded calibration decision

**Trạng thái (04/10/2026): HOÀN THÀNH.** 2/2 plan; `01-VERIFICATION.md` = 14/14 sau khi có artifact closeout thật trên Drive (`results/phase1_closeout_KGA_A01.json`, `results/phase1_metadata_KGA_A01.json`, `runs/phase1_calib/A01/20261004-155451/throughput.json`).

| Tiêu chí | Trạng thái | Bằng chứng / còn thiếu |
|---|---|---|
| 1. Dependency + train trên L4 | Đạt | Bước 3 + smoke 20 step chạy thật 27/09 (commit 1915d1b → 246c0e0) |
| 2. Kill-and-resume | Đạt (phần giết giữa chừng chuyển sang Phase 3) | Resume full-state 20→21 trên run dir mới; bỏ qua checkpoint hỏng (fa329d0). Bằng chứng giết tiến trình *giữa chừng* sẽ lấy ở lượt H1 đầu tiên (checkpoint rolling step 50) |
| 3. Checkpoint compact | Đạt, lệch tần suất | Compact + contract v2 đã có. Tần suất thực tế là rolling mỗi 50 step + `last.ckpt` cuối (D-09), khác con số 500–1000 trong PRE-05 — cần sửa câu chữ PRE-05 hoặc đổi config |
| 4. Split gate + leak cố ý + EXIF 100 ảnh | Đạt | 10.222 dòng không rò rỉ; ảnh rò rỉ cố ý bị chặn; EXIF datetime 100/100, location 100/100. **9/100 ảnh** có nhãn ngày/đêm theo giờ lệch với ảnh xám IR → Phase 2 phải xác định đêm theo ảnh xám |
| 5. `results.jsonl` + CU đo thật | Đạt | Registry có dòng smoke (`smoke/non-report`); CU đo bằng chênh lệch "Available" = 1,37 CU/phiên; tốc độ ổn định 5,53 s/step (cách chia tổng thời gian cũ cho 20,2 s/step — phóng đại 3,7×) |

**Việc của Phase 1 (đã xong 04/10):**
1. **Sinh artifact closeout** `phase1_metadata_KGA_A01.json` và `phase1_closeout_KGA_A01.json`: một phiên Colab chạy liền Bước 1→10 (Bước 10 dùng biến trong bộ nhớ của Bước 7–9; runtime 27/09 đã mất nên không chạy riêng Bước 10 được).
2. **Đo throughput thật (INFRA-03):** *code xong 04/10.* `ThroughputMonitor` ghi `throughput.json` (median giây/batch × 8, bỏ khởi động và validation); Bước 7 luôn train vào run dir mới `runs/phase1_calib/<site>/<giờ chạy>` nên không còn lỗi dừng ngay khi run dir đã ở step 21; closeout dự báo 2K step bằng tốc độ ổn định.
3. **CU thật (INFRA-04):** *code xong 04/10.* Bước 1 nhận `CU_AVAILABLE_AT_START`, Bước 10 nhận `CU_AVAILABLE_NOW` (số "Available" trong Colab Resources; mốc 04/10: 74,37 CU); closeout dùng chênh lệch làm CU đã tiêu.
4. **(Khuyến nghị) Kill giữa chừng (PRE-06):** trong lượt calibration, sau khi có checkpoint rolling step 50, ngắt runtime; runtime mới chạy lại → log phải có `Auto-resume selected ...`.
5. Cập nhật `.planning/STATE.md`, `01-VERIFICATION.md` và bảng Progress khi có artifact; dùng số đo để hiệu chỉnh bảng ngân sách (recalibration checkpoint 1).

**UI hint**: no

### Phase 2: Mở rộng dữ liệu toàn corpus + hạ tầng eval + baseline
**Goal**: Mở rộng lát cắt Phase 1 từ site KGA:A01 sang toàn bộ 10.222 ảnh Kgalagadi, hoàn thiện dataset (ảnh/mask crop khớp, log tỉ lệ crop chứa động vật), và harness eval đầy đủ (detection, species, chất lượng ảnh) chấm điểm được checkpoint gốc chưa fine-tune làm baseline chịu lực cho toàn bộ bảng ablation.
**Mode:** mvp
**Depends on**: Phase 1
**Requirements**: DATA-01, DATA-04, DATA-06, DATA-07, INFRA-01, INFRA-02, EVAL-02, EVAL-03, EVAL-04, EVAL-05, EVAL-06, EVAL-07, EVAL-09, EVAL-10, EVAL-11, EVAL-12, EVAL-13, EVAL-14, EVAL-15, EVAL-16
**Compute budget**: ≤10 CU gốc, đề xuất tạm ≤8 CU (~4,5h L4) — chủ yếu inference-only (MegaDetector trên toàn corpus, decode B0 trên tập dev); không có training. JPEG/WebP/VTM chạy CPU; CompressAI zoo rất nhẹ. Tải/chuẩn hoá/đóng gói corpus không tốn compute unit (CPU + I/O), chỉ tốn thời gian và Drive storage.
**Success Criteria** (what must be TRUE):
  1. Toàn bộ 10.222 ảnh Snapshot Kgalagadi không có người đã tải, chuẩn hoá, chia 70/15/15 theo sequence trong từng site và copy về đĩa local của session lúc khởi động — không đọc trực tiếp từng file nhỏ trên Drive; Serengeti chỉ dùng đánh giá bổ sung.
  2. Thống kê miền đo được trên corpus thật (tỉ lệ ảnh rỗng, tỉ lệ ngày RGB/đêm IR, phân bố diện tích bbox) và ROI mask sinh từ bbox **MegaDetector** (detector/ngưỡng cố định, pseudo-label — DATA-06) kèm histogram độ phủ mask cho tập train được xuất ra và hợp lý. File detections JSON phủ **mọi** ảnh (kể cả `detections: []`) vì preflight H2 chặn nếu thiếu.
  3. `CameraTrapDataset` (`dataset/camera_trap_dataset.py`, đã có từ 24/09 — thay cho tên `WildlifeLICDataset` trong INFRA-01) trả về ảnh+mask crop khớp tuyệt đối (đã có test `test_detection_mask_and_image_transform_stay_aligned`); **còn thiếu**: log tỉ lệ crop thực sự chứa động vật mỗi run (INFRA-02) và ảnh overlay mask để kiểm bằng mắt.
  4. Eval harness chạy MegaDetector V6 + SpeciesNet trong conda env cô lập giao tiếp qua JSON trên đĩa; baseline Diff-ICMH gốc (chưa fine-tune) được chấm mAP tách AP_small/medium/large, species accuracy 2 mức, tỉ lệ false positive trên ảnh rỗng, tỉ lệ ảo giác trên ảnh rỗng — tất cả tách ngày RGB/đêm IR, kèm bootstrap confidence interval. Hiện `tools/evaluate_kgalagadi.py` mới có bpp/PSNR/SSIM/fg-SSIM/LPIPS.
  5. Giao thức end-to-end ở độ phân giải gốc (EVAL-11) chạy được: decode/evaluate nhận ảnh gốc, xử lý ở độ phân giải trong (mặc định cạnh dài 1024), trả ảnh cùng kích thước gốc; chế độ crop 256 ở giữa ảnh không còn là mặc định cho số báo cáo. Thời gian giải mã mỗi ảnh ở 1024 (và 512 để so) đã đo để định cỡ tập dev.
  6. Tập dev cố định từ validation (EVAL-16) đã đóng băng thành file danh sách; B0 ở λ = 2/8/32 và các baseline phổ quát (EVAL-14) đã giải mã trên tập dev, kho lưu trữ EVAL-12 có đủ bitstream + ảnh tái tạo gốc + thời gian, và `results.jsonl` có đủ bộ chỉ số EVAL-13 kèm thông tin giao thức EVAL-15.
  7. Vùng bitrate của B0 và các baseline được đặt chung một đồ thị để biết dải chồng lấn trước khi chọn λ cho H1.

**Việc code cần làm trong phase này:** chế độ resize cạnh dài dùng chung cho dataset / `inference_partition.py` / `tools/evaluate_kgalagadi.py` (`utils/image_geometry.py`); script baseline JPEG/WebP và CompressAI zoo; mở rộng evaluator (MS-SSIM, DISTS, byte/ảnh, compression ratio trên ảnh gốc, thời gian) và schema registry; config H1 cho **mọi site** (hiện `KGA_SITE_ID` mặc định `KGA:A01`) + tag cache RAM++ cho cả 20 site.
**Plans**: 6 plans — xem `.planning/phases/02-mo-rong-corpus-eval-baseline/02-PLAN.md` — code xong 05/10 cho cả 6 plan (02-01 giao thức độ phân giải gốc; 02-02 evaluator đa chỉ số + bootstrap + registry; 02-03 tập dev; 02-04 baseline JPEG/WebP/CompressAI; 02-05 MegaDetector + chỉ số máy + thống kê miền; 02-06 config H1 chung + log crop + notebook Phase 2) và nhãn ngày/đêm theo nguồn sáng. **Còn:** chạy notebook Phase 2 trên Colab, đóng băng tập dev theo nhãn mới, SpeciesNet (EVAL-04), VTM/BPG để Phase 6
**UI hint**: no

### Phase 3: H1 — Fine-tuning thích ứng miền
**Goal**: Codec `E_c`/`D_c` + control module fine-tune trên dữ liệu bẫy ảnh từ checkpoint tác giả, SD 2.1 và RAM++ giữ đóng băng, sinh ra model đầu tiên có vùng bpp chồng lấn đủ với baseline để tính BD-rate, và bắt đầu viết Method/Setup của báo cáo từ giữa dự án.
**Mode:** mvp
**Depends on**: Phase 2
**Requirements**: H1-01, H1-02, H1-03, H1-04, H1-05, H1-06, REPT-02
**Compute budget**: ≤35 CU gốc, đề xuất tạm ≤28 CU (~16h L4) — toàn bộ dành cho H1 và dự phòng vận hành; chạy qua nhiều session Colab (mỗi session ≤~12h) nhờ resume đầy đủ trạng thái đã vá ở Phase 1. Không có lượt train nào dành cho bài Xie.
**Hạ tầng đã có (24/09):** `configs/train_kgalagadi_colab.yaml` (bf16, accumulate 8), `tools/train_kgalagadi_sites.py` (chạy tuần tự theo site — chỉ dùng cho mở rộng per-site). Theo G-1, H1 chính là **một model chung** trên train split của cả 20 site, một model cho mỗi λ = 2/8/32 (thứ tự: một λ làm pilot rồi mới mở rộng). Số epoch/step định theo throughput đo ở Phase 1. Run root dùng `runs/h1_v2/` (checkpoint trong `runs/h1/` có entropy model khởi tạo ngẫu nhiên, không được resume).
**Success Criteria** (what must be TRUE):
  1. Codec `E_c`/`D_c` + control module fine-tune xong trên dữ liệu bẫy ảnh, SD 2.1/RAM++ vẫn đóng băng (xác minh được bằng diff trọng số trước/sau).
  2. Từng thành phần loss (`bpp`, `dist`, `diff`, `sem`) được log tách biệt trong suốt quá trình train, không có dấu hiệu rate collapse bị bỏ sót.
  3. Kiểm tra catastrophic forgetting định kỳ (decode Kodak/COCO ở các mốc cố định) không cho thấy generative prior bị phá trong suốt quá trình train.
  4. Vùng bpp của model fine-tune chồng lấn vùng bpp của baseline được xác nhận **tăng dần trong lúc train**, không đợi tới lúc dựng RD curve mới phát hiện không tính được BD-rate.
  5. Bản nháp phần Method và Setup của báo cáo kỹ thuật tồn tại vào cuối phase này (giữa dự án), không hoãn tới phase cuối.
  6. Núm chỉnh phía giải mã (H1-06) đã quét trên tập dev cho checkpoint H1, cấu hình tốt nhất được cố định trước khi chạy test; ảnh tái tạo và chỉ số được lưu theo EVAL-12/13.
  7. Kiểm soát overfit của model chung (H1-05): checkpoint được chọn theo validation, đường loss train/val được log và không tách nhau bất thường; nếu tách, dừng sớm hoặc giảm LR thay vì train thêm.
**Plans**: TBD
**UI hint**: no

### Phase 4: H2 — ROI-weighted loss
**Goal**: `L_dist` và `L_sem` gắn trọng số theo ROI mask đúng vị trí không gian (Encoder Layer 9, không phải Middle Block), quét tham số α với hai biến thể V1/V2, và chứng minh trọng số thực sự chạm vào loss mà không gây nghịch lý bỏ đói nền.
**Mode:** mvp
**Depends on**: Phase 3 (fine-tune tiếp từ checkpoint H1). Có thể chạy **song song về lịch** với Phase 5 (H3 tầng L1/L2 không cần checkpoint H1/H2 để bắt đầu).
**Requirements**: H2-01, H2-02, H2-03, H2-04, H2-05, H2-06
**Compute budget**: ≤25 CU gốc, đề xuất tạm ≤20 CU (~11,5h L4) — quét α gồm nhiều lượt fine-tune ngắn hơn tiếp nối từ checkpoint H1, không train lại từ đầu. Trần này phải bao gồm cả đối chứng H1-control.
**Hạ tầng đã có (24/09):** `configs/train_kgalagadi_h2.yaml` (V1: chỉ weight `l_guide`, `roi_weight` 4, `bbox_crop_probability` 0,75), `configs/train_kgalagadi_h1_control.yaml` (đối chứng cùng 2 epoch, loss đều), log `l_guide_roi`/`l_guide_bg`/`roi_fraction`. V2 cần bật `roi_semantic_enabled` + `l_semantic_weight > 0` + `sl_loc: enc_9` (hiện `l_semantic_weight = 0`).
**Success Criteria** (what must be TRUE):
  1. `L_dist` gắn trọng số `W = 1 + (α−1)·M` ở không gian latent VAE (downsample 8×), và ảnh overlay mask được xuất ra khớp đúng vùng động vật sau crop khi kiểm bằng mắt.
  2. `L_sem` gắn trọng số ROI tại điểm áp dụng đã chuyển từ Middle Block (64×) sang Encoder Layer 9 (32×).
  3. `L_dist_roi` và `L_dist_bg` log tách biệt ngay từ run đầu tiên và hai đường tách nhau rõ rệt — bằng chứng mask thực sự vào loss chứ không phải no-op.
  4. Quét α có ít nhất một biến thể V1 (chỉ weight `L_dist`) và một biến thể V2 (weight cả `L_dist` và `L_sem`); tỉ lệ false positive trên ảnh rỗng được theo dõi riêng ở **từng giá trị α** để phát hiện sớm nghịch lý bỏ đói nền.
  5. H2 được so với **H1-control** (từ cùng checkpoint H1, cùng số optimizer step, seed, LR, loss đều), không chỉ so với H1 trước khi train thêm — để tách đóng góp của trọng số ROI khỏi việc train thêm.
  6. Khác biệt giữa H2 và phương pháp SGC của Xie được giải thích trong Method/Discussion: cả hai ưu tiên vùng động vật, nhưng H2 áp dụng trong codec sinh ảnh và được kiểm chứng trực tiếp bằng H1-control trên cùng giao thức. Chỉ số công bố của Xie chỉ dùng làm bối cảnh tài liệu, không coi là đối chứng cùng tập.
**Plans**: TBD
**UI hint**: no

### Phase 5: H3 — Domain-aware Tag Guidance Module (song song, không cần training)
**Goal**: Vocab RAM++ thu gọn (L1) và structured attribute từ metadata thật (L2) nối vào `TagGCM.extract_tag()` qua wrapper, không sửa RAM++ và không cần train lại — đây là đòn bẩy ngân sách lớn nhất dự án, chạy **song song về lịch** với Phase 3/4 thay vì xếp hàng sau chúng, và lấp ô ablation table gần như miễn phí bằng cách đổi prompt lúc decode trên checkpoint H1/H2 đã có.
**Mode:** mvp
**Depends on**: Phase 2 (cần checkpoint baseline + eval harness cho H3-01 đến H3-07 — các phần này **không phụ thuộc** Phase 3/4 và có thể bắt đầu ngay khi Phase 2 xong, chạy song song với toàn bộ Phase 3 và Phase 4). Riêng H3-08 (đổi prompt chồng lên checkpoint H1/H2) phụ thuộc thêm output của Phase 3 và Phase 4.
**Requirements**: H3-01, H3-02, H3-03, H3-04, H3-05, H3-06, H3-07, H3-08, H3-09, H3-10
**Compute budget**: ≤5 CU gốc, đề xuất tạm ≤4 CU (H3-09 gần như miễn phí; H3-10 chỉ thêm một lượt MegaDetector ở encoder và dùng lại ảnh giải mã sẵn có của các λ) — chỉ decode-only inference, **zero training**. Đây là đòn bẩy ngân sách quan trọng nhất: lấp nhiều ô ablation với chi phí gần bằng 0.
**Hạ tầng đã có (24/09) và chỗ lệch so với requirement:**
- Từ vựng hiện tại `data/vocab/kgalagadi_wildlife_tags.txt` có **37 tag ⇒ 6 bit/tag**, khác H3-01 (~256 tag, 8 bit). Quy mô cuối chốt sau H3-05; code đã tự tính `ceil(log2(n))` bit/tag nên đổi file từ vựng là đủ.
- Metadata 1 byte/ảnh (bit ngày/đêm + 2 bit mùa Nam bán cầu) đã vào bitstream và được tính vào BPP (`--domain-metadata`). `illumination` hiện suy từ **giờ trong timestamp**, chưa suy từ chính ảnh như H3-02 yêu cầu; `habitat` chỉ dùng khi có bảng site đã kiểm chứng (`--habitat-map`).
- Tag được nối qua `CachedTagCodec`/`inference_partition.py` (đọc tag RAM++ đã cache) thay vì wrapper quanh `TagGCM.extract_tag()`; tinh thần H3-03 (không sửa RAM++, không train lại) vẫn giữ.
**Success Criteria** (what must be TRUE):
  1. Vocab RAM++ thu gọn (tối đa ~256 tag liên quan động vật hoang dã, ⌈log₂ n⌉ bit/tag thay vì 13; hiện 37 tag / 6 bit) được chốt sau H3-05, và overhead bit thực tế của tag đo được, đối chiếu với bitstream latent.
  2. Structured attribute từ metadata thật (`illumination` suy từ ảnh, `season`/giờ từ timestamp, `habitat` từ site ID) nối vào qua wrapper quanh `TagGCM.extract_tag()` mà không sửa RAM++; trường thật và trường phải dự đoán (`occupancy`) được tách bạch rõ trong báo cáo.
  3. Giả định "RAM++ trả tag định dạng thay vì nội dung trên ảnh IR" được kiểm chứng thực nghiệm trên 50–100 ảnh đêm **trước khi chốt thiết kế H3**, và phương sai chroma của ảnh decode đêm được xác nhận gần 0.
  4. Ít nhất 3 template prompt được so sánh có hệ thống (bắt buộc một template nêu rõ tính chất hồng ngoại đơn sắc), chạy chồng lên checkpoint H1/H2 để lấp ô bảng ablation với chi phí gần bằng 0.
  5. Ép xám ảnh đêm khi giải mã (H3-09) được đo riêng trên ảnh đêm, có/không ép xám, cùng bitstream và seed; bit ngày/đêm được tính vào BPP.
  6. Chọn bitrate theo nội dung (H3-10) có đường RD riêng (cấu hình B5), báo cáo kèm tỉ lệ ảnh có thú bị MegaDetector bỏ sót và ảnh hưởng của nó lên detection/species, tách ngày/đêm.

  **Ghi chú song song:** H3-05 (kiểm chứng RAM++ trên ảnh IR) không phụ thuộc bất kỳ checkpoint nào — chỉ cần vài chục ảnh đêm và RAM++ có sẵn — nên có thể bắt đầu **ngay trong Phase 1**, sớm hơn cả phần còn lại của phase này.
**Plans**: TBD
**UI hint**: no

### Phase 6: Tổng hợp, ablation, cái giá chuyên biệt hoá & báo cáo
**Goal**: Mọi kết quả tích luỹ trong `results.jsonl` được tổng hợp thành RD curve, bảng ablation cộng dồn không còn ô trống, bảng cái giá chuyên biệt hoá trên cả hai trục, và một báo cáo kỹ thuật hoàn chỉnh kèm gói tái lập — trả lời trực tiếp câu hỏi Core Value của dự án.
**Mode:** mvp
**Depends on**: Phase 3, Phase 4, Phase 5 (cần checkpoint và kết quả từ cả ba luồng để lấp bảng ablation)
**Requirements**: ANLS-01, ANLS-02, ANLS-03, ANLS-04, ANLS-05, ANLS-06, ANLS-07, ANLS-08, ANLS-09, REPT-01, REPT-03, REPT-04, REPT-05
**Compute budget**: ≤12 CU gốc, đề xuất tạm ≤8 CU (~4,5h L4) — dự phòng cho điểm bitrate còn thiếu, re-run nếu vùng bpp không chồng lấn đủ để tính BD-rate, hoặc ô ablation chưa lấp được bằng đòn bẩy H3. Ưu tiên dùng lại kết quả đã có trong `results.jsonl` trước khi tiêu thêm compute unit.
**Success Criteria** (what must be TRUE):
  1. RD curve 3 điểm bitrate (λ_rate = 2, 8, 32) cho các cấu hình gốc/+H1/+full trên detection và species classification tồn tại, với BD-rate tính bằng `bjontegaard` (fit bậc hai, ghi rõ đây là fallback so với quy ước 4 điểm JVET).
  2. Bảng ablation cộng dồn (gốc → +H1 → +H1+H2 → +full) không còn ô trống, mọi ô có bootstrap CI; bảng cái giá chuyên biệt hoá tồn tại trên cả hai trục — miền tổng quát (COCO/Kodak) và camera-trap bổ sung (Serengeti).
  3. Trần vật lý của VAE chứng minh bằng lập luận Nyquist cộng thí nghiệm FFT trên ảnh gốc vs ảnh decode; anchor VTM/BPG chạy nền trên CPU không cạnh tranh compute unit với training; failure taxonomy có ví dụ ảnh kèm theo, không chỉ chọn ảnh đẹp; mọi figure sinh tự động từ `results.jsonl` qua `make_all_figures.py`.
  4. Báo cáo kỹ thuật hoàn chỉnh với Method, Setup, Experiments, Analysis và mục Limitations thực chất (trần VAE, rủi ro ảo giác, bản chất pseudo-GT của mask, sai lệch cố ý so với setup gốc); định vị tính mới cite đúng TLIC (DCC 2024) và arXiv:2604.01122 (Disney/ETH).
  5. Gói tái lập (env đã pin, `results.jsonl` cùng script sinh figure, manifest split, checkpoint chỉ chứa delta đã fine-tune, smoke test tối thiểu) và slide + notebook demo chạy được trên ít nhất 2 ảnh mẫu tồn tại.
  6. Bài so sánh ngoài (ANLS-09: Xie et al. 2025) có bảng literature comparison riêng: ghi nguyên số liệu bài công bố trên các trục tương thích như SSIM và compression ratio, đồng thời ghi dữ liệu/split/độ phân giải/giao thức của mỗi bên. Không trộn các điểm khác giao thức thành một đường RD, không tuyên bố thắng trực tiếp và không viết code hay thêm lượt train để tái hiện bài Xie.
**Plans**: TBD
**UI hint**: no

## Ngân sách compute-unit theo phase

Tổng ngân sách Colab Pro cho cả dự án: **~100 compute units (~50h tương đương L4, hoặc ~19h nếu buộc phải dùng A100)**. Các con số dưới đây là **trần ước tính TRƯỚC đo lường**, không phải cam kết cứng — Phase 1 (INFRA-03) đo throughput thật và hiệu chỉnh lại toàn bộ bảng này.

| Phase | Loại chi | Trần gốc (trước đo) | Trần đề xuất tạm (04/10, còn 74,37 CU) | Ghi chú |
|-------|----------|----------------|----------------|---------|
| 1 | Smoke-test + kill/resume | ≤8 CU (~4.5h L4) | ≤2 CU cho phần còn lại | **Xong.** Phiên closeout đo thật 1,37 CU |
| 2 | Inference-only (MegaDetector, baseline eval) | ≤10 CU (~6h L4) | ≤8 CU | Tải/chuẩn hoá corpus không tốn compute unit |
| 3 | Training H1 | ≤35 CU (~20h L4) | ≤28 CU | Khoản chi lớn nhất, đa session, resume đầy đủ; không dành CU cho việc tái hiện bài Xie |
| 4 | Training (H2 alpha sweep + H1-control) | ≤25 CU (~14.5h L4) | ≤20 CU | Tiếp nối từ checkpoint H1, không train lại từ đầu |
| 5 | Decode-only (H3 L1/L2) | ≤5 CU (~3h L4) | ≤4 CU | Zero training — đòn bẩy ngân sách lớn nhất dự án |
| 6 | Dự phòng tổng hợp | ≤12 CU (~7h L4) | ≤8 CU | Điểm bitrate/ô ablation còn thiếu, re-run nếu cần |
| **Tổng** | | **≤95 CU (~55h L4)** | **≤70 CU + ~4 CU dự phòng** | Cột đề xuất co tỉ lệ để vừa 74,37 CU còn lại |

**Số đo thật sau Phase 1 (04/10/2026, L4):** 5,53 s/optimizer step (batch 1 × accumulate 8 = 8 ảnh/step) ⇒ 1 epoch train 7.191 ảnh ≈ 900 step ≈ 1,4 giờ ≈ 2,1 CU (quy đổi 1,54 CU/giờ); mỗi phiên Colab tốn thêm **~1 CU cố định** cho cài đặt + chép ảnh, nên gộp việc vào ít phiên dài thay vì nhiều phiên ngắn. Với trần Phase 3 ≤28 CU, trừ ~5–6 CU chi phí phiên, còn ~14 giờ train ≈ 9.000 step ⇒ **khoảng 3.000 step (~3 epoch) cho mỗi λ** nếu chạy cả 3 λ. Chưa tính thời gian validation trong lượt train dài; xác nhận lại sau phiên H1 đầu tiên. Số dư hiện tại: **73,00 CU**.

**Ghi chú 04/10/2026:** cột "đề xuất tạm" chỉ là phân bổ lại cho vừa số dư hiện có, **chưa** dựa trên throughput đo thật. Thay bằng số đo sau khi Phase 1 có artifact closeout. Nếu gói Colab Pro được gia hạn thêm CU trong thời gian dự án, có thể quay lại cột trần gốc. CU đã tiêu thật = chênh lệch "Available" trong Colab Resources trước/sau mỗi phiên; ghi vào `cu_estimate` của `results.jsonl`.

## Điểm hiệu chỉnh lại ngân sách (recalibration checkpoints)

Vì tier GPU Colab được cấp không xác định trước phiên, ngân sách phải hiệu chỉnh lại ở **mọi ranh giới phase**, không chỉ một lần ở Phase 1:

1. **Sau Phase 1** — thay toàn bộ bảng ước tính ở trên bằng số đo thật từ smoke-test (INFRA-03); nếu lệch >20% so với giả định, viết lại trần các phase 2-6 trước khi vào Phase 2.
2. **Trong Phase 3 (H1)** — sau mỗi session Colab bị ngắt và resume, cộng dồn CU đã tiêu vào `results.jsonl` (INFRA-04) và đối chiếu với trần còn lại; nếu vượt tốc độ đốt dự kiến, áp dụng thứ tự hy sinh (xem bên dưới) trước khi tiếp tục.
3. **Sau Phase 3, trước khi định cỡ Phase 4** — số CU thực đã tiêu cho H1 quyết định số điểm α khả thi trong quét H2 (nhiều α hơn nếu còn dư, ít hơn nếu H1 vượt trần).
4. **Sau Phase 4, trước khi khoá phạm vi Phase 6** — nếu ngân sách còn lại không đủ cho cả 3 điểm bitrate + đủ ô ablation, áp dụng thứ tự hy sinh ngay tại đây, không phát hiện ở tuần cuối.

## Song song hai luồng nhân lực (TV-A / TV-B)

Giao diện giữa hai luồng là **hợp đồng thư mục** (file/schema cố định), không chờ nhau qua tin nhắn.

| Phase | TV-A (Data & Evaluation Lead) | TV-B (Model & Training Lead) |
|-------|-------------------------------|-------------------------------|
| 1 | Split an toàn (DATA-02/03/05), schema `results.jsonl` (EVAL-01) | Vá dependency + resume + checkpoint (PRE-01..06), smoke-test (INFRA-03/04/05), eval ảnh decode (EVAL-08) |
| 2 | Tải corpus, MegaDetector bbox → ROI mask, thống kê miền, eval harness (MegaDetector + SpeciesNet), tập dev, baseline phổ quát JPEG/WebP/CompressAI zoo/VTM, mở rộng evaluator + registry | Chế độ end-to-end ở độ phân giải gốc (EVAL-11), decode B0, log tỉ lệ crop chứa động vật + overlay mask (INFRA-02), config H1 model chung + tag cache 20 site |
| liên tục | Duy trì bảng trích dẫn và số liệu công bố của bài Xie cho literature comparison (ANLS-09); không tái hiện code | — |
| 3 | Vận hành eval harness theo dõi checkpoint H1 qua `STATUS.json`/poll nhẹ | Chạy fine-tuning H1, theo dõi loss/forgetting |
| 4 | Đo false-positive theo từng α, dựng overlay mask kiểm tra bằng mắt | Chạy quét α (H2), theo dõi `L_dist_roi`/`L_dist_bg` |
| **5 (song song với 3+4)** | **Có thể đảm nhận H3 hoàn toàn** — vocab, metadata, kiểm chứng IR — vì không cần GPU training | Không bị chặn: có thể hỗ trợ H3-08 (đổi prompt trên checkpoint) khi H1/H2 ra checkpoint mới |
| 6 | Bảng ablation, bảng cái giá chuyên biệt hoá, figure tự động, anchor VTM/BPG (CPU) | Method/Setup (bắt đầu từ Phase 3), Limitations, reproducibility package |

**Đòn bẩy song song quan trọng nhất:** Phase 5 (H3 L1/L2) không cần GPU training nên có thể chạy trọn vẹn trong khi Phase 3 và Phase 4 đang tiêu GPU — không xếp hàng sau chúng. H3-05 cụ thể có thể bắt đầu ngay từ Phase 1.

## Thứ tự hy sinh khi vượt ngân sách hoặc trễ tiến độ

Pre-committed **trước khi dự án bắt đầu**, để không phát hiện bức tường ngân sách giữa chừng. Áp dụng theo đúng thứ tự — không cắt tầng dưới trước khi cắt hết tầng trên:

1. **Cắt trước — điểm bitrate vượt mức tối thiểu**: Giữ tối thiểu đủ điểm để BD-rate tính được bằng fallback bậc hai (ANLS-02); cắt các điểm λ_rate bổ sung ngoài 3 điểm tối thiểu (2, 8, 32) trước tiên.
2. **Cắt tiếp theo — ô ablation không thiết yếu**: Giữ V1 (H2-05) và ít nhất một giá trị α tốt nhất; cắt các giá trị α còn lại trong quét trước khi cắt bất kỳ hàng nào trong bảng ablation cộng dồn chính (ANLS-03).
3. **Cắt sau cùng — phân tích tuỳ chọn**: Cắt các mở rộng không nằm trong danh sách requirement bắt buộc (ví dụ thêm template prompt H3 ngoài 3 template tối thiểu, FID mẫu lớn hơn) trước khi động tới bất kỳ hàng/cột nào của hai bảng chịu lực (ablation cộng dồn và cái giá chuyên biệt hoá) — hai bảng này trả lời trực tiếp Core Value và không được cắt.

**Không bao giờ cắt:** baseline (EVAL-10), gate split_check.py (DATA-03), bảng ablation cộng dồn (ANLS-03), bảng cái giá chuyên biệt hoá (ANLS-04), mục Limitations (REPT-01). Đây là năm thứ trả lời trực tiếp câu hỏi phản biện "đóng góp của các bạn khác gì với fine-tune thuần?".

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → (5 song song với 3 và 4) → 6

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Khung xương end-to-end tối thiểu + vá lỗi chặn | 2/2 | Complete (14/14) | 2026-10-04 |
| 2. Mở rộng dữ liệu toàn corpus + hạ tầng eval + baseline | 6/6 code | Chờ chạy notebook Phase 2 trên Colab | - |
| 3. H1 — Fine-tuning thích ứng miền | 0/TBD | Not started (config + launcher theo site đã có) | - |
| 4. H2 — ROI-weighted loss | 0/TBD | Not started (code V1 + config H1-control đã có, chưa train) | - |
| 5. H3 — Domain-aware Tag Guidance Module | 0/TBD | Not started (bitstream tag/metadata đã có, chưa decode thật) | - |
| 6. Tổng hợp, ablation, cái giá chuyên biệt hoá & báo cáo | 0/TBD | Not started | - |

### Lịch đề xuất cho phần còn lại (04/10 → 16/11/2026)

Lịch tạm, cần cả hai thành viên xác nhận; trễ thì áp dụng thứ tự hy sinh ở trên, không kéo lịch.

| Khoảng thời gian | Việc |
|---|---|
| 05–07/10 | Đóng Phase 1: calibration + closeout + cập nhật ngân sách; đo thời gian giải mã ở 1024/512 |
| 07–14/10 | Phase 2: chế độ end-to-end, tập dev, MegaDetector toàn corpus, thống kê miền, harness eval, B0 + baseline phổ quát. H3-05 (RAM++ trên ảnh đêm) chạy song song |
| 13–27/10 | Phase 3: H1 theo phạm vi đã chốt; nháp Method/Setup |
| 24/10–06/11 | Phase 4: H2 V1 + H1-control (+ V2 nếu còn ngân sách) |
| 14/10–06/11 | Phase 5: H3 decode-only song song với Phase 3–4 |
| trước 03/11 | Chốt bảng số liệu công bố và giới hạn so sánh với bài Xie cùng thầy (ANLS-09) |
| 03–16/11 | Phase 6: chạy test một lần, RD curve, ablation, cái giá chuyên biệt hoá, đồ thị so với bài đã chốt, báo cáo, gói tái lập |
