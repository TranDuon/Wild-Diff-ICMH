# Roadmap: Wild-Diff-ICMH

Kế hoạch triển khai từng bước theo tuần, người phụ trách, cổng kiểm tra và ma trận đối chứng: [CAMERA_TRAP_FINE_TUNING_PLAN.md](CAMERA_TRAP_FINE_TUNING_PLAN.md) (16/09/2026).

## Overview

Dự án đi từ một codebase Diff-ICMH đã chạy được nhưng **chưa có một byte dữ liệu bẫy ảnh nào**, tới một báo cáo kỹ thuật chứng minh H2 (ROI-weighted loss) hoặc H3 (domain-aware Tag Guidance Module) đóng góp vượt trên fine-tuning thuần (H1). Vì ngân sách compute thật (~50h L4-equivalent trên Colab Pro) nhỏ hơn giả định kế hoạch gốc 5-6 lần, roadmap này áp dụng **MVP theo chiều dọc (vertical slice)**: Phase 1 không xây "toàn bộ tầng dữ liệu rồi toàn bộ hạ tầng rồi mới thí nghiệm", mà chứng minh **một lát cắt mỏng chạy được đầu-cuối** — một tập ảnh mẫu nhỏ đi từ tải về, qua split an toàn theo site, qua một lượt train ngắn, tới decode và ghi một dòng `results.jsonl` — trước khi tiêu bất kỳ giờ GPU nghiêm túc nào. Các phase sau đó **mở rộng** lát cắt này về chiều rộng (toàn bộ 60K ảnh, harness eval đầy đủ) rồi chiều sâu (H1 → H2 → H3), tận dụng việc H3 tầng L1/L2 **không cần training** để chạy song song với các phase tốn GPU thay vì xếp hàng sau chúng. Hai rủi ro có thể làm dừng dự án — rò rỉ dữ liệu theo site/burst và cạn ngân sách compute-unit giữa chừng — được gắn thành cổng chặn cứng (hard gate) ngay từ Phase 1, không phải điều khoản ghi chú.

**Lưu ý về số lượng requirement:** `REQUIREMENTS.md` liệt kê **59 requirement v1** có ID cụ thể (không phải 52 như dòng tổng ghi ban đầu trong file — con số 52 là lỗi đếm từ bước định nghĩa requirements trước đó). Roadmap này map đủ cả 59, và dòng Coverage trong `REQUIREMENTS.md` đã được sửa lại cho khớp.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

- [ ] **Phase 1: Khung xương end-to-end tối thiểu + vá lỗi chặn** - Vá mọi lỗi chặn (dependency, resume, checkpoint) và chứng minh vòng lặp tải mẫu nhỏ → split an toàn → train ngắn → decode → ghi `results.jsonl` chạy được thật trên Colab, trước khi tiêu GPU nghiêm túc.
- [ ] **Phase 2: Mở rộng dữ liệu toàn corpus + hạ tầng eval + baseline** - Mở rộng sang toàn bộ ~60K ảnh, dựng dataset class không lỗi crop, dựng eval harness đầy đủ và chấm điểm baseline Diff-ICMH gốc — hàng đối chứng chịu lực của cả bảng ablation.
- [ ] **Phase 3: H1 — Fine-tuning thích ứng miền** - Fine-tune codec + control module trên dữ liệu bẫy ảnh, theo dõi rate collapse và catastrophic forgetting, bắt đầu viết Method/Setup của báo cáo.
- [ ] **Phase 4: H2 — ROI-weighted loss** - Gắn trọng số ROI vào `L_dist`/`L_sem` đúng vị trí không gian, quét α (V1/V2), theo dõi nghịch lý bỏ đói nền.
- [ ] **Phase 5: H3 — Domain-aware Tag Guidance Module (song song, không cần training)** - Vocab RAM++ thu gọn + metadata thật nối vào prompt lúc decode, lấp ô ablation gần như miễn phí, chạy song song với Phase 3/4.
- [ ] **Phase 6: Tổng hợp, ablation, cái giá chuyên biệt hoá & báo cáo** - RD curve, bảng ablation cộng dồn, bảng cái giá chuyên biệt hoá, báo cáo kỹ thuật hoàn chỉnh và gói tái lập.

## Phase Details

### Phase 1: Khung xương end-to-end tối thiểu + vá lỗi chặn
**Goal**: Toàn bộ vòng lặp — tải một tập ảnh mẫu nhỏ, split an toàn theo site/burst, một lượt train ngắn có thể resume an toàn, decode, và ghi một dòng kết quả — chạy được thật trên môi trường Colab hiện hành, với mọi lỗi chặn code đã được vá trước khi bất kỳ giờ GPU nghiêm túc nào bị tiêu.
**Mode:** mvp
**Depends on**: Nothing (first phase)
**Requirements**: PRE-01, PRE-02, PRE-03, PRE-04, PRE-05, PRE-06, DATA-02, DATA-03, DATA-05, INFRA-03, INFRA-04, INFRA-05, EVAL-01, EVAL-08
**Compute budget**: ≤8 CU (~4.5h tương đương L4) — smoke-test ~2K iterations + kill-and-resume test + vài lượt decode pyiqa. Đây là con số ước tính TRƯỚC đo lường; INFRA-03 trong chính phase này sẽ đo throughput thật và hiệu chỉnh lại toàn bộ ngân sách các phase sau.
**Success Criteria** (what must be TRUE):
  1. Cài đặt sạch dependencies (`numpy`, `lightning>=2.6`, `pyiqa`, `xformers` khớp ABI torch hiện hành) và `train.py` chạy hết smoke-test ~2K iterations trên bất kỳ tier GPU nào Colab cấp (L4 hoặc A100) mà không cần sửa code tay.
  2. Kill-and-resume test vượt qua: giết tiến trình training giữa chừng, resume qua `trainer.fit(ckpt_path=...)`, `global_step` và optimizer state tiếp tục đúng chỗ — không phải warm-start lại từ 0.
  3. Checkpoint ghi ra chỉ chứa trọng số trainable (đã loại SD 2.1/VAE/RAM++ đông cứng) với tần suất 500–1000 step, không phình đĩa.
  4. Trên một tập ảnh mẫu nhỏ đã tải, split theo site+burst được dựng và `split_check.py` chạy tự động trước job training/eval, chặn được một trường hợp rò rỉ site/burst cố ý tạo ra để kiểm thử; EXIF/location trên 100 ảnh đầu được xác minh còn dùng được (hoặc fallback JSON của LILA).
  5. Một dòng kết quả (bpp + PSNR/LPIPS từ `pyiqa` trên ảnh decode của checkpoint vừa smoke-train) được ghi vào `results.jsonl` đúng schema cố định, kèm compute-unit đã tiêu **đo được thật** — con số này dùng để hiệu chỉnh ngân sách mọi phase còn lại.
**Plans**: TBD
**UI hint**: no

### Phase 2: Mở rộng dữ liệu toàn corpus + hạ tầng eval + baseline
**Goal**: Mở rộng lát cắt Phase 1 từ mẫu nhỏ sang toàn bộ corpus ~60K ảnh, dựng xong dataset class không còn lỗi lệch crop ảnh/mask, và harness eval đầy đủ (detection, species, chất lượng ảnh) chấm điểm được checkpoint gốc chưa fine-tune làm baseline chịu lực cho toàn bộ bảng ablation.
**Mode:** mvp
**Depends on**: Phase 1
**Requirements**: DATA-01, DATA-04, DATA-06, DATA-07, INFRA-01, INFRA-02, EVAL-02, EVAL-03, EVAL-04, EVAL-05, EVAL-06, EVAL-07, EVAL-09, EVAL-10
**Compute budget**: ≤10 CU (~6h tương đương L4) — chủ yếu inference-only (SAM mask generation, decode baseline cho harness eval); không có training. Tải/chuẩn hoá/đóng gói corpus không tốn compute unit (CPU + I/O), chỉ tốn thời gian và Drive storage.
**Success Criteria** (what must be TRUE):
  1. Toàn bộ 10.222 ảnh Snapshot Kgalagadi không có người đã tải, chuẩn hoá, chia 70/15/15 theo sequence trong từng site và copy về đĩa local của session lúc khởi động — không đọc trực tiếp từng file nhỏ trên Drive; Serengeti chỉ dùng đánh giá bổ sung.
  2. Thống kê miền đo được trên corpus thật (tỉ lệ ảnh rỗng, tỉ lệ ngày RGB/đêm IR, phân bố diện tích bbox) và histogram độ phủ ROI mask sinh bằng SAM 2.1 cho tập train được xuất ra và hợp lý.
  3. `WildlifeLICDataset` trả về cặp ảnh+mask crop khớp tuyệt đối (stack ảnh+mask trước khi crop, không gọi `random_crop_arr` hai lần độc lập) và tỉ lệ crop thực sự chứa động vật được log ra mỗi run theo tỉ lệ định hướng bbox cấu hình được.
  4. Eval harness chạy MegaDetector V6 + SpeciesNet trong conda env cô lập giao tiếp qua JSON trên đĩa; baseline Diff-ICMH gốc (chưa fine-tune) được chấm mAP tách AP_small/medium/large, species accuracy 2 mức, tỉ lệ false positive trên ảnh rỗng, tỉ lệ ảo giác trên ảnh rỗng — tất cả tách ngày RGB/đêm IR, kèm bootstrap confidence interval.
**Plans**: TBD
**UI hint**: no

### Phase 3: H1 — Fine-tuning thích ứng miền
**Goal**: Codec `E_c`/`D_c` + control module fine-tune trên dữ liệu bẫy ảnh từ checkpoint tác giả, SD 2.1 và RAM++ giữ đóng băng, sinh ra model đầu tiên có vùng bpp chồng lấn đủ với baseline để tính BD-rate, và bắt đầu viết Method/Setup của báo cáo từ giữa dự án.
**Mode:** mvp
**Depends on**: Phase 2
**Requirements**: H1-01, H1-02, H1-03, H1-04, REPT-02
**Compute budget**: ≤35 CU (~20h tương đương L4) — khoản chi lớn nhất trong toàn dự án; chạy qua nhiều session Colab (mỗi session ≤~12h) nhờ resume đầy đủ trạng thái đã vá ở Phase 1.
**Success Criteria** (what must be TRUE):
  1. Codec `E_c`/`D_c` + control module fine-tune xong trên dữ liệu bẫy ảnh, SD 2.1/RAM++ vẫn đóng băng (xác minh được bằng diff trọng số trước/sau).
  2. Từng thành phần loss (`bpp`, `dist`, `diff`, `sem`) được log tách biệt trong suốt quá trình train, không có dấu hiệu rate collapse bị bỏ sót.
  3. Kiểm tra catastrophic forgetting định kỳ (decode Kodak/COCO ở các mốc cố định) không cho thấy generative prior bị phá trong suốt quá trình train.
  4. Vùng bpp của model fine-tune chồng lấn vùng bpp của baseline được xác nhận **tăng dần trong lúc train**, không đợi tới lúc dựng RD curve mới phát hiện không tính được BD-rate.
  5. Bản nháp phần Method và Setup của báo cáo kỹ thuật tồn tại vào cuối phase này (giữa dự án), không hoãn tới phase cuối.
**Plans**: TBD
**UI hint**: no

### Phase 4: H2 — ROI-weighted loss
**Goal**: `L_dist` và `L_sem` gắn trọng số theo ROI mask đúng vị trí không gian (Encoder Layer 9, không phải Middle Block), quét tham số α với hai biến thể V1/V2, và chứng minh trọng số thực sự chạm vào loss mà không gây nghịch lý bỏ đói nền.
**Mode:** mvp
**Depends on**: Phase 3 (fine-tune tiếp từ checkpoint H1). Có thể chạy **song song về lịch** với Phase 5 (H3 tầng L1/L2 không cần checkpoint H1/H2 để bắt đầu).
**Requirements**: H2-01, H2-02, H2-03, H2-04, H2-05, H2-06
**Compute budget**: ≤25 CU (~14.5h tương đương L4) — quét α gồm nhiều lượt fine-tune ngắn hơn tiếp nối từ checkpoint H1, không train lại từ đầu.
**Success Criteria** (what must be TRUE):
  1. `L_dist` gắn trọng số `W = 1 + (α−1)·M` ở không gian latent VAE (downsample 8×), và ảnh overlay mask được xuất ra khớp đúng vùng động vật sau crop khi kiểm bằng mắt.
  2. `L_sem` gắn trọng số ROI tại điểm áp dụng đã chuyển từ Middle Block (64×) sang Encoder Layer 9 (32×).
  3. `L_dist_roi` và `L_dist_bg` log tách biệt ngay từ run đầu tiên và hai đường tách nhau rõ rệt — bằng chứng mask thực sự vào loss chứ không phải no-op.
  4. Quét α có ít nhất một biến thể V1 (chỉ weight `L_dist`) và một biến thể V2 (weight cả `L_dist` và `L_sem`); tỉ lệ false positive trên ảnh rỗng được theo dõi riêng ở **từng giá trị α** để phát hiện sớm nghịch lý bỏ đói nền.
**Plans**: TBD
**UI hint**: no

### Phase 5: H3 — Domain-aware Tag Guidance Module (song song, không cần training)
**Goal**: Vocab RAM++ thu gọn (L1) và structured attribute từ metadata thật (L2) nối vào `TagGCM.extract_tag()` qua wrapper, không sửa RAM++ và không cần train lại — đây là đòn bẩy ngân sách lớn nhất dự án, chạy **song song về lịch** với Phase 3/4 thay vì xếp hàng sau chúng, và lấp ô ablation table gần như miễn phí bằng cách đổi prompt lúc decode trên checkpoint H1/H2 đã có.
**Mode:** mvp
**Depends on**: Phase 2 (cần checkpoint baseline + eval harness cho H3-01 đến H3-07 — các phần này **không phụ thuộc** Phase 3/4 và có thể bắt đầu ngay khi Phase 2 xong, chạy song song với toàn bộ Phase 3 và Phase 4). Riêng H3-08 (đổi prompt chồng lên checkpoint H1/H2) phụ thuộc thêm output của Phase 3 và Phase 4.
**Requirements**: H3-01, H3-02, H3-03, H3-04, H3-05, H3-06, H3-07, H3-08
**Compute budget**: ≤5 CU (~3h tương đương L4) — chỉ decode-only inference, **zero training**. Đây là đòn bẩy ngân sách quan trọng nhất: lấp nhiều ô ablation với chi phí gần bằng 0.
**Success Criteria** (what must be TRUE):
  1. Vocab RAM++ thu gọn còn ~256 tag liên quan động vật hoang dã, mã hoá 8 bits/tag thay vì 13, và overhead bit thực tế của tag đo được, đối chiếu với bitstream latent.
  2. Structured attribute từ metadata thật (`illumination` suy từ ảnh, `season`/giờ từ timestamp, `habitat` từ site ID) nối vào qua wrapper quanh `TagGCM.extract_tag()` mà không sửa RAM++; trường thật và trường phải dự đoán (`occupancy`) được tách bạch rõ trong báo cáo.
  3. Giả định "RAM++ trả tag định dạng thay vì nội dung trên ảnh IR" được kiểm chứng thực nghiệm trên 50–100 ảnh đêm **trước khi chốt thiết kế H3**, và phương sai chroma của ảnh decode đêm được xác nhận gần 0.
  4. Ít nhất 3 template prompt được so sánh có hệ thống (bắt buộc một template nêu rõ tính chất hồng ngoại đơn sắc), chạy chồng lên checkpoint H1/H2 để lấp ô bảng ablation với chi phí gần bằng 0.

  **Ghi chú song song:** H3-05 (kiểm chứng RAM++ trên ảnh IR) không phụ thuộc bất kỳ checkpoint nào — chỉ cần vài chục ảnh đêm và RAM++ có sẵn — nên có thể bắt đầu **ngay trong Phase 1**, sớm hơn cả phần còn lại của phase này.
**Plans**: TBD
**UI hint**: no

### Phase 6: Tổng hợp, ablation, cái giá chuyên biệt hoá & báo cáo
**Goal**: Mọi kết quả tích luỹ trong `results.jsonl` được tổng hợp thành RD curve, bảng ablation cộng dồn không còn ô trống, bảng cái giá chuyên biệt hoá trên cả hai trục, và một báo cáo kỹ thuật hoàn chỉnh kèm gói tái lập — trả lời trực tiếp câu hỏi Core Value của dự án.
**Mode:** mvp
**Depends on**: Phase 3, Phase 4, Phase 5 (cần checkpoint và kết quả từ cả ba luồng để lấp bảng ablation)
**Requirements**: ANLS-01, ANLS-02, ANLS-03, ANLS-04, ANLS-05, ANLS-06, ANLS-07, ANLS-08, REPT-01, REPT-03, REPT-04, REPT-05
**Compute budget**: ≤12 CU (~7h tương đương L4) — dự phòng cho điểm bitrate còn thiếu, re-run nếu vùng bpp không chồng lấn đủ để tính BD-rate, hoặc ô ablation chưa lấp được bằng đòn bẩy H3. Ưu tiên dùng lại kết quả đã có trong `results.jsonl` trước khi tiêu thêm compute unit.
**Success Criteria** (what must be TRUE):
  1. RD curve 3 điểm bitrate (λ_rate = 2, 8, 32) cho các cấu hình gốc/+H1/+full trên detection và species classification tồn tại, với BD-rate tính bằng `bjontegaard` (fit bậc hai, ghi rõ đây là fallback so với quy ước 4 điểm JVET).
  2. Bảng ablation cộng dồn (gốc → +H1 → +H1+H2 → +full) không còn ô trống, mọi ô có bootstrap CI; bảng cái giá chuyên biệt hoá tồn tại trên cả hai trục — miền tổng quát (COCO/Kodak) và camera-trap bổ sung (Serengeti).
  3. Trần vật lý của VAE chứng minh bằng lập luận Nyquist cộng thí nghiệm FFT trên ảnh gốc vs ảnh decode; anchor VTM/BPG chạy nền trên CPU không cạnh tranh compute unit với training; failure taxonomy có ví dụ ảnh kèm theo, không chỉ chọn ảnh đẹp; mọi figure sinh tự động từ `results.jsonl` qua `make_all_figures.py`.
  4. Báo cáo kỹ thuật hoàn chỉnh với Method, Setup, Experiments, Analysis và mục Limitations thực chất (trần VAE, rủi ro ảo giác, bản chất pseudo-GT của mask, sai lệch cố ý so với setup gốc); định vị tính mới cite đúng TLIC (DCC 2024) và arXiv:2604.01122 (Disney/ETH).
  5. Gói tái lập (env đã pin, `results.jsonl` cùng script sinh figure, manifest split, checkpoint chỉ chứa delta đã fine-tune, smoke test tối thiểu) và slide + notebook demo chạy được trên ít nhất 2 ảnh mẫu tồn tại.
**Plans**: TBD
**UI hint**: no

## Ngân sách compute-unit theo phase

Tổng ngân sách Colab Pro cho cả dự án: **~100 compute units (~50h tương đương L4, hoặc ~19h nếu buộc phải dùng A100)**. Các con số dưới đây là **trần ước tính TRƯỚC đo lường**, không phải cam kết cứng — Phase 1 (INFRA-03) đo throughput thật và hiệu chỉnh lại toàn bộ bảng này.

| Phase | Loại chi | Trần ước tính | Ghi chú |
|-------|----------|----------------|---------|
| 1 | Smoke-test + kill/resume | ≤8 CU (~4.5h L4) | Đo throughput thật để hiệu chỉnh các dòng bên dưới |
| 2 | Inference-only (SAM, baseline eval) | ≤10 CU (~6h L4) | Tải/chuẩn hoá corpus không tốn compute unit |
| 3 | Training (H1) | ≤35 CU (~20h L4) | Khoản chi lớn nhất, đa session, resume đầy đủ |
| 4 | Training (H2 alpha sweep) | ≤25 CU (~14.5h L4) | Tiếp nối từ checkpoint H1, không train lại từ đầu |
| 5 | Decode-only (H3 L1/L2) | ≤5 CU (~3h L4) | Zero training — đòn bẩy ngân sách lớn nhất dự án |
| 6 | Dự phòng tổng hợp | ≤12 CU (~7h L4) | Điểm bitrate/ô ablation còn thiếu, re-run nếu cần |
| **Tổng** | | **≤95 CU (~55h L4)** | **~5 CU dự phòng** so với trần 100 CU |

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
| 2 | Toàn bộ: tải corpus, SAM mask, dataset class, eval harness, baseline | Có thể rảnh tay — bắt đầu chuẩn bị config H1 song song |
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
| 1. Khung xương end-to-end tối thiểu + vá lỗi chặn | 0/TBD | Not started | - |
| 2. Mở rộng dữ liệu toàn corpus + hạ tầng eval + baseline | 0/TBD | Not started | - |
| 3. H1 — Fine-tuning thích ứng miền | 0/TBD | Not started | - |
| 4. H2 — ROI-weighted loss | 0/TBD | Not started | - |
| 5. H3 — Domain-aware Tag Guidance Module | 0/TBD | Not started | - |
| 6. Tổng hợp, ablation, cái giá chuyên biệt hoá & báo cáo | 0/TBD | Not started | - |
