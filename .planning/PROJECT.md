# Wild-Diff-ICMH

## What This Is

Chuyên biệt hoá **Diff-ICMH** — codec nén ảnh dựa trên generative prior của Stable Diffusion 2.1, phục vụ đồng thời máy và người — cho **miền ảnh bẫy ảnh động vật hoang dã (camera trap)**. Dự án fine-tune codec + control module từ checkpoint tác giả trên dữ liệu bẫy ảnh, rồi thêm hai cơ chế chuyên biệt hoá: **ROI-weighted loss** ưu tiên bit cho vùng có động vật, và **domain-aware Tag Guidance Module** khai thác metadata sẵn có của bẫy ảnh (ánh sáng ngày/đêm IR, sinh cảnh theo site, mùa từ EXIF).

Người dùng cuối là các nhà sinh thái học vận hành mạng bẫy ảnh trong khu bảo tồn: hàng triệu ảnh mỗi mùa, đường truyền vệ tinh/2G/LoRa vài chục kbps, cần ảnh vừa đủ cho pipeline máy (phát hiện — định loài — đếm) vừa đủ cho chuyên gia kiểm tra thủ công.

## Core Value

**Chứng minh được rằng cơ chế chuyên biệt hoá (H2 ROI-weighted loss hoặc H3 domain-aware TGM) đóng góp vượt trên fine-tuning thuần (H1)** — tức trả lời được câu hỏi phản biện "đóng góp của các bạn khác gì với việc fine-tune một model có sẵn trên dataset khác?". Nếu mọi thứ khác thất bại, điều này phải đứng vững.

## Requirements

### Validated

<!-- Đã có và đã xác minh trong repo trước khi bắt đầu -->

- ✓ Codebase Diff-ICMH/DiffEIC đầy đủ trong repo — `model/diffeic.py`, `ldm/`, `utils/`, `inference.py` — existing
- ✓ `train.py` là entrypoint PyTorch Lightning **hoạt động được**, không phải stub: `DataModule`, `instantiate_from_config`, resume checkpoint (có cờ `--resume_codec` load riêng codec), callbacks `ImageLogger` + `ModelCheckpoint`, `accumulate_grad_batches` — existing
- ✓ Hàm loss có sẵn điểm can thiệp: `p_losses` trong `model/diffeic.py` với `l_bpp`, `l_guide`, `l_simple`, `l_semantic_weight`, và CFG dropout `c_ucg_rate=0.1` — existing
- ✓ Checkpoint pretrained của tác giả đã chạy inference thành công — existing
- ✓ RAM++ tag extractor có trong `src/recognize-anything/` — existing
- ✓ Config training có cấu trúc rõ: `configs/train_diffeic.yaml` + `configs/model/diffeic.yaml` + `configs/dataset/lic_*.yaml` — existing
- ✓ Datalist dùng biến `$DATA_DIR` → portable sang Colab không cần sửa code — existing

### Active

<!-- Phạm vi hiện tại. Tất cả đều là giả thuyết cho tới khi ship và xác minh. -->

- [ ] Toàn bộ 10.222 ảnh Snapshot Kgalagadi không có người được tải, chuẩn hoá và chia 70/15/15 **theo sequence trong từng site**; Serengeti chỉ dùng đánh giá bổ sung
- [ ] ROI mask Kgalagadi sinh từ bbox MegaDetector; ghi rõ đây là pseudo-label và giữ detector/ngưỡng cố định giữa mọi ablation
- [ ] Eval harness đo detection (mAP tách theo AP_s/m/l), species classification, segmentation — mọi số ghi vào `results.jsonl` một schema duy nhất
- [ ] Baseline Diff-ICMH gốc trên miền bẫy ảnh, đo tách **ngày RGB / đêm IR**
- [ ] Đo trần vật lý của VAE (giới hạn tái tạo hoa văn) để định lại mục tiêu bằng số liệu, không tranh luận suông
- [ ] **H1** — Domain-adaptive fine-tuning codec + control module trên dữ liệu bẫy ảnh
- [ ] **H2** — ROI-weighted `L_dist` (và `L_sem` ở Encoder Layer 9), quét tham số α
- [ ] **H3** — Domain-aware TGM: vocab loài thu gọn (L1) + structured attributes từ metadata (L2)
- [ ] RD curve nhiều điểm bitrate cho các cấu hình gốc / +H1 / +full
- [ ] Bảng ablation cô lập đóng góp từng thành phần
- [ ] Bảng đo **cái giá của chuyên biệt hoá**: eval chéo trên miền tổng quát (COCO/Kodak) và tập camera-trap bổ sung (Serengeti)
- [ ] Báo cáo kỹ thuật + reproducibility package + slide/demo

### Out of Scope

- **H4 — Task-aware SC loss (DINOv2/BioCLIP)** — mâu thuẫn trực tiếp với luận điểm task-agnostic của paper gốc; cần ≥3 run × 20h mà ngân sách Colab Pro không có. Ghi vào Future Work.
- **Retrain từ đầu** — paper dùng 4× A100 cho 2 stage × 200K iterations; trên Colab Pro là bất khả thi. Chỉ fine-tune từ checkpoint tác giả.
- **Individual re-ID (định danh cá thể qua hoa văn)** — bị chặn bởi trần vật lý của VAE: chu kỳ sọc ngựa vằn còn 0,6–1 ô latent sau downsample 8×, **dưới ngưỡng Nyquist**. Sẽ chứng minh bằng thí nghiệm, không hứa hẹn.
- **Đánh bại VTM-18.2 về PSNR** — bản chất của generative codec là thua VTM ở PSNR tại cùng bpp. Baseline so sánh đúng là **chính Diff-ICMH gốc trên miền bẫy ảnh**.
- **Sửa trọng số SD 2.1 UNet hoặc VAE** — phá tan generative prior, mâu thuẫn thiết kế gốc.
- **Nhãn segmentation thật cấp pixel** — không tồn tại cho miền bẫy ảnh. Dùng pseudo-GT từ SAM và ghi rõ ảnh hưởng trong Limitations.

## Context

**Điểm xuất phát khác kế hoạch gốc ở ba chỗ quan trọng:**

1. **`train.py` đã hoạt động.** Kế hoạch 8 tuần trong `docs/ke-hoach-difficmh-wildlife-8-tuan.md` coi "train.py là stub" (rủi ro R1) là **rủi ro lớn nhất, xác suất cao, cổng sinh tử cuối Tuần 2**. Khảo sát repo cho thấy nó là entrypoint Lightning hoàn chỉnh với data module, resume checkpoint, callbacks và gradient accumulation. Rủi ro này coi như đã đóng, giải phóng 1–2 tuần khỏi lịch — chuyển sang phase dữ liệu và eval.

2. **Nút thắt đã đổi từ người-giờ sang compute.** Kế hoạch gốc giả định 1 GPU 24GB chạy liên tục 8 tuần = 1.344 giờ khả dụng, cần ~290h (hệ số an toàn 4,6×). Thực tế là **Colab Pro ~100 compute units/tháng**, tương đương ~20h L4 hoặc ~7,5h A100 mỗi tháng → **~50h L4 cho cả dự án**. Ngân sách thật nhỏ hơn giả định của kế hoạch **khoảng 5–6 lần**. Đây là ràng buộc định hình mọi quyết định về iterations, crop size, kích thước tập eval, và số DDIM steps.

3. **Chưa có một byte dữ liệu nào.** Thu thập + tiền xử lý + kiểm tra chất lượng là một phase thật, đứng trước mọi thứ khác — không phải giả định đã có sẵn.

**Đặc điểm miền làm chuyên biệt hoá có cơ sở information-theoretic:**

- Camera **đứng yên tuyệt đối** → hàng nghìn ảnh cùng một nền, mức dư thừa cấu trúc cao nhất trong mọi miền ảnh tự nhiên
- Phân phối ánh sáng **lưỡng thái**: ban đêm là IR đơn kênh nhân bản ra 3 kênh → hai kênh chroma mang gần như 0 thông tin, codec pretrain trên ảnh màu đang **lãng phí bit** trên toàn bộ ảnh đêm
- Động vật chiếm **<5–8% diện tích**, 60–75% ảnh là ảnh rỗng → hơn 90% ngân sách bit đang chi cho cỏ, lá, đất — thứ mà generative prior của SD sinh lại được gần như miễn phí. Đây là lập luận mạnh nhất cho H2.

**Lợi thế đặc thù cho H3:** phần lớn structured attribute ở tầng L2 là **metadata THẬT có sẵn ở phía encoder**, không phải đoán → 0 rủi ro "bẫy oracle". `illumination` suy ra không mất mát từ chính ảnh; `season`/giờ từ EXIF timestamp; `habitat` cố định theo site ID. Chỉ `occupancy` mới cần dự đoán và mới cần bảng oracle-vs-predicted.

**Hai chi tiết kỹ thuật quyết định thiết kế:** (i) `L_dist` tính trên latent VAE (downsample 8×) còn `L_sem` tính ở SD UNet middle block (downsample 64×) — nên ROI weighting có ý nghĩa trên `L_dist` nhưng gần như vô nghĩa trên `L_sem` ở middle block; (ii) SC loss gốc đã là tổng theo vị trí không gian `n`, nên thêm ROI weighting chỉ là thay `1/N` bằng `w_n / Σw_n` — sửa đổi 1 dòng, không tốn VRAM, không thêm tham số.

**Nguồn tham chiếu chính:** `docs/ke-hoach-difficmh-wildlife-8-tuan.md` (kế hoạch nghiên cứu chi tiết: 25 task, ma trận RACI, sổ rủi ro R0–R12, thứ tự hy sinh khi chậm tiến độ), `docs/NeurIPS-2025-diff-icmh-*.pdf` (paper gốc), `docs/Diff_ICMH__NeurIPS_2025___Camera_Ready_Appendix.pdf` (appendix camera-ready).

## Constraints

- **Compute**: Colab Pro ~100 units/tháng ≈ ~50h L4 hoặc ~19h A100 cho cả dự án — Đây là tài nguyên khan hiếm nhất, nhỏ hơn giả định của kế hoạch gốc ~5–6 lần. Mọi run phải vừa ngân sách này.
- **Compute**: Session Colab ngắt sau ~12h và có thể ngắt bất kỳ lúc nào — **Mọi training run bắt buộc phải resume được từ checkpoint**; không có run nào chạy liền mạch 15–24h như kế hoạch gốc giả định.
- **VRAM**: GPU Colab (L4 24GB / A100 40GB tuỳ phiên) — Crop 256² thay vì 512² của paper; batch nhỏ + gradient accumulation để giữ effective batch khớp paper.
- **Timeline**: 9–10 tuần từ 07/09/2026, chậm nhất **~16/11/2026** — Trễ là phải cắt phạm vi theo thứ tự hy sinh, không kéo lịch.
- **Nhân lực**: 2 người — TV-A (Data & Evaluation Lead), TV-B (Model & Training Lead). Giao diện giữa hai luồng là **hợp đồng thư mục**, không chờ nhau qua tin nhắn.
- **Lưu trữ**: Google Drive 2TB — đủ cho corpus 60K + checkpoint + ảnh decode; không phải ràng buộc.
- **Dữ liệu**: Không có nhãn segmentation cấp pixel cho miền bẫy ảnh — Phải dùng pseudo-GT sinh bằng SAM từ bbox, và ghi rõ ảnh hưởng trong Limitations.
- **Kiến trúc**: SD 2.1 UNet, VAE, RAM++ **đóng băng** — Chỉ được can thiệp vào codec `E_c`/`D_c`, control module, hàm loss, và nội dung điều kiện text `c`.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Phạm vi H1 + H2 + H3, cắt H4 | H4 mâu thuẫn luận điểm task-agnostic của paper gốc và cần ≥60h GPU mà ngân sách Colab không có | — Pending |
| Thu thập 60K ảnh làm corpus nhưng **train trên subset lấy mẫu** | Corpus lớn cho thống kê miền và split theo site đáng tin cậy; ngân sách compute không cho phép duyệt hết 60K trong một run | — Pending |
| Fine-tune từ checkpoint tác giả, không retrain | Paper dùng 4× A100 × 400K iters; đồng thời cô lập đúng biến số cần đo (hiệu ứng chuyên biệt hoá) thay vì trộn với biến động pretrain | — Pending |
| ROI-weight `L_sem` ở **Encoder Layer 9** (32×) thay vì Middle Block (64×) | Ở Middle Block một con linh dương chiếm <1 ô — ROI weighting vô nghĩa. Enc Layer 9 là lựa chọn tốt thứ hai trong ablation Fig 8(b) của paper | — Pending |
| Crop training có định hướng theo bbox (tỉ lệ ghi trong config) | Động vật chiếm <8% diện tích; crop ngẫu nhiên khiến model dành gần hết ngân sách học cho việc nén cỏ và lá | — Pending |
| Protocol eval: resize toàn khung 1024×768, không center-crop 768² | Center-crop ảnh bẫy ảnh sẽ loại bỏ động vật ở rìa khung | — Pending |
| Baseline so sánh là Diff-ICMH gốc, không phải VTM | Generative codec vốn thua VTM ở PSNR; so với chính nó trên miền hẹp mới là so sánh đúng và đủ | — Pending |
| Mọi kết quả báo cáo **tách ngày RGB / đêm IR** | Gộp hai chế độ ảnh khác nhau về bản chất sẽ che giấu chính hiệu ứng mà đề tài muốn đo | — Pending |
| `results.jsonl` là nguồn chân lý duy nhất, mọi figure sinh tự động | Chống hardcode số liệu và chống sai lệch giữa bảng trong báo cáo và biểu đồ | — Pending |
| Viết Method + Setup của báo cáo từ giữa dự án, không đợi phase cuối | Rủi ro R9 (dồn viết lách vào tuần cuối) có xác suất cao nhất trong sổ rủi ro; đây là biện pháp phòng ngừa rẻ nhất | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-09-07 after initialization*
