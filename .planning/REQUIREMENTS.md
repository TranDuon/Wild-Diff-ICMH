# Requirements: Wild-Diff-ICMH

**Defined:** 2026-09-08
**Core Value:** Chứng minh được rằng cơ chế chuyên biệt hoá (H2 ROI-weighted loss hoặc H3 domain-aware TGM) đóng góp vượt trên fine-tuning thuần (H1)

## v1 Requirements

Requirements cho milestone đầu (nghiệm thu đề tài). Mỗi requirement map tới đúng một phase trong roadmap.

### Pre-flight — Sửa lỗi chặn trước khi tiêu bất kỳ GPU-hour nào

- [ ] **PRE-01**: Dependency của repo cài được sạch trên runtime Colab hiện hành — `numpy`, `lightning>=2.6` (thay `pytorch_lightning==1.5.0`), `pyiqa` (thay `lpips==0.1.4`), `xformers` khớp ABI của torch hiện tại
- [ ] **PRE-02**: `train.py` chạy được trên Lightning 2.x — xoá import `LightningCLI` chết, đổi `accelerator: ddp` thành `accelerator: gpu, devices: 1` trong `configs/train_diffeic.yaml`
- [ ] **PRE-03**: Training resume được **đầy đủ trạng thái** qua `trainer.fit(ckpt_path=...)` — giữ nguyên optimizer state, vị trí LR schedule và `global_step` sau khi Colab ngắt session (thay cho đường `load_state_dict` weights-only hiện tại ở `train.py:73-88`)
- [ ] **PRE-04**: Checkpoint chỉ lưu trọng số trainable — override `on_save_checkpoint`/`on_load_checkpoint` trong `DiffEIC` để loại SD 2.1 UNet, VAE và RAM++ đóng băng khỏi file checkpoint
- [ ] **PRE-05**: Tần suất checkpoint phù hợp session Colab — chọn theo thời gian thực đo của một step (hiện: rolling mỗi 50 optimizer step + `last.ckpt` cuối mỗi `trainer.fit`), chỉ giữ `last.ckpt` + `best.ckpt` để không phình đĩa
- [ ] **PRE-06**: Kill-and-resume test vượt qua — giết tiến trình training giữa chừng, resume, và xác nhận `global_step` cùng optimizer state tiếp tục đúng chỗ (không phải warm-start từ 0)

### Data — Nền tảng dữ liệu bẫy ảnh

- [ ] **DATA-01**: Toàn bộ 10.222 ảnh Snapshot Kgalagadi không có nhãn người được tải và chuẩn hoá; Snapshot Serengeti chỉ dùng đánh giá bổ sung
- [ ] **DATA-02**: Mỗi site Kgalagadi được chia train/val/test 70/15/15 theo **sequence/burst**, không phải theo ảnh. Split đã đóng băng (`kgalagadi_site_split.jsonl` + SHA-256) và không đổi trong suốt dự án. (Từ 04/10: H1 chính là một model chung cho mọi site — xem H1-05; model riêng từng site chỉ là mở rộng tuỳ chọn)
- [ ] **DATA-03**: `split_check.py` assert không có sequence/burst nào xuất hiện ở hai split và **chạy tự động trước mọi job training/eval** đọc file split (site được phép xuất hiện ở cả ba split theo protocol per-site)
- [ ] **DATA-04**: Thống kê miền đo được trên chính corpus của nhóm — tỉ lệ ảnh rỗng, tỉ lệ ngày RGB vs đêm IR, phân bố diện tích bbox so với khung hình
- [ ] **DATA-05**: Xác minh metadata `datetime` và `location` còn dùng được — kiểm tra EXIF trên 100 ảnh đầu, fallback sang trường JSON của LILA nếu EXIF bị strip
- [ ] **DATA-06**: ROI mask Kgalagadi sinh từ bbox MegaDetector với detector/ngưỡng cố định, kèm histogram độ phủ mask; ghi rõ đây là pseudo-label
- [ ] **DATA-07**: Corpus đóng gói dạng shard (tar/webdataset) và copy về đĩa local của session lúc khởi động, không đọc trực tiếp từng file nhỏ trên Drive

### Infra — Hạ tầng training và quản lý ngân sách compute

- [ ] **INFRA-01**: `CameraTrapDataset` (`dataset/camera_trap_dataset.py`) trả về ảnh và ROI mask **crop khớp nhau tuyệt đối** — ảnh và mask cùng một phép resize/crop/lật, không crop hai lần độc lập; crop lấy từ ảnh đã đưa về đúng độ phân giải xử lý của EVAL-11
- [ ] **INFRA-02**: Lấy mẫu crop có định hướng theo bbox với tỉ lệ cấu hình được, và tỉ lệ crop thực sự chứa động vật được log ra mỗi run
- [ ] **INFRA-03**: Throughput và mức đốt compute unit **đo thật** bằng smoke-test ~2K iterations, dùng để hiệu chỉnh lại ngân sách của mọi phase còn lại
- [ ] **INFRA-04**: Ngân sách compute unit được theo dõi liên tục trong `results.jsonl` và đối chiếu lại ở mỗi ranh giới phase, vì tier GPU Colab được cấp không xác định trước
- [ ] **INFRA-05**: Notebook dùng cấu hình L4 24GB đã kiểm chứng; người vận hành chọn L4 thủ công và log ghi rõ GPU thực tế, không tự đổi batch/precision theo T4/A100 làm thay đổi thí nghiệm

### Eval — Harness đánh giá và nguồn chân lý số liệu

- [ ] **EVAL-01**: `results.jsonl` là nguồn chân lý duy nhất, schema cố định gồm `exp_id, dataset, lambda_rate, illumination, ddim_steps, metric, value, n_images, cu_estimate, git_commit, date`
- [ ] **EVAL-02**: Eval harness chạy MegaDetector V6 và SpeciesNet trong **conda env cô lập**, giao tiếp qua JSON trên đĩa, không import chung
- [ ] **EVAL-03**: Detection mAP báo cáo **tách theo AP_small / AP_medium / AP_large**, không chỉ mAP tổng
- [ ] **EVAL-04**: Species accuracy báo cáo ở **cả mức loài và mức nhóm/taxonomic fallback**
- [ ] **EVAL-05**: Mọi metric chính báo cáo **tách ngày RGB / đêm IR**, không bao giờ gộp
- [ ] **EVAL-06**: Tỉ lệ false positive trên ảnh rỗng được theo dõi như metric hạng nhất, dẫn xuất từ cùng một lượt inference MegaDetector đã chạy cho mAP
- [ ] **EVAL-07**: Tỉ lệ ảo giác trên ảnh rỗng — đo tần suất decoder sinh ra động vật không có trong ảnh gốc
- [ ] **EVAL-08**: Metric chất lượng ảnh qua `pyiqa` — PSNR, MS-SSIM, LPIPS, DISTS, FID
- [ ] **EVAL-09**: Bootstrap confidence interval tính cho **mọi con số headline**, không chỉ bảng AP của H2
- [ ] **EVAL-10**: Baseline Diff-ICMH gốc (checkpoint tác giả, chưa fine-tune) được chấm điểm trên miền bẫy ảnh — đây là hàng đối chứng chịu lực của toàn bộ bảng ablation

### Eval-G — Đánh giá độc lập với bài so sánh (thêm 04/10/2026)

Mục tiêu: một bộ checkpoint và **một lần giải mã** dùng được cho bất kỳ bài so sánh nào được chốt sau, không phải train hay giải mã lại.

- [ ] **EVAL-11**: Giao thức đo **end-to-end ở độ phân giải gốc**: đầu vào là ảnh gốc (2592×2000), đầu ra là ảnh cùng kích thước, mọi chỉ số tính so với ảnh gốc. Độ phân giải xử lý bên trong (thu nhỏ trước khi nén, phóng to sau khi giải mã) là tham số chọn trên tập dev, mặc định cạnh dài 1024. Crop 256 ở giữa ảnh chỉ dùng cho smoke test, không dùng cho số báo cáo
- [ ] **EVAL-12**: Mọi lượt giải mã đều lưu trên Drive: bitstream, ảnh tái tạo ở độ phân giải gốc, thời gian encode/decode từng ảnh. Chỉ số mới được tính lại từ kho lưu trữ này, không giải mã lại
- [ ] **EVAL-13**: Mỗi lượt đánh giá tính đủ bộ chỉ số: PSNR, SSIM, MS-SSIM, LPIPS, DISTS (FID khi đủ mẫu), SSIM foreground (bbox MegaDetector), byte/ảnh, bpp, compression ratio = 24·H·W / số bit (H, W của ảnh gốc), thời gian encode/decode, kích thước model; chỉ số tác vụ máy theo EVAL-02..07
- [ ] **EVAL-14**: Baseline phổ quát chạy trên cùng tập và cùng giao thức EVAL-11: JPEG và WebP (CPU, quét chất lượng), các model pretrained của CompressAI (`bmshj2018-hyperprior`, `mbt2018`, `cheng2020-attn`, chỉ inference), cùng với B0 (EVAL-10) và VTM/BPG (ANLS-06)
- [ ] **EVAL-15**: Mỗi dòng `results.jsonl` ghi kèm thông tin giao thức: độ phân giải xử lý, độ phân giải đánh giá, thư viện + phiên bản SSIM, cách tính compression ratio, số bước DDIM, seed, định danh checkpoint (mở rộng schema EVAL-01, giữ tương thích ngược)
- [ ] **EVAL-16**: Tập dev cố định lấy từ **validation** (vài trăm ảnh, phân tầng theo site, ngày/đêm, rỗng/có thú) dùng để chọn λ, núm giải mã, α, prompt và độ phân giải xử lý. Test split chỉ chạy một lần ở Phase 6

### H1 — Domain-adaptive fine-tuning

- [ ] **H1-01**: Codec `E_c`/`D_c` + control module fine-tune được trên dữ liệu bẫy ảnh từ checkpoint tác giả, SD 2.1 và RAM++ giữ đóng băng
- [ ] **H1-02**: Từng thành phần loss (`bpp`, `dist`, `diff`, `sem`) log riêng biệt để phát hiện sớm rate collapse
- [ ] **H1-03**: Kiểm tra catastrophic forgetting định kỳ trong lúc train — decode Kodak/COCO ở các mốc cố định để bắt sớm việc lr phá generative prior
- [ ] **H1-04**: Vùng bpp của model fine-tune **chồng lấn** vùng bpp của baseline, kiểm tra tăng dần chứ không đợi tới lúc dựng RD curve mới phát hiện không tính được BD-rate
- [ ] **H1-05**: H1 chính là **một model chung** fine-tune trên train split **gộp của cả 20 site** Kgalagadi (7.191 ảnh) — ít overfit hơn model riêng từng site (nhiều site chỉ có 7–33 sequence train), rẻ hơn ~20 lần và đánh giá được trên tập test của bài so sánh bất kỳ. Chống overfit: chọn checkpoint theo validation, log chênh lệch loss train/val, ít epoch + LR thấp, kiểm tra Kodak định kỳ (H1-03). Model riêng từng site là V2-10
- [ ] **H1-06**: Núm chỉnh phía giải mã không tốn bit (`c_cfg_scale`, `anchor_prior_strength`, số bước DDIM, giải mã thẳng `c_latent` qua VAE) được quét trên tập dev EVAL-16 để thêm điểm RD mà không phải train

### H2 — ROI-weighted loss

- [ ] **H2-01**: `L_dist` gắn trọng số theo ROI mask ở không gian latent VAE (downsample 8×), công thức `W = 1 + (α−1)·M`
- [ ] **H2-02**: `L_sem` gắn trọng số theo ROI, với điểm áp dụng **chuyển từ Middle Block (64×) sang Encoder Layer 9 (32×)** — ở Middle Block một con vật cỡ trung chiếm dưới 1 ô nên ROI weighting vô nghĩa
- [ ] **H2-03**: `L_dist_roi` và `L_dist_bg` log tách biệt ngay từ run đầu tiên — nếu hai đường không tách nhau thì mask không thực sự vào loss
- [ ] **H2-04**: Ảnh overlay mask xuất ra để kiểm tra bằng mắt rằng mask khớp đúng vùng động vật sau crop
- [ ] **H2-05**: Quét tham số α, có ít nhất một biến thể chỉ weight `L_dist` (V1) và một biến thể weight cả hai (V2)
- [ ] **H2-06**: Tỉ lệ false positive trên ảnh rỗng theo dõi ở **từng giá trị α**, để phát hiện nghịch lý bỏ đói nền

### H3 — Domain-aware Tag Guidance Module (tầng L1 + L2, không cần training)

- [ ] **H3-01**: Vocab RAM++ thu gọn (tối đa ~256 tag liên quan động vật hoang dã), mã hoá ⌈log₂ n⌉ bit/tag thay vì 13; bản hiện tại 37 tag / 6 bit (`data/vocab/kgalagadi_wildlife_tags.txt`), quy mô cuối chốt sau H3-05
- [ ] **H3-02**: Structured attribute từ metadata thật — `illumination` suy từ chính ảnh, `season`/giờ từ timestamp, `habitat` từ site ID; tách bạch trong báo cáo với trường phải dự đoán
- [ ] **H3-03**: Tag mới nối vào qua wrapper quanh `TagGCM.extract_tag()`, **không sửa RAM++ và không cần training lại**
- [ ] **H3-04**: Overhead bit thực tế của tag đo được và đối chiếu với bitstream latent
- [ ] **H3-05**: Kiểm chứng thực nghiệm giả định "RAM++ trả về tag mô tả định dạng thay vì nội dung trên ảnh IR" — chạy RAM++ trên 50–100 ảnh đêm trước khi chốt thiết kế H3
- [ ] **H3-06**: Ít nhất 3 template prompt được so sánh có hệ thống, bắt buộc có một template nêu rõ tính chất hồng ngoại đơn sắc
- [ ] **H3-07**: Kiểm tra ảo giác màu trên ảnh đêm IR — xác nhận phương sai chroma của ảnh decode gần 0, vì SD 2.1 gần như không có prior cho ảnh IR đơn sắc
- [ ] **H3-08**: Thí nghiệm đổi prompt lúc decode chạy chồng lên checkpoint H1/H2 để lấp ô bảng ablation với chi phí gần bằng 0
- [ ] **H3-09**: Ảnh đêm IR được **ép thành ảnh xám khi giải mã**, dựa trên bit ngày/đêm có trong bitstream (bit này phải được truyền và tính vào BPP kể cả khi không bật metadata H3). Đo riêng trên ảnh đêm: SSIM, LPIPS, phương sai chroma trước/sau khi ép xám. Không cần train (thêm 04/10/2026)
- [ ] **H3-10**: **Chọn bitrate theo nội dung** ở encoder: MegaDetector (detector/ngưỡng cố định) chạy trên ảnh gốc; ảnh rỗng mã hoá bằng checkpoint λ cao (nén mạnh), ảnh có thú bằng λ thấp; 1 bit trong bitstream cho decoder biết dùng checkpoint nào, tính vào BPP. Báo cáo bắt buộc: tỉ lệ ảnh có thú bị detector bỏ sót (bị nén mạnh nhầm) và ảnh hưởng của nó lên detection/species. Chỉ dùng checkpoint có sẵn, không train (thêm 04/10/2026)

### Analysis — Tổng hợp và đo cái giá của chuyên biệt hoá

- [ ] **ANLS-01**: RD curve **3 điểm bitrate** (λ_rate = 2, 8, 32) cho các cấu hình gốc / +H1 / +full, trên detection và species classification
- [ ] **ANLS-02**: BD-rate tính bằng `bjontegaard`, dùng fit bậc hai cho 3 điểm và **ghi rõ đây là fallback** so với quy ước 4 điểm của JVET
- [ ] **ANLS-03**: Bảng ablation cộng dồn (gốc → +H1 → +H1+H2 → +full) với bootstrap CI ở **mọi ô**, không còn ô trống
- [ ] **ANLS-04**: Bảng cái giá của chuyên biệt hoá trên **cả hai trục** — miền tổng quát (COCO/Kodak) và camera-trap bổ sung (Snapshot Serengeti)
- [ ] **ANLS-05**: Trần vật lý của VAE chứng minh bằng **lập luận Nyquist cộng thí nghiệm FFT** trên ảnh gốc vs ảnh decode, không cần model re-ID
- [ ] **ANLS-06**: Anchor VTM/BPG chạy nền trên CPU để đặt kết quả vào bối cảnh, không cạnh tranh compute unit với training
- [ ] **ANLS-07**: Failure taxonomy — phân loại các dạng lỗi quan sát được kèm ví dụ ảnh, không chỉ chọn ảnh đẹp
- [ ] **ANLS-08**: Mọi figure sinh tự động từ `results.jsonl` qua `make_all_figures.py`, không hardcode số liệu
- [ ] **ANLS-09**: Bài so sánh ngoài được chốt **trước khi vào Phase 6** (ứng viên: Xie et al., CCAI@NeurIPS 2025, cùng Snapshot Kgalagadi; *Class-Agnostic Triple Attention ... Wildlife Camera Trap Images* (Springer); SLIM, arXiv 2512.18200). Khi đã chốt: tính lại chỉ số của bài đó từ kho EVAL-12 và vẽ chung đồ thị — **không train lại**. Nếu bài có code công khai thì chạy lại trên đúng tập test và giao thức EVAL-11

### Report — Đóng gói và bàn giao

- [ ] **REPT-01**: Báo cáo kỹ thuật với Method, Setup, Experiments, Analysis và **mục Limitations thực chất** — nêu trần VAE, rủi ro ảo giác với tính toàn vẹn dữ liệu sinh thái, bản chất pseudo-GT của mask, và các sai lệch cố ý so với setup gốc
- [ ] **REPT-02**: Method và Setup bắt đầu viết từ **giữa dự án**, không dồn vào phase cuối
- [ ] **REPT-03**: Định vị tính mới được phát biểu chính xác — cite TLIC (DCC 2024) và arXiv:2604.01122 (Disney/ETH), khoanh đóng góp vào việc áp dụng trong kiến trúc dual-loss của Diff-ICMH ở miền có tỉ lệ nền/vật cực đoan
- [ ] **REPT-04**: Reproducibility package — env đã pin, `results.jsonl` cùng script sinh figure, manifest split, checkpoint chỉ chứa delta đã fine-tune, và một smoke test tối thiểu
- [ ] **REPT-05**: Slide trình bày và notebook demo chạy được trên ít nhất 2 ảnh mẫu

## v2 Requirements

Hoãn sang sau. Có ghi nhận nhưng không nằm trong roadmap hiện tại.

### Mở rộng đánh giá

- **V2-01**: Segmentation (SAM 2.1 pseudo-GT, mIoU trên CCT) — tác vụ máy thứ ba
- **V2-02**: Đo trần re-ID bằng MegaDescriptor với số liệu thật thay vì lập luận Nyquist
- **V2-03**: Baseline neural codec thứ hai (TransTIC / ELIC / Adapter-ICMH)
- **V2-04**: FID với cỡ mẫu lớn hơn
- **V2-05**: Điểm bitrate thứ 4 để BD-rate đạt đúng quy ước JVET

### Mở rộng phương pháp

- **V2-06**: H3 tầng L3 — grid không gian 3×3 và đếm cá thể thô, cần control module học đọc prompt dạng mới
- **V2-07**: H2 biến thể V3 — thêm term rate-allocation tường minh
- **V2-08**: Dataset bẫy ảnh thứ ba (Wellington / Idaho / Missouri) cho phân tích tổng quát hoá rộng hơn
- **V2-09**: Ảnh nền tham chiếu theo site — gửi một ảnh nền mỗi site một lần, từng ảnh sau chỉ mã hoá phần khác biệt so với nền; khai thác tối đa việc camera đứng yên nhưng cần đổi kiến trúc codec (Future Work)
- **V2-10**: Fine-tune riêng từng site từ model H1 chung, chỉ trên site đủ dữ liệu (B06, A01), để đo lợi ích chuyên biệt hoá theo site

## Out of Scope

Loại trừ tường minh. Ghi lại để chống scope creep.

| Feature | Reason |
|---------|--------|
| H4 — Task-aware SC loss (DINOv2/BioCLIP) | Mâu thuẫn trực tiếp luận điểm task-agnostic của paper gốc; cần ≥60h GPU mà ngân sách Colab Pro (~50h tổng) không có. Ghi vào Future Work. |
| Retrain từ đầu | Paper dùng 4× A100 × 400K iterations. Bất khả thi trên Colab và làm mất khả năng cô lập biến số cần đo. |
| Individual re-ID qua hoa văn | Bị chặn bởi trần vật lý VAE: chu kỳ sọc còn 0,6–1 ô latent sau downsample 8×, dưới ngưỡng Nyquist. Sẽ chứng minh giới hạn này, không hứa hẹn vượt qua. |
| Đánh bại VTM-18.2 về PSNR | Generative codec vốn thua VTM ở PSNR cùng bpp. Baseline đúng là chính Diff-ICMH gốc. |
| Sửa trọng số SD 2.1 UNet hoặc VAE | Phá tan generative prior — mâu thuẫn thiết kế gốc và là nguồn gốc của chính trần chất lượng đang đo. |
| Segmentation trong v1 | Cắt để tiết kiệm 1 conda env và thời gian decode. Detection/species vẫn đủ cho bảng cái giá chuyên biệt hoá. |
| MegaDescriptor trong v1 | Cắt để tiết kiệm env hay xung đột dependency. Trần VAE chứng minh bằng Nyquist + FFT với chi phí gần bằng 0. |
| Nhãn segmentation thật cấp pixel | Không tồn tại cho miền bẫy ảnh. Mọi mask là pseudo-label từ bbox MegaDetector và phải ghi rõ trong Limitations. |
| Background execution của Colab | Chỉ có ở gói Pro+. Kiến trúc phải giả định session chết bất kỳ lúc nào, không trông cậy tính năng này. |

## Traceability

Phase nào phủ requirement nào. Cập nhật khi tạo roadmap.

| Requirement | Phase | Status |
|-------------|-------|--------|
| PRE-01 | Phase 1 | Pending |
| PRE-02 | Phase 1 | Pending |
| PRE-03 | Phase 1 | Pending |
| PRE-04 | Phase 1 | Pending |
| PRE-05 | Phase 1 | Pending |
| PRE-06 | Phase 1 | Pending |
| DATA-02 | Phase 1 | Pending |
| DATA-03 | Phase 1 | Pending |
| DATA-05 | Phase 1 | Pending |
| INFRA-03 | Phase 1 | Pending |
| INFRA-04 | Phase 1 | Pending |
| INFRA-05 | Phase 1 | Pending |
| EVAL-01 | Phase 1 | Pending |
| EVAL-08 | Phase 1 | Pending |
| DATA-01 | Phase 2 | Pending |
| DATA-04 | Phase 2 | Pending |
| DATA-06 | Phase 2 | Pending |
| DATA-07 | Phase 2 | Pending |
| INFRA-01 | Phase 2 | Pending |
| INFRA-02 | Phase 2 | Pending |
| EVAL-02 | Phase 2 | Pending |
| EVAL-03 | Phase 2 | Pending |
| EVAL-04 | Phase 2 | Pending |
| EVAL-05 | Phase 2 | Pending |
| EVAL-06 | Phase 2 | Pending |
| EVAL-07 | Phase 2 | Pending |
| EVAL-09 | Phase 2 | Pending |
| EVAL-10 | Phase 2 | Pending |
| EVAL-11 | Phase 2 | Pending |
| EVAL-12 | Phase 2 | Pending |
| EVAL-13 | Phase 2 | Pending |
| EVAL-14 | Phase 2 | Pending |
| EVAL-15 | Phase 2 | Pending |
| EVAL-16 | Phase 2 | Pending |
| H1-01 | Phase 3 | Pending |
| H1-02 | Phase 3 | Pending |
| H1-03 | Phase 3 | Pending |
| H1-04 | Phase 3 | Pending |
| H1-05 | Phase 3 | Pending |
| H1-06 | Phase 3 | Pending |
| REPT-02 | Phase 3 | Pending |
| H2-01 | Phase 4 | Pending |
| H2-02 | Phase 4 | Pending |
| H2-03 | Phase 4 | Pending |
| H2-04 | Phase 4 | Pending |
| H2-05 | Phase 4 | Pending |
| H2-06 | Phase 4 | Pending |
| H3-01 | Phase 5 | Pending |
| H3-02 | Phase 5 | Pending |
| H3-03 | Phase 5 | Pending |
| H3-04 | Phase 5 | Pending |
| H3-05 | Phase 5 | Pending |
| H3-06 | Phase 5 | Pending |
| H3-07 | Phase 5 | Pending |
| H3-08 | Phase 5 | Pending |
| H3-09 | Phase 5 | Pending |
| H3-10 | Phase 5 | Pending |
| ANLS-01 | Phase 6 | Pending |
| ANLS-02 | Phase 6 | Pending |
| ANLS-03 | Phase 6 | Pending |
| ANLS-04 | Phase 6 | Pending |
| ANLS-05 | Phase 6 | Pending |
| ANLS-06 | Phase 6 | Pending |
| ANLS-07 | Phase 6 | Pending |
| ANLS-08 | Phase 6 | Pending |
| ANLS-09 | Phase 6 | Pending |
| REPT-01 | Phase 6 | Pending |
| REPT-03 | Phase 6 | Pending |
| REPT-04 | Phase 6 | Pending |
| REPT-05 | Phase 6 | Pending |

**Coverage:**
- v1 requirements: 70 total (59 ban đầu + 11 thêm ngày 04/10: EVAL-11..16, H1-05, H1-06, H3-09, H3-10, ANLS-09)
- Mapped to phases: 70
- Unmapped: 0 ✓

---
*Requirements defined: 2026-09-08*
*Last updated: 2026-10-04 — quyết định đánh giá độc lập với bài so sánh (Eval-G), H1 model chung, sửa PRE-05/DATA-02/INFRA-01/H3-01 cho khớp code*
