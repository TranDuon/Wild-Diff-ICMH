# Kế hoạch từng bước: fine-tune Diff-ICMH cho ảnh bẫy ảnh

**Ngày lập:** 16/09/2026  
**Hạn hoàn thành:** 16/11/2026  
**Nhân sự:** TV-A phụ trách dữ liệu và đánh giá; TV-B phụ trách mô hình và training.  
**Giới hạn điều hành:** tối đa 100 compute units (CU) cho kế hoạch v1; đo CU thực tế sau mỗi phiên và lập lại ngân sách. Con số giờ GPU trong tài liệu cũ chỉ là ước lượng.

## 0. Mục tiêu và bằng chứng cần có

Câu hỏi chính: **H2 (loss ưu tiên vùng động vật) hoặc H3 (tag/metadata theo miền) có cải thiện vượt H1 (fine-tune thuần) ở cùng mức bitrate hay không?** Tập ảnh, cách decode, DDIM/DDPM steps, seed, bộ phân loại/phát hiện và ngưỡng đánh giá phải được khóa trước khi so sánh.

Các cấu hình tối thiểu:

| Mã | Cấu hình | Vai trò |
|---|---|---|
| B0 | Checkpoint Diff-ICMH gốc | Mốc trước thích ứng miền |
| B1 | H1: fine-tune codec + control, loss gốc | Mốc chính để xét đóng góp mới |
| B1c | B1 train tiếp đúng số bước/batch/seed như H2, `alpha=1` | Đối chứng cho hiệu ứng train thêm |
| B2 | H1 rồi H2, cùng điểm xuất phát và ngân sách bước như B1c | Đo hiệu ứng ROI loss |
| B3 | B1 + H3 ở đường encode/decode | Đo hiệu ứng H3 độc lập với H2 |
| B4 | B2 + H3 | Đo hiệu ứng kết hợp |

So sánh chính là **B2 so với B1c** và **B3 so với B1** trên cùng ảnh/cùng bpp. B4 cho biết hiệu ứng cộng dồn. Chỉ tuyên bố cải thiện khi chênh lệch có khoảng tin cậy bootstrap theo **site** và không đánh đổi quá mức chất lượng ảnh trống/ảnh đêm.

## 1. Lịch triển khai và cổng kiểm tra

### Tuần 1 — 16–22/09: khóa giao thức và dựng lát cắt 200 ảnh

1. **TV-A tải annotation trước ảnh.** Lấy metadata và bbox chính thức của Snapshot Serengeti; tạo danh sách khoảng 200 ảnh gồm có/không có động vật, ngày/đêm, nhiều site và sequence. Tải từng ảnh được chọn từ kho ảnh đã giải nén, tránh tải ZIP cả season. Ghi nguồn, license, hash và image ID.
2. **TV-A định nghĩa hợp đồng dữ liệu** trong `data/manifests/`: mỗi hàng có `image_id`, `source`, `relative_path`, `site_id`, `sequence_id`, `datetime`, `illumination`, `species`, `boxes`, `width`, `height`, `split`, `sha256`. Trường thiếu phải để `null` với lý do, không tự suy đoán. Kiểm tra EXIF trên 100 ảnh đầu; nếu thiếu, dùng JSON của LILA.
3. **TV-A chia tập theo site.** Mọi ảnh của một site chỉ thuộc một split; sequence/burst không bị xé đôi. `split_check.py` được gọi bắt buộc khi train và eval, có ca kiểm thử cố ý chèn site trùng để xác nhận nó chặn job.
4. **TV-B kiểm kê runtime thật.** Ghi Python, torch/CUDA, GPU, VRAM, package và checkpoint checksum. Đối chiếu `requirements.txt` với runtime Colab; khóa file cài đặt thực sự chạy được thay vì sao chép pin cũ. Notebook Kaggle hiện có chỉ được kiểm tra tĩnh, chưa là bằng chứng GPU inference thành công.
5. **TV-B sửa đường train trên một GPU.** `train.py` hiện tải weights bằng `load_state_dict()` rồi gọi `trainer.fit()` không có `ckpt_path`; cách đó không khôi phục optimizer và bước train. `DiffEIC.training_step(..., optimizer_idx)` hiện dùng hai optimizer; Lightning 2.x cần manual optimization hoặc một phương án đã chứng minh tương đương. Kiểm tra gradient accumulation, hai optimizer, mixed precision và `global_step` bằng run ngắn.
6. **Hai người chạy một vòng tối thiểu:** ảnh → manifest/split → 20–50 bước train thử → checkpoint → encode/decode bằng `inference_partition.py` → một dòng `results.jsonl` gồm actual file-byte bpp, PSNR/LPIPS, seed, config hash, checkpoint hash và CU đã dùng.

**Cổng G1:** không vào training dài nếu chưa có ảnh decode thật, bitstream đọc lại được, split gate chạy tự động, và training trên GPU không lỗi. Dùng tối đa khoảng **8 CU** cho kiểm tra này; nếu một smoke run đã vượt trần, đo lại kế hoạch trước khi tiếp tục.

### Tuần 2 — 23–29/09: xác nhận resume và baseline gốc

7. **TV-B thử ngắt rồi resume thật:** lưu checkpoint khi optimizer đã có state; dừng phiên, khởi tạo tiến trình mới, gọi `trainer.fit(..., ckpt_path=...)`; so sánh `global_step`, LR, optimizer state và loss kế tiếp. Giữ tối thiểu hai checkpoint resume gần nhất. Tạo thêm bản export chỉ gồm trọng số đã train để bàn giao. Nếu loại trọng số SD/VAE/RAM++ khỏi checkpoint resume để tiết kiệm dung lượng, phải nạp lại checkpoint gốc trước rồi chứng minh strict restore và optimizer state vẫn khớp; không giả định bản delta tự resume được.
8. **TV-A khóa protocol eval.** Chọn cố định tập Serengeti held-out theo site và tập CCT hoàn toàn ngoài train; giữ cả ảnh trống. Resize toàn khung hình theo quy tắc chung, pad theo yêu cầu codec, biến đổi bbox cùng hình học. Bpp lấy **tổng byte bitstream thực tế × 8 / số pixel ảnh đánh giá trước pad**, cộng cả header, tag và metadata phải truyền.
9. **TV-A chạy B0 trên tập pilot**, cố định số bước lấy mẫu, seed và checkpoint. Chạy detector/species consumer đã khóa phiên bản trên ảnh gốc và ảnh decode. Kiểm tra phân loại đêm IR, taxonomy mapping và thống kê mAP, AP nhỏ/vừa/lớn, species accuracy, PSNR, MS-SSIM, LPIPS; đánh dấu những metric chưa đủ nhãn hợp lệ. CCT dùng cho đánh giá ngoài miền, không để chọn siêu tham số.
10. **Hai người đo hiệu năng:** thời gian/iteration, thời gian encode/decode/ảnh, CU/giờ trên GPU thực cấp, VRAM peak, thời gian copy checkpoint. Lập lịch lại số bước H1/H2 và cỡ eval dựa trên số đo.

**Cổng G2:** B0 có kết quả tái lập được trên ít nhất một tập ảnh ngày và một tập ảnh đêm; full-state resume đã qua thử nghiệm; ngân sách cho các run còn lại đã tính từ tốc độ đo được. Phân bổ tham chiếu: thêm **10 CU** cho baseline/eval nền tảng.

### Tuần 3 — 30/09–06/10: mở rộng dữ liệu và đánh giá

11. **TV-A mở rộng corpus theo từng đợt.** Mục tiêu metadata/corpus khoảng 60.000 ảnh, nhưng chỉ tải các ảnh cần cho train/eval đã chọn; không buộc mọi run quét hết corpus. Khởi đầu train subset khoảng 10.000–20.000 ảnh cân bằng theo site, ngày/đêm, có/không động vật và kích thước bbox. Quy mô cuối khóa sau phép đo throughput. CCT luôn giữ ngoài train. Ghi số ảnh thật và phân bố thật, không dùng tỉ lệ ước đoán từ tài liệu.
12. **TV-A sinh mask pseudo-GT bằng SAM từ bbox train.** Giữ mask cùng image ID và phép biến đổi hình học; ảnh trống có mask bằng 0. Kiểm tra thủ công một mẫu ngày/đêm, vật nhỏ/lớn; log tỉ lệ phủ và mask lỗi. Mask là pseudo-label cho loss ROI, không trình bày như segmentation GT.
13. **TV-B thêm `WildlifeLICDataset`.** Crop 256×256, biến đổi ảnh, bbox và mask bằng cùng một tham số; lấy mẫu crop chứa vật theo tỉ lệ cấu hình và vẫn giữ crop nền. Kiểm tra overlay sau augment, số crop có vật thực tế, shape/range đầu vào và batch có ảnh trống. Eval không dùng random crop.
14. **TV-A hoàn chỉnh eval harness.** Detector và classifier chạy môi trường riêng nếu dependency xung đột; giao tiếp qua JSON trên đĩa. Lưu prediction từng ảnh để tính lại ngưỡng mà không decode lại. Bootstrap theo site, stratify ngày/đêm, ghi số site và số ảnh. Phát hiện động vật giả trên ảnh trống là metric riêng; một lượt detector cho cả mAP lẫn false positive.

**Cổng G3:** split manifest đã đóng băng; không có leakage site/sequence; mask khớp ảnh sau crop; B0 có bảng ngày/đêm và ảnh trống. Không mở rộng train nếu dữ liệu/nhãn dùng để tính metric chưa qua kiểm tra thủ công.

### Tuần 4–5 — 07–20/10: H1, fine-tune thuần

15. **TV-B khóa H1 config từ checkpoint tác giả.** Chỉ codec `E_c/D_c` và control module được cập nhật; SD 2.1 UNet, VAE, RAM++ đóng băng và xác nhận bằng danh sách `requires_grad` cùng checksum trước/sau. Ghi rõ `l_semantic_weight`: config hiện để `0.0`, nên phải quyết định theo recipe tác giả và giữ nhất quán giữa H1/H2. Không gọi H1 là tái tạo SC loss nếu nó đang tắt.
16. **TV-B chạy pilot ngắn ở một lambda, ưu tiên điểm giữa (ví dụ `lambda_rate=8`).** Bắt đầu LR thấp, batch nhỏ + accumulation được kiểm tra, crop 256². Log bpp ước lượng, bpp file thực, diffusion/guide/semantic loss, gradient norm, VRAM, throughput. Decode fixed panel sau mỗi checkpoint; dừng nếu NaN, rate collapse, chất lượng đêm giảm mạnh hoặc không còn bpp overlap với B0.
17. **TV-A đánh giá bất đồng bộ từng checkpoint H1** trên cùng fixed panel, đặc biệt CCT và một mẫu ảnh tổng quát Kodak/COCO để bắt mất khả năng tổng quát. Chỉ dùng Serengeti validation để chọn checkpoint. Viết Method và Setup từ config/manifest đã khóa.
18. **TV-B mở rộng H1 sang các lambda cần thiết** sau khi pilot đạt G4 và còn đủ CU. Mỗi lambda bắt đầu từ checkpoint tác giả tương ứng; không coi thay đổi lambda trên một checkpoint là tương đương một RD model được huấn luyện riêng nếu không được kiểm chứng.

**Cổng G4:** H1 hơn/khác B0 ở bpp chồng lấn đủ để so sánh; không có lỗi giải mã; frozen weights không đổi; resume qua nhiều phiên giữ nguyên trạng thái. Trần tham chiếu **35 CU** cho H1.

### Tuần 5–6 — 14–27/10: H2 và đối chứng train thêm

19. **TV-B thực hiện ROI loss trên tensor vị trí.** Với mask `M` đã downsample đúng kích thước, dùng `W=1+(alpha-1)M` và loss `sum(W*error)/sum(W)`. Bắt đầu V1: weight `l_guide`/`L_dist` tại latent VAE; kiểm tra `alpha=1` tái tạo loss cũ. Nếu bật SC loss, thử V2 tại `sl_loc=enc_9` sau khi xác nhận shape feature thực tế; không dựa vào giả định tên layer. Log riêng ROI và background, nhưng không coi hai loss phải luôn khác nhau là bằng chứng duy nhất: kiểm tra gradient và thử mask toàn 0/toàn 1.
20. **TV-B tạo B1c và B2 từ cùng H1 checkpoint.** B1c tiếp tục loss gốc (`alpha=1`), B2 dùng alpha thử đầu tiên (ví dụ 2), cùng ảnh, thứ tự batch, seed, số optimizer steps và LR schedule. Nếu còn ngân sách, thử alpha thứ hai (ví dụ 4) và V2 ở alpha tốt nhất; chọn bằng Serengeti validation, tuyệt đối không chọn bằng CCT.
21. **TV-A đo mọi alpha trên cùng ảnh**: AP nhỏ, AP chung, species accuracy, ROI perceptual quality, background quality, false positive và hallucination trên ảnh trống, ngày/đêm. Vẽ paired difference B2−B1c theo site, không chỉ B2−B1.

**Cổng G5:** hiệu ứng H2 còn sau đối chứng B1c; mask có tác động gradient đúng vùng; background và ảnh trống không xuống cấp ngoài ngưỡng đã định trước. Trần tham chiếu **25 CU**, gồm cả B1c. Nếu không đủ CU, giữ B1c + một alpha V1 trước khi quét thêm.

### Tuần 3–7 — 30/09–03/11: H3 chạy cùng lịch với H1/H2

22. **TV-A kiểm tra RAM++ trên 50–100 ảnh đêm** và một mẫu ảnh ngày: đo tỉ lệ tag sai loài, tag chỉ mô tả định dạng, tag trống. Từ đó lập vocab tối đa 256 tag có bảng ánh xạ ID cố định; không mặc định RAM++ hoạt động tốt trên IR.
23. **TV-B sửa cả encoder lẫn decoder tag.** Code hiện ước lượng 13 bit/tag và bitstream mặc định `tag_codelength=13`; vocab 8 bit cần ID remap, version của bảng từ vựng, kiểm tra round-trip tag và đo overhead header thật. Ba template prompt được kiểm trên cùng bitstream/seed; một template nêu rõ ảnh hồng ngoại đơn sắc. Đường DDIM và DDPM phải cho conditioning đúng trước khi so sánh.
24. **Hai người khóa giao thức metadata.** `season/hour` chỉ dùng timestamp có sẵn tại decoder hoặc đã được truyền trong bitstream; `habitat` chỉ dùng khi decoder có site ID/bảng site được chia sẻ; `illumination` suy từ ảnh ở encoder thì phải truyền bit tương ứng nếu decoder không thể biết. Tính cả bit và độ thiếu/sai metadata. Không dùng nhãn GT hoặc thông tin chỉ có ở phía encoder như một oracle không khai báo.
25. **TV-A đánh giá B3 và B4** trên chính bitstream H1/H2, cùng ảnh và seed; so sánh có/không H3 ở bpp gồm overhead. Kiểm tra ảnh đêm có chroma giả và động vật xuất hiện giả trên ảnh trống. Chỉ tuyên bố H3 không cần training sau khi round-trip encoder/decoder chạy thật.

**Cổng G6:** prompt phía decoder được tái tạo chỉ từ bitstream + side information đã khai báo; bpp tăng được đo từ file; kết quả B3/B4 có paired CI. Trần tham chiếu **5 CU** cho H3, chủ yếu inference.

### Tuần 8–9 — 04–16/11: khóa kết quả và bàn giao

26. **TV-A khóa test set và chạy đánh giá cuối một lần.** Serengeti held-out site là kiểm tra trong miền; CCT là kiểm tra ngoài miền. Metric loài trên CCT chỉ báo cáo cho nhãn có mapping taxonomy hợp lệ; ghi rõ số ảnh/loài bị loại. Detector/classifier giữ nguyên giữa B0–B4. Mọi dòng kết quả có ID ảnh, site, checkpoint, bitstream bytes, lambda, seed, steps và phiên bản code.
27. **Hai người tạo RD curves** ở các điểm lambda đã thực sự huấn luyện và có vùng bpp chồng lấn. Tính BD-rate chỉ khi số điểm, monotonicity và overlap đủ cho phép nội suy có nghĩa; nếu không đủ, trình bày chênh lệch ở matched bpp và nêu rõ không tính BD-rate. Không suy diễn đường cong từ một checkpoint bằng cách đổi tham số decode chưa được xác minh.
28. **TV-A làm bảng ablation và cái giá chuyên biệt hóa:** B0/B1/B1c/B2/B3/B4, bootstrap theo site; phân tách ngày/đêm, ảnh có vật/ảnh trống, kích thước vật. Bảng ngoài miền gồm CCT và tập ảnh tổng quát; thêm ảnh failure, không chỉ ví dụ đẹp. Figure sinh từ `results.jsonl` và prediction cache.
29. **TV-B đóng gói tái lập:** config/env lock, manifest split và hash, script kiểm leakage, recipe lấy checkpoint gốc, full-state checkpoint nếu cần tiếp tục train, delta checkpoint để phát hành, script encode/decode/eval, smoke test trên hai ảnh. Báo cáo Limitations nêu pseudo-mask, rủi ro hallucination, taxonomy mismatch, khác biệt crop 256² và tài nguyên so với paper.
30. **Hai người rà soát kết luận.** Nếu B2/B3 không vượt B1c/B1 một cách tin cậy, báo cáo kết quả âm và phân tích nguyên nhân; không đổi metric hoặc test split sau khi xem kết quả.

**Cổng G7:** mọi số trong báo cáo truy ngược được đến manifest, checkpoint, bitstream và code; baseline, H1, đối chứng B1c, H2/H3 cùng bảng ngoài miền đều có kết quả. Trần tham chiếu **12 CU** cho các lượt đo cuối; giữ **5 CU** chưa phân bổ làm dự phòng.

## 2. Quy tắc vận hành mỗi phiên GPU

1. Ghi GPU/VRAM, Python, torch/CUDA, dependency lock, Git revision và CU còn lại; fail fast nếu checksum checkpoint/manifest sai.
2. Copy shard cần dùng từ Drive sang ổ local; chạy `split_check.py` trước khi tạo DataLoader.
3. Chạy 1 batch forward/backward và 1 encode/decode ngắn; kiểm tra finite loss, shape, bitstream.
4. Resume từ full-state checkpoint gần nhất; log số bước bắt đầu và optimizer LR. Checkpoint mỗi khoảng 500–1000 optimizer steps **hoặc theo thời gian thực đo** để mất ít CU khi ngắt. Ghi file tạm, xác nhận checksum, rồi mới cập nhật con trỏ `latest`.
5. Khi dừng, ghi số CU thực, bước cuối, checkpoint hash, throughput và các lỗi vào run manifest; cập nhật dự báo ngân sách trước phiên sau.

## 3. Chỉ số và ngưỡng ra quyết định

- **Tác vụ máy:** detection mAP/AP nhỏ-vừa-lớn; species accuracy ở mức loài và nhóm hợp lệ; false positive và hallucination trên ảnh trống. Báo cáo theo ngày RGB/đêm IR, site, kích thước bbox.
- **Chất lượng nhìn:** PSNR, MS-SSIM, LPIPS/DISTS; ROI và nền đo riêng khi mask/bbox hợp lệ. FID chỉ báo cáo nếu cỡ mẫu và giao thức đủ tin cậy.
- **Rate:** actual file bytes, gồm header, tag, metadata, chia cho pixel trước pad. Bpp ước lượng lúc train là chỉ báo theo dõi, không thay actual bpp.
- **Độ bất định:** paired bootstrap theo site/burst cho chênh lệch giữa cấu hình; kèm `n_images`, `n_sites`, seed và khoảng tin cậy 95%.
- **Tiêu chí ưu tiên:** chọn cấu hình theo gain AP nhỏ/species ở matched bpp trên validation, đồng thời đặt giới hạn trước cho false positive và chất lượng ảnh đêm. Không dùng CCT để chọn alpha, prompt hay checkpoint.

## 4. Ngân sách tham chiếu và phương án khi thiếu CU

| Hạng mục | Trần tham chiếu |
|---|---:|
| Lát cắt + resume | 8 CU |
| Dữ liệu/eval + B0 | 10 CU |
| H1 | 35 CU |
| H2 gồm B1c | 25 CU |
| H3 inference | 5 CU |
| Eval cuối và điểm rate thiếu | 12 CU |
| Dự phòng chưa phân bổ | 5 CU |

Đây là **trần lập kế hoạch**, không phải số CU đã tiêu. Sau G1 và mỗi phase, thay bằng phép đo thực tế. Nếu dự báo vượt 100 CU: giảm cỡ tập eval trong các vòng phát triển (giữ nguyên tập test cuối), giảm số alpha/template thử, giảm số điểm rate phụ và bỏ FID mẫu lớn. Vẫn giữ B0, B1, B1c, ít nhất một B2, kiểm tra H3, split gate và đánh giá ngoài miền. Nếu không đủ điểm RD, không công bố BD-rate.

## 5. Các việc mã nguồn cần làm trước tiên

| Thứ tự | Vị trí hiện tại | Việc cần chứng minh |
|---|---|---|
| 1 | `train.py`, `configs/train_diffeic.yaml` | Runtime một GPU chạy được và `ckpt_path` khôi phục toàn trạng thái |
| 2 | `model/diffeic.py` | Hai optimizer hoạt động đúng trên Lightning hiện hành; chỉ train module được phép |
| 3 | `dataset/licdataset.py`, dataset mới | Ảnh, bbox, mask dùng cùng crop/augment; train 256², eval toàn khung |
| 4 | `model/diffeic.py` loss | ROI loss chuẩn hóa đúng vị trí; alpha=1 bằng loss gốc |
| 5 | `model/lfgcm.py`, `inference_partition.py` | Tag/metadata round-trip, conditioning đúng ở sampler, bpp đo từ file |
| 6 | `src/data/`, `src/eval/` mới | Split gate, manifest, prediction cache, results schema và báo cáo tái lập |

## Nguồn để kiểm tra lại khi triển khai

- [Diff-ICMH bản gốc và checkpoint](https://github.com/RuoyuFeng/Diff-ICMH); [bài báo NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/5c33e9aedee21daeda9e03f43ec4865d-Abstract-Conference.html).
- [Snapshot Serengeti trên LILA](https://lila.science/datasets/snapshot-serengeti), [Caltech Camera Traps trên LILA](https://lila.science/datasets/caltech-camera-traps), [hướng dẫn tải ảnh theo subset](https://lila.science/image-access).
- [Lightning: resume checkpoint](https://lightning.ai/docs/pytorch/2.5.1/common/checkpointing_basic.html), [Lightning: manual optimization với nhiều optimizer](https://lightning.ai/docs/pytorch/stable/model/manual_optimization.html).
