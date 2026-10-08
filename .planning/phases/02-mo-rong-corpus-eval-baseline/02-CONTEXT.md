---
phase: 2
created: 2026-10-05
---

# Phase 2 — Context: khái niệm cần nắm

Giải thích cho cả hai thành viên các khái niệm Phase 2 dùng. Giao thức chính thức nằm ở REQUIREMENTS.md (EVAL-11..16) và ROADMAP.md (G-1..G-5).

### A1. Corpus, split, dev set, test set
- **Corpus** = toàn bộ dữ liệu của dự án: 10.222 ảnh Snapshot Kgalagadi, 20 site (vị trí đặt
  camera), tất cả 2592×2000. Đã chia sẵn trong `data/manifests/kgalagadi_site_split.jsonl`:
  **train 7.191 / val 1.499 / test 1.532**. 77% ảnh rỗng (không có con vật), chỉ 376 ảnh đêm (3,7%).
- **Chia theo sequence**: camera bẫy chụp một loạt 3 ảnh gần giống hệt nhau (một sequence). Nếu ảnh
  1 vào train và ảnh 2 vào test thì model đã "thấy" ảnh test, khiến điểm số đẹp giả. Đó là **rò rỉ
  dữ liệu**, và `split_check.py` có nhiệm vụ chặn nó.
- **Tập dev (EVAL-16)**: vài trăm ảnh lấy từ **val**, cố định thành một file danh sách. Mọi lựa chọn
  (λ, số bước DDIM, độ phân giải xử lý, α của H2, prompt H3) đều thử trên dev.
- **Tập test** chỉ chạy **một lần** ở Phase 6. Lý do: chọn cấu hình bằng cách nhìn điểm test thì
  thực chất là "train trên test", con số báo cáo sẽ lạc quan giả.

### A2. Baseline, B0, "hàng đối chứng chịu lực"
- **Baseline** = phương pháp để so sánh với. Câu hỏi trung tâm của dự án là *"fine-tune có giúp
  không, và H2/H3 có giúp hơn fine-tune không"*. Muốn trả lời thì phải biết model **chưa fine-tune**
  cho điểm bao nhiêu trên chính dữ liệu này.
- **B0** = checkpoint Diff-ICMH gốc của tác giả, chạy nguyên trên ảnh bẫy ảnh. Mọi cải thiện của
  B1 (H1), B2 (H2), B3 (H3) đều tính so với B0, nên gọi là "chịu lực": thiếu nó thì cả bảng ablation
  sụp.
- **Baseline phổ quát (EVAL-14)**: JPEG, WebP (codec cổ điển, chạy CPU, quét mức chất lượng),
  các model nén học sâu có sẵn của CompressAI (`bmshj2018-hyperprior` — gần giống codec bài Xie,
  `mbt2018`, `cheng2020-attn`), VTM/BPG nếu kịp. Có các đường này thì khi thầy chỉ định bài so sánh
  nào, gần như chắc đã có điểm tham chiếu chung.
- **Bài Xie et al. 2025** (CCAI@NeurIPS, cùng Snapshot Kgalagadi) là tài liệu so sánh ngoài, không
  phải baseline thực thi của dự án. Dự án không tái hiện Xie-SGC/Xie-FT vì bài không công bố code
  hay split; chỉ trích các số bài đã công bố (ví dụ compression ratio 403× và 253×) vào bảng
  literature comparison, kèm ghi chú khác dữ liệu/giao thức và không tuyên bố thắng trực tiếp.
  `bmshj2018-hyperprior` trong EVAL-14 vẫn là baseline pretrained độc lập, chỉ inference; không được
  gọi là bản tái hiện phương pháp Xie.

### A3. Bitrate, bpp, compression ratio, λ, đường RD
- **bpp** (bit per pixel) = tổng số bit của file nén ÷ số pixel ảnh gốc. Càng nhỏ càng nén mạnh.
- **Compression ratio** = 24·H·W / số bit (24 = 3 kênh × 8 bit). Bài Xie báo cáo theo chỉ số này
  (ví dụ 403×), nhưng chỉ được đối chiếu như số liệu tài liệu nếu giao thức không trùng.
- **λ (lambda-rate, `BPP_WEIGHT`)**: hệ số đánh đổi lúc train, λ lớn thì phạt bit mạnh → file nhỏ,
  ảnh kém hơn. Mỗi λ là một checkpoint riêng → một điểm trên đồ thị.
- **Đường RD (rate–distortion)**: trục x = bpp, trục y = chất lượng. Mỗi phương pháp là một đường;
  đường nằm trên-trái tốt hơn. Hai đường chỉ so được ở **dải bpp chồng lên nhau** (tiêu chí 7: vẽ B0
  và baseline lên chung đồ thị để biết dải chồng lấn trước khi chọn λ cho H1).
- **BD-rate**: một con số tóm tắt "cùng chất lượng thì tiết kiệm bao nhiêu % bit" giữa hai đường RD.

### A4. Chỉ số chất lượng ảnh (người xem)
| Chỉ số | Đo gì | Hướng |
|---|---|---|
| PSNR | Sai khác từng pixel | cao = tốt |
| SSIM / MS-SSIM | Giống cấu trúc (MS = đa thang) | cao = tốt |
| LPIPS, DISTS | Khác biệt cảm nhận, dùng mạng nơ-ron | thấp = tốt |
| FID | Phân bố ảnh tái tạo có "giống ảnh thật" không (cần nhiều ảnh) | thấp = tốt |
| fg-SSIM | SSIM chỉ trong bbox con vật | cao = tốt |

Codec sinh ảnh (diffusion) thường có PSNR thấp nhưng LPIPS/DISTS tốt, vì nó "vẽ lại" chi tiết hợp
lý chứ không chép đúng pixel. Vì vậy phải báo nhiều chỉ số (EVAL-13).

### A5. Chỉ số tác vụ máy (máy xem) và pseudo-label
- **MegaDetector (MD)**: model phát hiện động vật/người/xe chuẩn của ngành bẫy ảnh, trả về bbox.
- **SpeciesNet**: model định loài của Google.
- **Pseudo-label**: Kgalagadi **không có bbox thật** (manifest: `boxes: null` cho cả 10.222 ảnh). Ta
  chạy MD trên **ảnh gốc**, coi kết quả là "đáp án". Sau đó chạy MD trên **ảnh đã nén–giải mã** và so
  với đáp án đó. Phải ghi rõ trong báo cáo đây là nhãn giả.
- **mAP, AP_small/medium/large**: điểm phát hiện tổng hợp, tách theo cỡ con vật (<32², 32²–96², >96²
  pixel). Con vật nhỏ, ở xa là nơi nén dễ làm mất nhất, và cũng là nơi H2 hy vọng giúp nhiều nhất.
- **Species accuracy 2 mức**: đúng loài, và đúng nhóm (ví dụ "linh dương" dù sai loài). Kgalagadi
  có nhãn loài thật từ người gán nhãn (`species` trong manifest), nên chỉ số này **không** phải nhãn giả.
- **False positive trên ảnh rỗng**: MD báo có con vật ở ảnh vốn rỗng.
- **Tỉ lệ ảo giác (EVAL-07)**: rủi ro riêng của codec diffusion là **tự vẽ ra con vật không có
  thật**. Với 77% ảnh rỗng, đây là chỉ số sống còn với người dùng sinh thái học.
- **Tách ngày / đêm (EVAL-05)**: ảnh đêm có hành vi rất khác; gộp chung sẽ che mất lỗi.
  **Định nghĩa (05/10):** "đêm" = ảnh do camera tự chiếu sáng (flash trắng → ảnh màu, hoặc đèn IR →
  ảnh xám), không phải "sau 19 giờ". Xác định từ chính file ảnh, dùng được cho mọi dataset
  (`utils/illumination.py`): EXIF Flash bật → đêm; ảnh xám → đêm; EXIF Flash tắt → ngày; nếu dataset
  có toạ độ thì độ cao mặt trời ≤ −6°; cuối cùng mới đến độ sáng phần trên ảnh (độ tin thấp).
  Trên 450 ảnh Kgalagadi: cả 450 có thẻ Flash; 46 ảnh đêm khớp 100% với độ cao mặt trời; nhãn theo
  giờ cũ gán nhầm 32 ảnh bình minh/hoàng hôn là đêm. Camera Kgalagadi (Cuddeback) dùng flash trắng
  nên ảnh đêm là ảnh **màu**.
- **Env cô lập (EVAL-02)**: MD và SpeciesNet ghim thư viện xung đột nhau, nên mỗi cái chạy trong môi
  trường Python riêng và chỉ trao đổi qua file JSON.

### A6. Đo end-to-end ở độ phân giải gốc (EVAL-11) — thay đổi lớn nhất
- Hiện tại: cắt ô 256×256 ở giữa ảnh → nén → so. Ô đó chỉ chiếm ~1,3% diện tích ảnh, thường không
  chứa con vật. Đây **không** phải điều người dùng thật trải nghiệm.
- Giao thức mới: ảnh gốc 2592×2000 → **thu nhỏ** về cạnh dài 1024 (1024×790) → nén → giải mã →
  **phóng to** lại 2592×2000 → tính mọi chỉ số so với **ảnh gốc**. bpp và compression ratio tính trên
  số pixel ảnh gốc.
- Thu nhỏ vì ảnh gốc cần ~26 GiB VRAM. 1024 là mặc định; chọn lại trên dev (so với 512) dựa trên
  thời gian giải mã và chất lượng.
- Train (Phase 3) cũng phải lấy crop 256 **từ ảnh đã thu về 1024** để con vật có cùng kích thước
  lúc train và lúc test (INFRA-01).

### A7. Kho lưu trữ giải mã (EVAL-12) và thông tin giao thức (EVAL-15)
- Mỗi lần giải mã (tốn GPU) lưu lên Drive: **bitstream**, **ảnh tái tạo ở độ phân giải gốc**, **thời
  gian** encode/decode. Sau này muốn thêm chỉ số hay so với bài mới thì tính lại trên CPU, không giải
  mã lại. Đây là cách "chờ bài so sánh" mà không tốn thêm CU.
- Mỗi dòng `results.jsonl` ghi thêm: độ phân giải xử lý/đánh giá, thư viện SSIM + phiên bản, công
  thức compression ratio, số bước DDIM, seed, checkpoint. Thiếu những thứ này thì về sau không biết
  con số được đo thế nào.

### A8. Các khái niệm còn lại
- **Bootstrap CI (EVAL-09)**: lấy mẫu lại ảnh (theo site) hàng nghìn lần để có khoảng tin cậy, ví dụ
  "LPIPS 0,21 ± 0,01". Chỉ tuyên bố "H2 tốt hơn" khi khoảng tin cậy của hiệu số không chứa 0.
- **Thống kê miền (DATA-04)**: tỉ lệ rỗng, ngày/đêm, phân bố diện tích bbox. Dùng để mô tả dữ liệu
  trong báo cáo và để biết con vật nhỏ đến mức nào.
- **ROI mask (DATA-06)**: mặt nạ vùng con vật dựng từ bbox MD. H2 (Phase 4) dùng nó để đặt trọng
  số loss. Phase 2 sinh ra mask và histogram độ phủ.
- **Shard (DATA-07)**: gói ~10K file nhỏ thành vài file `.tar` theo site. Chép một file lớn từ Drive
  nhanh hơn rất nhiều so với chép 10K file nhỏ.
- **Log tỉ lệ crop có con vật (INFRA-02)**: khi train, crop ngẫu nhiên 256 có thể toàn nền; cần biết
  bao nhiêu % crop thực sự chứa con vật.

---
