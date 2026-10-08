# Phase 2 — Kết quả B0 và baseline trên tập dev

Lượt chạy: notebook `Wild_Diff_ICMH_Phase2_Eval.ipynb` trên Colab, ngày 06/10/2026 (registry cập nhật lần cuối
14:32 UTC). Nguồn số liệu: `wild_diff_icmh/results/results.jsonl` trên Drive, các dòng `exp_id = p2dev_*`.

File trong `results/` (sinh tự động, không sửa tay):

| File | Nội dung |
|---|---|
| [`results/phase2_table.md`](results/phase2_table.md) | Bảng đủ 37 điểm, toàn bộ ảnh (ngày + đêm) |
| [`results/phase2_points.csv`](results/phase2_points.csv) | Mọi điểm, tách `all` / `day` / `night`, kèm khoảng tin cậy 95% |
| [`results/rd_dev.png`](results/rd_dev.png) | 6 đồ thị RD: PSNR, MS-SSIM, LPIPS, mAP, tỉ lệ mất con vật, tỉ lệ con vật "ảo" |

Sinh lại sau khi tải `results.jsonl` mới về `tmp/`:

```
python tools/summarize_phase2.py --registry tmp/results.jsonl --out .planning/phases/02-mo-rong-corpus-eval-baseline/results
```

## Giao thức

- **Tập ảnh:** `phase2/kgalagadi_dev_b0eval.txt`, 202 ảnh validation (167 ngày, 35 đêm) = 100 ảnh dev ngẫu nhiên
  ∪ mọi ảnh dev có con vật (136). Tập này **dư ảnh có con vật** so với corpus (76% ảnh rỗng), nên tỉ lệ ở đây
  không đại diện cho tỉ lệ khi vận hành thật.
- **So cặp:** B0 cạnh dài 512 và mọi baseline chạy trên đúng 202 ảnh này. Ngoại lệ: `B0_ls1024_ddim50` chỉ có 30
  ảnh (tập thử `sub30`), chỉ để tham khảo, không so cặp.
- **bpp** tính trên số pixel của ảnh gốc 2592×2000; mọi chỉ số ảnh so với ảnh gốc (EVAL-11).
- **Chỉ số máy:** MegaDetector trên ảnh decode so với MegaDetector trên ảnh gốc (pseudo-GT, con vật có
  confidence ≥ 0,2). `AP_small` trống vì tập dev không có bbox nào nhỏ hơn 32² pixel.
  - *Tỉ lệ mất con vật*: con vật có trên ảnh gốc nhưng không còn được phát hiện trên ảnh decode.
  - *Tỉ lệ con vật "ảo"* và *báo nhầm trên ảnh rỗng*: phát hiện con vật trên ảnh decode của ảnh gốc không có con vật.
- **Khoảng tin cậy 95%:** bootstrap theo cụm site, 200 lần lấy mẫu lại.
- `mbt2018` và `cheng2020-attn` dùng **bitrate ước lượng** từ likelihood (EVAL-14); các đường khác là mã hoá thật.

## Thống kê miền (toàn corpus, `phase2/domain_stats.json`)

MegaDetector trên ảnh gốc, ngưỡng 0,2 (pseudo-label, không phải nhãn người).

| Chỉ số | Giá trị |
|---|---|
| Số ảnh / có bản ghi MegaDetector | 10.222 / 10.222 |
| Ảnh rỗng theo nhãn người | 77,1% |
| Ảnh MegaDetector thấy con vật | 2.978 (29,1%) — nhiều hơn số ảnh có con vật theo nhãn người (22,9%) |
| Ảnh đêm (nguồn sáng, EXIF Flash) | 238 (2,3%) |
| Số bbox con vật | 3.983 |
| Cỡ bbox theo COCO trên ảnh gốc: large / medium / small | 3.495 / 461 / 27 |
| Cỡ bbox theo COCO ở cạnh dài 1024: large / medium / small | 2.205 / 1.407 / 371 |
| Bbox chiếm < 1% khung hình | 1.594 (40%) |
| Độ phủ ROI mask trung bình | 4,5% khung hình; 7.244 ảnh không có mask |

Hệ quả: trên ảnh gốc gần như không có bbox "small", nên `AP_small` trống ở bảng dưới; nhưng 40% con vật chiếm
dưới 1% khung hình, nên sau khi thu nhỏ để nén chúng thành nhỏ (371 bbox small ở cạnh 1024). Split train/val/test
có tỉ lệ rỗng (77,3% / 77,2% / 76,2%) và độ phủ mask (4,4% / 4,0% / 5,3%) gần nhau.

## Bảng chính: các điểm quanh dải bpp của B0 (toàn bộ ảnh, n = 202)

| Phương pháp | bpp | PSNR | LPIPS ↓ | mAP [CI 95%] | Mất con vật | Con vật "ảo" |
|---|---|---|---|---|---|---|
| **B0 λ=32** | 0,0006 | 16,26 | 0,645 | 0,340 [0,25–0,47] | 25,0% | 6,8% |
| **B0 λ=8** | 0,0029 | 19,82 | 0,564 | 0,566 [0,47–0,70] | 21,2% | 13,6% |
| **B0 λ=2** | 0,0061 | 22,00 | 0,527 | **0,696 [0,61–0,82]** | 12,1% | 5,1% |
| mbt2018 q1 (512) | 0,0047 | 23,93 | 0,645 | 0,186 [0,13–0,26] | 40,9% | 52,5% |
| cheng2020-attn q1 (512) | 0,0049 | 24,18 | 0,641 | 0,089 [0,05–0,14] | 40,9% | 81,4% |
| JPEG q5 (512) | 0,0056 | 21,93 | 0,702 | 0,003 [0,00–0,01] | 100% | 0% |
| Ballé hyperprior q1 (512) | 0,0057 | 23,69 | 0,649 | 0,119 [0,06–0,18] | 34,8% | 81,4% |
| WebP q5 (512) | 0,0093 | 24,03 | 0,664 | 0,368 [0,30–0,43] | 42,4% | 16,9% |
| Ballé hyperprior q3 (512) | 0,0140 | 25,12 | 0,607 | 0,437 [0,34–0,55] | 22,0% | 54,2% |
| WebP q5 (1024) | 0,0319 | 26,15 | 0,532 | 0,679 [0,59–0,77] | 17,4% | 1,7% |
| mbt2018 q3 (1024) | 0,0423 | 28,06 | 0,461 | 0,720 [0,62–0,81] | 10,6% | 18,6% |
| JPEG q15 (1024) | 0,0521 | 26,45 | 0,491 | 0,677 [0,58–0,75] | 20,5% | 0% |

## Nhận xét

1. **Ở cùng bitrate, B0 giữ con vật cho máy tốt hơn hẳn mọi baseline.** Ở ~0,005–0,006 bpp (tỉ lệ nén
   ~3.900×), B0 λ=2 đạt mAP 0,70, trong khi baseline tốt nhất cùng mức chỉ 0,19 (mbt2018) và JPEG gần như mất
   hết con vật. Baseline cần **khoảng 5–8 lần bitrate** (0,03–0,05 bpp, xử lý ở cạnh 1024) mới đạt mAP ~0,68–0,72.
   Ngay cả B0 λ=8 ở 0,0029 bpp (mAP 0,57) cũng cao hơn mọi baseline 512 tới 0,014 bpp.
2. **Về pixel thì B0 kém hơn, về cảm nhận thì tốt hơn.** Cùng bitrate, PSNR của B0 thấp hơn các codec học sâu
   1,7–2,2 dB (ngang JPEG) và MS-SSIM thấp hơn (0,64 so với 0,70–0,71), nhưng LPIPS tốt hơn rõ (0,53 so với 0,64–0,70). Đây là đặc trưng của
   codec sinh ảnh: ảnh sắc nét nhưng không khớp từng pixel. Báo cáo phải trình bày cả hai mặt.
3. **Codec học sâu tối ưu MSE sinh nhiều con vật "ảo" ở bitrate thấp** (52–85% ảnh không có con vật bị
   MegaDetector báo có), trong khi B0 5–14% và JPEG/WebP gần 0%. Hiện tượng này lớn bất thường, **cần kiểm bằng
   mắt** vài ảnh trước khi đưa vào báo cáo (có thể do ảnh nhoè làm MegaDetector báo nhầm ở ngưỡng 0,2).
4. **Dải bpp chồng nhau hẹp.** B0 ở cạnh 512 trải từ 0,0006 đến 0,0061 bpp; baseline thấp nhất là ~0,0047 bpp.
   Chỉ quanh **λ=2** mới có baseline cùng bitrate. λ=8 và λ=32 thấp hơn mọi baseline, nên với chúng chỉ so được
   với chính B0 (và với H1 sau này), không tính được BD-rate so với baseline.
5. **Cạnh 1024 tốn bit gấp ~4 lần.** B0 λ=2 ở cạnh 1024 dùng 0,024 bpp, mAP 0,87, LPIPS 0,30 (chỉ 30 ảnh, chưa so
   cặp). Mọi số so cặp hiện có đều ở cạnh 512.
6. **Ngày/đêm:** B0 λ=2 có mAP 0,71 ban ngày và 0,66 ban đêm. Chỉ có 35 ảnh đêm và rất ít ảnh đêm rỗng, nên
   tỉ lệ con vật "ảo" ban đêm (0% hoặc 50%) chưa có ý nghĩa thống kê.

## Hệ quả cho Phase 3 (H1)

- **λ pilot đề xuất: λ = 2.** Đây là λ duy nhất có baseline cùng bitrate, và vẫn còn chỗ cho H1 cải thiện
  (mất 12% con vật).
- **Mốc H1 phải vượt:** B0 λ=2 ở cạnh 512: mAP 0,696, mất con vật 12,1%, con vật "ảo" 5,1%, LPIPS 0,527 ở
  0,0061 bpp. H1 phải tốt hơn ở cùng hoặc thấp hơn bitrate.
- **Độ phân giải xử lý:** số liệu ủng hộ train và đánh giá H1 ở cạnh 512 để so cặp được với toàn bộ bảng này.
  Việc này sẽ chốt khi thảo luận Phase 3.

## Giới hạn

- Pseudo-GT là MegaDetector trên ảnh gốc, nên mAP đo mức **đồng thuận với MegaDetector**, không đo độ chính xác
  thật so với nhãn người.
- n = 202 và bootstrap theo site cho khoảng tin cậy rộng; chênh lệch nhỏ (< ~0,1 mAP) chưa chắc có ý nghĩa.
- Chưa có SpeciesNet (EVAL-04), nên chưa có chỉ số định loài.
- Bitrate của mbt2018 và cheng2020-attn là ước lượng, có thể thấp hơn một chút so với mã hoá thật.
