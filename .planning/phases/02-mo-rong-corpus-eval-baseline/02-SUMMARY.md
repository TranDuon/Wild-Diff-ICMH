---
phase: 2
plan: 02-PLAN.md (plans 02-01..02-06)
status: complete
completed: 2026-10-08
---

# Phase 2 — Summary

Code cho 6 plan xong ngày 05/10/2026; hai notebook Colab chạy 05–06/10 (Prepare, rồi Eval); kết quả đưa vào
repo ngày 08/10 (`02-RESULTS.md`, quick 261008-inp).

## Đã giao

- **02-01 Độ phân giải gốc (EVAL-11, INFRA-01):** resize cạnh dài / phóng về 2592×2000, bpp trên pixel ảnh gốc.
- **02-02 Kho lưu trữ, evaluator, registry (EVAL-12/13/15):** MS-SSIM, DISTS, LPIPS, compression ratio, thời gian,
  bootstrap CI theo site, trường giao thức trong `results.jsonl`.
- **02-03 Tập dev (EVAL-16):** `kgalagadi_dev.txt` đóng băng; tập chấm B0 `kgalagadi_dev_b0eval.txt` (202 ảnh).
- **02-04 Baseline (EVAL-14):** JPEG, WebP, CompressAI `bmshj2018-hyperprior` / `mbt2018` / `cheng2020-attn`
  (hai model tự hồi quy dùng bitrate ước lượng).
- **02-05 MegaDetector + chỉ số máy (DATA-04/06, EVAL-02/03/05/06/07/10):** bbox trên cả 10.222 ảnh gốc,
  thống kê miền, mAP / mất con vật / con vật "ảo" / báo nhầm ảnh rỗng, tách ngày/đêm.
- **02-06 Chuẩn bị H1 chung (DATA-01/07, H1-05):** tag RAM++ 20 site, `configs/train_kgalagadi_pooled.yaml`,
  `CropStatsMonitor`, hai notebook Phase 2.
- **Kết quả:** B0 λ=2/8/32 + 34 điểm baseline trên 202 ảnh dev so cặp — xem `02-RESULTS.md`.

## Dời sang phase sau (08/10)

- INFRA-02 (ảnh overlay mask, bằng chứng log crop) → Phase 4.
- EVAL-04 (SpeciesNet) → Phase 6, nên làm sớm.

## Ghi chú cho Phase 3

- λ pilot đề xuất: 2. Mốc phải vượt: B0 λ=2 cạnh 512 — mAP 0,696, mất con vật 12,1%, LPIPS 0,527 ở 0,0061 bpp.
- Tỉ lệ con vật "ảo" 52–85% của codec CompressAI ở bitrate thấp cần kiểm bằng mắt / ngưỡng 0,5 trước khi báo cáo.
