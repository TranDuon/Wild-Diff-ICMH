---
phase: 2
plans: 6
status: executing
created: 2026-10-05
updated: 2026-10-07
branch: phase2
---

# Phase 2 — Plan: mở rộng toàn corpus + hạ tầng đánh giá + baseline

Requirements: DATA-01, 04, 06, 07; INFRA-01, 02; EVAL-02..07, 09..16. Ngân sách ≤8 CU.
Khái niệm: xem `02-CONTEXT.md`.

## Plans

Chia ra **code ở máy local (0 CU)** trước, rồi **một notebook Colab Phase 2** chạy phần cần GPU.

### Plan 02-01 — Giao thức độ phân giải gốc (EVAL-11, INFRA-01)
- `utils/image_geometry.py`: thêm `resize_long_side(image, long_side)` và
  `restore_original_size(image, size)` (BICUBIC), cùng hàm map bbox; giữ `center_crop_*` cho smoke.
- `inference_partition.py`: thêm `--processing-long-side` (mặc định 1024 khi có `--manifest`);
  bỏ mặc định crop 256 trong `resolve_crop_size` (giữ `--crop-size` khi truyền tường minh). Codec đã
  tự pad bội 64 (`utils.image.pad`, dòng ~456) và tính bpp trên pixel gốc (dòng ~495): sửa để mẫu
  số là **2592×2000 gốc**, không phải ảnh đã thu nhỏ. Ảnh tái tạo phóng về kích thước gốc trước khi lưu.
- `dataset/camera_trap_dataset.py`: tham số `processing_long_side` — resize trước rồi mới crop (ảnh
  và mask cùng một phép biến đổi, giữ test căn chỉnh hiện có).
- `tools/evaluate_kgalagadi.py`: so với ảnh gốc khi không có `--crop-size`.

### Plan 02-02 — Kho lưu trữ, evaluator, registry (EVAL-12, 13, 15, 09, 08)
- Decode ghi `bitstreams/`, `recon/` (ảnh gốc kích thước, PNG), `timing.jsonl` theo một cấu trúc thư
  mục cố định trên Drive: `archive/<method>/<lambda>/<processing>/<split>/`.
- Evaluator: thêm MS-SSIM, DISTS (pyiqa), byte/ảnh, compression ratio trên ảnh gốc, thời gian; tách
  ngày/đêm (đã có), thêm bootstrap CI theo site.
- `utils/results_registry.py`: thêm trường giao thức EVAL-15 (tuỳ chọn, giữ tương thích ngược;
  `REQUIRED_FIELDS` không đổi).

### Plan 02-03 — Tập dev đóng băng (EVAL-16)
- `tools/data/build_dev_set.py`: chọn ~300 ảnh từ val, phân tầng site × ngày/đêm × rỗng/có thú, lấy
  **cả sequence** (để chọn không bị lệch), seed cố định → `data/manifests/kgalagadi_dev.txt` + test
  khẳng định chỉ chứa ảnh val và không đổi giữa các lần chạy. Lấy dư ảnh đêm (chỉ 3,7%).

### Plan 02-04 — Baseline phổ quát (EVAL-14)
- `tools/baselines/run_classical.py`: JPEG/WebP quét chất lượng qua Pillow, cùng giao thức A6, ghi
  vào cùng kho lưu trữ + registry (CPU, chạy được ở máy local).
- `tools/baselines/run_compressai_zoo.py`: 3 model pretrained × vài mức chất lượng, chỉ inference.
- VTM/BPG: để ANLS-06 (Phase 6) nếu còn thời gian.

### Plan 02-05 — MegaDetector, thống kê miền, chỉ số máy (DATA-04, 06, EVAL-02..07, 10)
- `tools/detect/run_megadetector.py` (env riêng, `PytorchWildlife` MDv6): chạy trên **ảnh gốc** cả
  10.222 ảnh → `detections/kgalagadi_megadetector.json` có record cho **mọi** ảnh (đúng định dạng mà
  preflight H2 đang yêu cầu: `images[].file`, `images[].detections`).
- `tools/data/domain_stats.py`: tỉ lệ rỗng, ngày/đêm, histogram diện tích bbox và độ phủ mask.
- `tools/eval_machine.py`: MD trên ảnh tái tạo → pycocotools mAP tách small/medium/large so với
  pseudo-GT; FP rate và tỉ lệ ảo giác trên ảnh rỗng; SpeciesNet (env riêng) → accuracy 2 mức so với
  `species` trong manifest; tất cả tách ngày/đêm + bootstrap.

### Plan 02-06 — Chuẩn bị H1 chung + notebook Phase 2 (DATA-01, 07, INFRA-02, H1-05)
- `tools/precompute_ram_tags.py` chạy không `--site-id` → một file tag cho cả 20 site.
- `configs/train_kgalagadi_pooled.yaml`: `site_id: null`, `processing_long_side: 1024`.
- Log tỉ lệ crop có con vật (callback hoặc trong dataset) → INFRA-02.
- Shard theo site (`tar`) trên Drive + giải nén về `/content`.
- Notebook mới `Wild_Diff_ICMH_Phase2_Eval.ipynb` (sinh từ file builder như notebook hiện tại):
  chép corpus → MD toàn corpus → tag 20 site → đo thời gian giải mã 1024 vs 512 trên 5 ảnh →
  decode B0 trên dev ở các λ có sẵn → CompressAI zoo → evaluate → đồ thị RD chung.

**Thứ tự:** 02-01 → 02-02 → 02-03 có thể làm ngay ở máy local (0 CU). 02-04 JPEG/WebP chạy được
local. 02-05 và 02-06 cần Colab GPU, gộp vào **1–2 phiên Colab** sau khi Phase 1 đóng (đã có số đo
throughput thật từ `ver2`).

### Cần kiểm tra trước khi chạy Colab (không chặn phần code)
1. Trên Drive, `images/snapshot_kgalagadi/` đã đủ 10.222 ảnh chưa? (Bước 4 chép mọi file có trong
   thư mục; máy local hiện chỉ có 450 ảnh của manifest cũ.)
2. Checkpoint tác giả cho **λ = 8 và 32** (`CNscale1.0_1_1_{8,32}_...`) đã có trên Drive chưa? Hiện
   notebook chỉ dùng λ = 2. Không có thì B0 chỉ có 1 điểm RD, và quyết định G-3 phải xem lại.

---

### Bổ sung sau Phase 1 (04/10)
- **Ngày/đêm theo nguồn sáng (05/10, xong code):** `utils/illumination.py` + `tools/data/label_illumination.py`
  → sidecar `data/manifests/kgalagadi_illumination.jsonl` (chạy trên Colab vì cần đủ ảnh; ~vài phút CPU),
  được evaluator, `build_dev_set.py`, decode H3 và `CameraTrapDataset(illumination_sidecar=...)` dùng.
  Nhãn theo giờ trong manifest chỉ còn để đối chiếu. **Tập dev phải đóng băng lại** sau khi có sidecar
  (bản `f3534b5` phân tầng theo giờ chụp là bản tạm).
- Mùa trong H3 đang mã hoá theo Nam bán cầu — phải tổng quát hoá ở Phase 5 trước khi dùng cho dataset khác.
- Mỗi phiên Colab tốn ~1 CU cố định ⇒ gộp MegaDetector + tag 20 site + decode B0 vào ít phiên nhất.

## Verification
- `python -m pytest -q` sau mỗi plan (hiện 105 passed, 2 skipped), cộng test mới:
  - geometry: thu nhỏ → phóng to giữ đúng kích thước; bpp dùng pixel ảnh gốc;
  - dev set: chỉ chứa ảnh val, lấy trọn sequence, chạy lại ra đúng danh sách cũ;
  - registry: dòng cũ (không có trường giao thức) vẫn hợp lệ;
  - baseline JPEG trên 3 ảnh mẫu local ra bpp/PSNR hợp lý;
  - định dạng detections JSON qua được `_run_asset_preflight` của `train.py` với H2 config.
- Trên Colab: notebook Phase 2 chạy được trên 5 ảnh dev trước khi chạy cả tập dev; ghi CU
  trước/sau mỗi phiên; tổng ≤8 CU.
- Cập nhật ROADMAP (Phase 2: Plans), STATE, `.planning/phases/02-*/` theo khung GSD; commit tiếng
  Anh, không tag Claude, chỉ push khi người dùng đồng ý.
