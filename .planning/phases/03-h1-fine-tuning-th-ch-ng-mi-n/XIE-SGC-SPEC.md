---
phase: 3
plan: 03-01 (dự kiến — plan đầu tiên của Phase 3)
requirement: EVAL-17
status: spec, chưa code
created: 2026-10-07
moved_from: .planning/phases/02-mo-rong-corpus-eval-baseline/02-PLAN.md (Plan 02-07)
---

# Spec — Baseline tái hiện Xie-SGC (EVAL-17)

Chuyển từ Plan 02-07 sang **đầu Phase 3** ngày 07/10/2026 (quick 261007-lm6), để Phase 2 đóng mà không
chờ Xie-SGC. `/gsd-plan-phase 3` dựng plan 03-01 từ spec này.

Bài: Xie, Kay, Haucke, Beery — *Saliency-guided deployment-adaptive compression for wildlife camera
traps*, CCAI@NeurIPS 2025 (`tmp/pdfs/xie2025_saliency_camera_traps.pdf`). Không công bố code/split.
Khái niệm: `../02-mo-rong-corpus-eval-baseline/02-CONTEXT.md` mục A2.

**Đầu vào (đã có từ Phase 2):** split `data/manifests/kgalagadi_site_split.jsonl`, bbox MegaDetector
`phase2/detections/originals.jsonl`, `BASELINE_DEV` (`phase2/kgalagadi_dev_b0eval.txt`), hàm
`score_archive()` và kho baseline của notebook Eval. Không phụ thuộc checkpoint H1.

**Ngân sách:** ≤2 CU, nằm trong trần Phase 3. Pilot một site trước.

## Code

- `tools/baselines/train_xie_sgc.py` (env torch + CompressAI như `run_compressai_zoo.py`):
  - nạp `bmshj2018-hyperprior(quality=q, metric='mse', pretrained=True)`, q ∈ {1, 2, 3};
  - chỉ `g_a` và `h_a` có `requires_grad`; `g_s`, `h_s`, `entropy_bottleneck`, `gaussian_conditional`
    đóng băng (kể cả bảng CDF — không gọi `update()` lại cho phần đã đóng băng); aux loss bỏ qua;
  - dữ liệu: ảnh **train split** của một site (`--site-id`), resize cạnh dài 512 rồi crop 256 ngẫu
    nhiên; mask `W` từ bbox con vật trong `phase2/detections/originals.jsonl` (cùng ngưỡng DATA-06),
    1 trong bbox, `--bg-weight` (mặc định 0,001) ngoài bbox; ảnh rỗng vẫn dùng (toàn bộ là nền);
  - loss `bpp_y + bpp_z + λ_q · 255² · mean(W ⊙ (x − x̂)²)` với λ_q của CompressAI
    (q1 = 0,0018, q2 = 0,0035, q3 = 0,0067); `--bg-weight 1` cho Xie-FT;
  - số bước cố định cho mọi site (định bằng pilot), Adam, lưu state encoder + optimizer để resume;
  - sau train: code + decode ảnh `BASELINE_DEV` **của site đó** bằng entropy coder thật, ghi vào
    kho `BASELINE_ROOT/xie-sgc_ls512/q<q>/` cùng định dạng `run_classical.py` để `score_archive()`
    chấm (curve `xie-sgc_ls512`, `exp_id = p2dev_xie-sgc_ls512__q<q>` — giữ tiền tố `p2dev_` vì cùng
    tập dev với các baseline Phase 2); `run_info.json` ghi `bg_weight`, số bước, site, hash trọng số decoder.
- Test local (model CompressAI giả như `tests/test_compressai_rate.py`): chỉ encoder đổi trọng số,
  decoder giữ nguyên bit-by-bit; `W` khớp bbox sau resize/crop; ảnh rỗng cho `W` toàn nền; chạy lại
  với `--skip-existing` không train lại.

## Chạy trên Colab

- Bước riêng (notebook riêng hoặc phần đầu notebook Phase 3 — chốt khi lập plan 03-01): pilot 1 site
  (B06, nhiều ảnh train nhất) ở q = 2 để đo thời gian → chạy 20 site × q = 1/2/3 → chấm. Kho chung một
  curve cho cả 20 site (mỗi ảnh nén bằng model site của nó). Xie-FT chạy sau cùng, chỉ khi còn CU.
- Đồ thị: SSIM theo compression ratio (trục của bài Xie) gồm JPEG, Ballé pretrained, Xie-FT,
  Xie-SGC, B0; số công bố của bài (403×, 253×) là điểm tham khảo riêng, không so cặp.

## Tiêu chí xong

Xie-SGC có ít nhất q = 1/2/3 trên `BASELINE_DEV` ở cạnh dài 512, mỗi ảnh nén bằng model của site nó,
decoder đúng là decoder pretrained (kiểm bằng so trọng số), chấm đủ chỉ số EVAL-13 + MegaDetector
trước/sau, và có đồ thị SSIM–compression ratio đặt cạnh JPEG, Ballé pretrained và B0.

## Lựa chọn diễn giải (bài gốc không nói rõ)

- "Encoder" = `g_a` + `h_a`; `g_s`, `h_s` và entropy model giữ nguyên để bên nhận dùng decoder pretrained.
- Trọng số nền nhân vào MSE theo pixel, lấy trung bình trên mọi pixel (không chuẩn hoá theo tổng W).
- Ảnh rỗng vẫn dùng khi fine-tune. Bỏ LoRA và blur nền (ghi Limitations).
