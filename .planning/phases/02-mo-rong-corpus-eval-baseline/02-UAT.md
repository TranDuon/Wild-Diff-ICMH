---
status: complete
phase: 02-mo-rong-corpus-eval-baseline
source: [ROADMAP.md Phase 2 success criteria 1-7, 02-PLAN.md, 02-RESULTS.md (no *-SUMMARY.md in phase dir)]
started: 2026-10-08T07:00:00Z
updated: 2026-10-08T08:00:00Z
---

## Current Test

[testing complete]

## Tests

### 1. Corpus đủ 10.222 ảnh và chép về ổ local của Colab
expected: Trong notebook Prepare, ô P2-0b báo đủ 10.222 ảnh Snapshot Kgalagadi (không phải tải bù), và Bước 4 chép ảnh từ Drive về /content/data/... trước khi chạy, không đọc từng file trên Drive.
result: pass

### 2. Bbox MegaDetector phủ mọi ảnh và thống kê miền hợp lý
expected: Trên Drive có phase2/detections/originals.jsonl với một dòng cho mỗi ảnh trong 10.222 ảnh (kể cả ảnh không có con vật), và phase2/domain_stats.json cho tỉ lệ ảnh rỗng ~76%, tỉ lệ đêm vài phần trăm, có histogram diện tích bbox.
result: pass

### 5. Đo end-to-end ở độ phân giải gốc (EVAL-11)
expected: Ảnh tái tạo của B0 trong phase2/archive/B0_ls512_ddim50/lambda_*/ có kích thước 2592×2000 như ảnh gốc, và bpp tính trên số pixel ảnh gốc.
result: pass

### 6. Tập dev đóng băng, B0 3 λ + baseline đã giải mã và chấm
expected: phase2/kgalagadi_dev.txt và kgalagadi_dev_b0eval.txt (202 ảnh) có trên Drive; B0 λ=2/8/32 và mọi baseline có trong results.jsonl với đủ chỉ số EVAL-13 và trường giao thức; bảng 02-RESULTS.md trong repo khớp với những gì bạn thấy ở Bước 10 notebook Eval.
result: pass

### 7. Đồ thị RD chung để biết dải chồng lấn
expected: .planning/phases/02-mo-rong-corpus-eval-baseline/results/rd_dev.png trên GitHub có B0 và mọi baseline trên cùng trục bpp, đọc được dải bpp chồng nhau (quanh λ=2).
result: pass

## Summary

total: 5
passed: 5
issues: 0
pending: 0
skipped: 0
blocked: 0

## Gaps

[none yet]

## Moved Out of Phase 2

Ngày 08/10/2026, theo quyết định của người dùng, hai tiêu chí sau được dời khỏi Phase 2 (ROADMAP,
REQUIREMENTS cập nhật cùng ngày) nên không còn là test của phase này. Ban đầu là test 3 và 4, người dùng bỏ qua.

- Test 3 — *Log tỉ lệ crop chứa con vật và ảnh overlay mask* (INFRA-02) → **Phase 4**, trước lượt train H2 đầu tiên.
- Test 4 — *Eval harness MegaDetector + SpeciesNet* (EVAL-04, phần SpeciesNet) → **Phase 6**, nên làm sớm song song Phase 3.

## Deferred Follow-Ups

- test: 3
  idea: "Ảnh overlay mask ROI + kiểm log crop_stats (INFRA-02) — làm trước Phase 4"
  deferred_at: 2026-10-08
- test: 4
  idea: "SpeciesNet (EVAL-04) cho B0 và baseline (baseline phải decode lại) — làm trước Phase 6"
  deferred_at: 2026-10-08
