---
phase: 2
status: passed
verified: 2026-10-08
method: UAT with user (02-UAT.md, 5/5 pass) + 02-RESULTS.md evidence
---

# Phase 2 — Verification

## Tiêu chí thành công (ROADMAP Phase 2)

| # | Tiêu chí | Kết quả | Bằng chứng |
|---|---|---|---|
| 1 | Corpus 10.222 ảnh, split 70/15/15, chép về ổ local | ✅ passed | UAT test 1; notebook Prepare P2-0b, Bước 4 |
| 2 | Thống kê miền + bbox MegaDetector phủ mọi ảnh | ✅ passed | UAT test 2; `phase2/detections/originals.jsonl`, `phase2/domain_stats.json` |
| 3 | Log tỉ lệ crop + ảnh overlay mask (INFRA-02) | ↪ moved | Dời sang Phase 4 ngày 08/10 (ROADMAP, REQUIREMENTS) |
| 4 | Eval harness MegaDetector + SpeciesNet | ✅ MegaDetector / ↪ SpeciesNet moved | Phần MegaDetector: `02-RESULTS.md`; EVAL-04 dời sang Phase 6 |
| 5 | End-to-end ở độ phân giải gốc (EVAL-11) | ✅ passed | UAT test 5 |
| 6 | Tập dev đóng băng, B0 3 λ + baseline đã giải mã và chấm | ✅ passed | UAT test 6; `results/phase2_points.csv` (37 điểm) |
| 7 | Đồ thị RD chung, biết dải chồng lấn | ✅ passed | UAT test 7; `results/rd_dev.png` |

## Requirements của Phase 2 (sau khi dời)

DATA-01, DATA-04, DATA-06, DATA-07, INFRA-01, EVAL-02, EVAL-03, EVAL-05, EVAL-06, EVAL-07, EVAL-09, EVAL-10,
EVAL-11..16 — có bằng chứng trong `02-SUMMARY.md` và `02-RESULTS.md`.

## Acknowledged Gaps

- INFRA-02 → Phase 4 (cần trước lượt train H2 đầu tiên).
- EVAL-04 → Phase 6 (nên làm sớm, song song Phase 3).
- Tỉ lệ con vật "ảo" rất cao của codec CompressAI ở bitrate thấp chưa được kiểm bằng mắt (ghi trong `02-RESULTS.md`).
