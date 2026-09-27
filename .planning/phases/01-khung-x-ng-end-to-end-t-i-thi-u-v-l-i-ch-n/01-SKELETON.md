# Walking Skeleton — Wild-Diff-ICMH

**Phase:** 1  
**Generated:** 2026-09-27

## Capability Proven End-to-End

Một người vận hành Colab có thể lấy mã nguồn, chuẩn bị lát cắt KGA:A01, train/resume,
decode và ghi metric vào registry bền vững trên Drive.

## Architectural Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Runtime | Google Colab L4 | Tier chuẩn trong ngân sách; người dùng chọn thủ công |
| Data contract | JSONL manifest, sequence-disjoint | Kiểm tra được và phù hợp protocol per-site |
| Persistence | Google Drive | Sống qua runtime reset cho checkpoint/log/result |
| Training | Lightning 2.x full-state resume | Khôi phục optimizer/scheduler/global step |
| Result truth | `results/results.jsonl` | Một registry idempotent duy nhất |

## Stack Touched in Phase 1

- [x] Colab bootstrap và dependency contract
- [x] Manifest/split preflight
- [x] Train, compact checkpoint và resume
- [x] Decode/evaluate và registry
- [ ] Machine-readable closeout artifact

## Out of Scope

- Toàn corpus, ROI mask, MegaDetector/SpeciesNet và baseline đầy đủ — Phase 2.
- H1/H2/H3 thí nghiệm dài — Phase 3–5.

