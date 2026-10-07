---
quick_id: 261006-hxn
slug: fix-phase2-eval-step7-compressai
date: 2026-10-06
mode: quick
---

# Fix Phase 2 Eval notebook step 7 (CompressAI baselines) and make scoring disconnect-safe

## Evidence (Drive logs, run of 2026-10-05)

- `p2_compressai-mbt2018_ls1024_q*.log`: 300 images took ~2.4 h per point (~29 s/img);
  CompressAI warns "Inference on GPU is not recommended for the autoregressive models".
  `mbt2018` and `cheng2020-attn` encode/decode the latent position by position.
- The runtime died at 01:58 while starting `mbt2018_ls1024_q3` (empty log). No
  `*.done.json` exists: step 8 never ran, so every baseline reconstruction (local SSD,
  by design) was lost and nothing was scored.
- The notebook advertised step 7 as "~10 phút"; the real cost of the plan was ~12 h.

## Tasks

1. **`tools/baselines/run_compressai_zoo.py` — estimated rate for autoregressive models.**
   `--rate {auto,coded,estimated}`; `auto` = `estimated` for `mbt2018` / `cheng2020-attn`,
   `coded` otherwise. Estimated = one forward pass, bits = sum(-log2 likelihoods)
   (CompressAI's own `eval_model --entropy-estimation` convention), no bitstream file,
   `estimated_bits` in `decode_log.jsonl`, `rate` in `run_info.json`.
   **`tools/evaluate_kgalagadi.py`**: accept `estimated_bits` when `data/<stem>` is absent,
   write `rate_source` per row and `rate` in the protocol. Tests for both.
2. **`tools/build_colab_phase2_notebook.py` — score each point as soon as it is coded.**
   Move the step-8 body into `score_archive(root)` defined in step 4; steps 6 and 7 call it
   per point (disconnect loses at most one point); step 8 scores B0 and leftovers.
   Realistic time table. Step 11 no longer raises when CU is blank; it records the session,
   flushes Drive and disconnects the runtime when `AUTO_DISCONNECT = True`.
3. Regenerate both notebooks, update `tests/test_phase2_notebook_contract.py`, run the suite.

## Verify

- `python -m pytest -q` green.
- `run_compressai_zoo.py` estimated mode exercised by a test with a stub model (CompressAI
  has no Windows wheel; real run happens on Colab).
