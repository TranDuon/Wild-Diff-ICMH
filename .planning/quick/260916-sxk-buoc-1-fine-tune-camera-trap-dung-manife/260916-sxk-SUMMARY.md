---
phase: quick-260916-sxk
plan: 01
subsystem: data
tags: [camera-trap, snapshot-serengeti, snapshot-kgalagadi, lila, data-pipeline, split-leakage-gate]

requires: []
provides:
  - "Frozen, site/sequence-disjoint Serengeti train/val manifest (4000/400 rows) with provenance"
  - "Frozen Snapshot Kgalagadi test manifest (450 rows, all 20 sites covered) replacing CCT per user decision"
  - "split_check.assert_no_leakage: importable + CLI leakage/contract gate, called automatically by build_manifests.py and before download_images.py writes .list files"
  - "Resumable, mirror-fallback, zip-slip-safe image downloader with sha256/EXIF/grayscale metadata"
affects: [phase-1-train-py-rework, phase-2-corpus-expansion, wildlife-lic-dataset, split-check-cli-hook]

actuals:
  tokens: 30600
  tasks: 3
  commits: 3
plan_head_before: aebaff6f61d0253e09e3f482d5887aa50e0a539a

tech-stack:
  added: []
  patterns:
    - "Standard-library-only tools/data/*.py scripts (json/zipfile/hashlib/argparse/random/urllib/concurrent.futures), Pillow imported lazily inside functions only"
    - "Purpose-scoped random.Random(f\"{seed}:{purpose}:...\") namespacing for every sampling draw, so adding a new draw never shifts existing ones"
    - "split_check.py owns REQUIRED_FIELDS/VALID_SPLITS as the single data contract; build_manifests.py and download_images.py both import it and call assert_no_leakage before producing trusted output"
    - "safe_join() does a pure lexical path join (no Path.resolve()) to avoid a real Windows race where concurrent threads resolving a not-yet-created root directory misjudge path containment"

key-files:
  created:
    - tools/data/build_manifests.py
    - tools/data/split_check.py
    - tools/data/download_images.py
    - tests/data/conftest.py
    - tests/data/test_pipeline_e2e.py
    - tests/data/test_build_manifests.py
    - tests/data/test_split_check.py
    - tests/data/test_download_images.py
    - data/manifests/serengeti_trainval.jsonl
    - data/manifests/kgalagadi_test.jsonl
    - data/manifests/build_info.json
    - data/manifests/.gitattributes
    - data/manifests/README.md
    - data/raw/.gitignore
  modified: []

key-decisions:
  - "Kgalagadi retrieval strategy: per-image URL under a KGA_public/ prefix on both LILA mirrors (probed live, both return 200 image/jpeg), not the 10.56 GB KGA_S1.lila.zip. The zip strategy is implemented and tested but its member-name mapping was never verified live since it wasn't needed."
  - "Kgalagadi license recorded as CDLA-Permissive-1.0, sourced from the LILA Snapshot Kgalagadi dataset page (its own JSON info block does not state a license)."
  - "safe_join() rewritten to a pure lexical join instead of Path.resolve()+relative_to(): a genuine Windows-only race where concurrent worker threads resolving paths under a not-yet-existing root directory intermittently misjudged containment (flaky ValueError: relative_path escapes root)."

patterns-established:
  - "Data contract owned by one module (split_check.REQUIRED_FIELDS) and imported everywhere else that touches manifest rows."
  - "Every manifest row: any null field has a matching null_reasons entry; nothing is silently guessed."

requirements-completed: [DATA-02, DATA-03, DATA-05]

coverage:
  - id: D1
    description: "build_manifests.py produces site/sequence-disjoint, day/night-stratified Serengeti train/val manifest and a Kgalagadi test manifest covering every site with a non-empty candidate, from the real LILA JSONs"
    requirement: "DATA-02"
    verification:
      - kind: unit
        ref: "tests/data/test_build_manifests.py"
        status: pass
      - kind: e2e
        ref: "tests/data/test_pipeline_e2e.py::test_pipeline_end_to_end"
        status: pass
      - kind: other
        ref: "python tools/data/build_manifests.py (real run against data/raw/snapshot_serengeti/bboxes.json.zip and data/raw/snapshot_kgalagadi/SnapshotKgalagadi_S1_v1.0.json.zip) -> data/manifests/{serengeti_trainval,kgalagadi_test}.jsonl"
        status: pass
    human_judgment: false
  - id: D2
    description: "split_check.py gates leakage/contract violations (shared site, shared sequence, duplicate image_id, invalid split, missing field, null-without-reason, unsafe relative_path, list-file mismatch, build_info sha256 tamper) and runs automatically inside build_manifests.py and before download_images.py writes .list files"
    requirement: "DATA-03"
    verification:
      - kind: unit
        ref: "tests/data/test_split_check.py (leak-injection + contract cases, including a subprocess CLI exit-code check)"
        status: pass
      - kind: other
        ref: "python tools/data/split_check.py data/manifests/serengeti_trainval.jsonl data/manifests/kgalagadi_test.jsonl --build-info data/manifests/build_info.json"
        status: pass
    human_judgment: false
  - id: D3
    description: "download_images.py resumes, falls back between mirrors, is zip-slip/path-traversal safe, records sha256/width/height/is_grayscale/exif_datetime per image, and fetches Kgalagadi images without the 10.56 GB zip"
    requirement: "DATA-05"
    verification:
      - kind: unit
        ref: "tests/data/test_download_images.py (retry/backoff, mirror fallback, integrity, resume/recover, zip strategy, grayscale/EXIF, leakage block, dry-run)"
        status: pass
      - kind: e2e
        ref: "tests/data/test_download_images.py::test_real_download_from_committed_manifests (network, opt-in, run once with WILD_DATA_NET_TESTS=1)"
        status: pass
    human_judgment: false

duration: ~35min
completed: 2026-09-16
status: complete
---

# Quick Task 260916-sxk: Camera-Trap Manifests, Split Gate, Downloader Summary

**Site/sequence-disjoint Serengeti train/val (4000/400) and Kgalagadi test (450, all 20 sites) manifests built from live LILA JSONs, gated by a leak-blocking split_check.py, fetched by a resumable/retrying/zip-slip-safe downloader whose Kgalagadi strategy (per-image URL, not the 10.56 GB zip) was determined by a live probe.**

## Performance

- **Duration:** ~35 min (commits span 21:07:30-21:22:19 local time; includes upfront reading/schema probing before the first commit)
- **Tasks:** 3
- **Files modified:** 14 created, 0 modified (all new)

## Accomplishments

- `tools/data/build_manifests.py`: loads the real Snapshot Serengeti COCO-CT bbox JSON and the real Snapshot Kgalagadi JSON, applies the full contract (`REQUIRED_FIELDS` + `null_reasons`), splits Serengeti by site (never splitting a sequence), stratifies by day/night with proportional `sample_rank` interleaving, excludes person/vehicle-box and null-datetime Serengeti images, and builds the Kgalagadi test set with round-robin site coverage and human-label/no-annotation exclusion.
- `tools/data/split_check.py`: owns the manifest contract (`REQUIRED_FIELDS`, `VALID_SPLITS`, `SplitLeakageError`), `find_violations`/`assert_no_leakage` checked against missing fields, unreasoned nulls, unsafe `relative_path`, shared sites/sequences across splits, duplicate `image_id`, invalid `split`, `.list`-file/manifest mismatches, and `build_info.json` sha256 tampering. Called automatically at the end of `build_manifests.py` and at the start of `download_images.write_list_files`.
- `tools/data/download_images.py`: retry/backoff+jitter fetch with mirror fallback, JPEG SOI + Pillow `verify()` integrity gate, resumable (skip on matching size, recover an unrecorded existing file by hashing it, `--verify-existing` to re-hash), `HttpRangeFile` + thread-local `ZipFile`-per-source cache for the documented zip-strategy fallback, post-download `width`/`height`/`is_grayscale`/`exif_datetime`/`exif_status` metadata, and an `--exif-report` summary.
- Real manifests built and committed from the live LILA JSONs (seed `20260916`), gated, with `output_sha256` provenance in `build_info.json`.
- Live probe determined the Kgalagadi retrieval strategy and confirmed both mirrors work for the lowest-rank Serengeti row; recorded in `download_images.py` and `data/manifests/README.md`.
- `data/manifests/README.md`: field contract table, split rules, rebuild command, full Colab runbook (mount, run, `--limit` slice, strategy flags, `.list` file location, `split_check` gate command), and known limits.

## Task Commits

Each task was committed atomically:

1. **Task 1: End-to-end tracer** - `6f31ad7` (feat)
2. **Task 2: Full manifest contract, Kgalagadi test set, real build** - `4daf863` (feat)
3. **Task 3: Downloader hardening, Colab runbook** - `3d9b82b` (feat)

_The plan's `tdd="true"` tasks (2 and 3) were executed by writing the behavior-covering test files before implementing/hardening the corresponding logic, then verifying all tests pass; each task landed as a single commit (implementation + tests together) rather than separate RED/GREEN commits, since this is a solo quick-task execution rather than a strict TDD-gated phase plan._

## Files Created/Modified

- `tools/data/build_manifests.py` - COCO-CT loader, illumination-from-datetime, Serengeti site/sequence-disjoint stratified sampler, Kgalagadi test-set builder, build_info.json writer
- `tools/data/split_check.py` - manifest contract (`REQUIRED_FIELDS`), leakage/contract violation checks, CLI
- `tools/data/download_images.py` - retrying/mirror-fallback/zip-strategy downloader, `HttpRangeFile`, integrity + metadata extraction, `.list` file writer
- `tests/data/conftest.py` - `make_coco_ct` and `jpeg_mirror` fixtures, `network` marker (opt-in via `WILD_DATA_NET_TESTS=1`)
- `tests/data/test_pipeline_e2e.py` - synthetic end-to-end tracer test
- `tests/data/test_build_manifests.py` - illumination boundaries, contract, exclusions, determinism, Kgalagadi site coverage
- `tests/data/test_split_check.py` - leak-injection cases including a subprocess CLI exit-code check
- `tests/data/test_download_images.py` - local-HTTP-server retry/mirror/integrity/resume/zip-strategy/EXIF tests plus one opt-in network test
- `data/manifests/serengeti_trainval.jsonl` - real, frozen Serengeti train/val manifest (4000/400 rows)
- `data/manifests/kgalagadi_test.jsonl` - real, frozen Kgalagadi test manifest (450 rows)
- `data/manifests/build_info.json` - provenance, per-split stats, exclusion counts, `output_sha256`
- `data/manifests/.gitattributes` - `-text` on `*.jsonl`/`*.json` so `core.autocrlf` never rewrites the manifests
- `data/manifests/README.md` - field contract, split rules, rebuild command, Colab runbook, known limits
- `data/raw/.gitignore` - keeps the multi-GB LILA source zips (bboxes.json.zip, metadata.json.zip, Kgalagadi JSON) out of git

## Real Manifest Counts (from `data/manifests/build_info.json`, seed `20260916`)

**Serengeti** (`serengeti_trainval.jsonl`, 4400 candidate images -> 4000 train + 400 val after exclusions/sampling):
- Excluded before sampling: 1184 images with a null/unparseable `datetime` (illumination_none), 465 images with a person/vehicle box.
- `train`: 4000 images, 203 sites, 4000 sequences (max_per_sequence=1), 2670 day / 1330 night, 243 zero-box images, 6881 total boxes.
- `val`: 400 images, 22 sites, 400 sequences, 242 day / 158 night, 15 zero-box images, 651 total boxes.
- Train and val sites are fully disjoint (203 + 22 = 225 sites total, matching the source JSON's 225 locations); all sequences are disjoint across splits.

**Kgalagadi** (`kgalagadi_test.jsonl`, all `split="test"`):
- Excluded before selection: 135 human-labeled images (`human_label`); 0 with no annotation; 0 with unparseable datetime — the real Kgalagadi JSON has annotations and datetimes for every image.
- 450 rows: 400 `nonempty` + 50 `empty_check`, 338 sequences, 372 day / 78 night.
- Sites covered: **20 / 20** (every Kgalagadi site with a non-empty candidate is represented) — no shortfall.
- Species distribution recorded in `build_info.json` (`species_counts`); top species: `gemsbokoryx` (178), `birdother` (64), `jackalblackbacked` (31), `steenbok` (32), `ostrich` (13).

**No shortfalls** were recorded for either manifest at these default sizes.

## Kgalagadi JSON Schema Findings (live, real file)

Downloaded `SnapshotKgalagadi_S1_v1.0.json.zip` (306,888 bytes) from the verified GCS URL into `data/raw/snapshot_kgalagadi/` (git-ignored). Inspected directly (not assumed):

- Zip member name is `SnapshotKgalagai_S1_v1.0.json` (note: missing the "d" — this is a real typo in the LILA-published archive, not a bug in our code).
- Top-level keys: `info`, `categories`, `annotations`, `images` (same COCO-CT shape as Serengeti).
- `info`: `{"version": "1.0", "description": "Camera trap data from the Snapshot Kgalagai program", "date_created": "2019", "contributor": "Snapshot Safari"}` — **no license key**. License was instead confirmed from the LILA Snapshot Kgalagadi dataset page (fetched via `curl`): *"This data set is released under the [Community Data License Agreement (permissive variant)](https://cdla.io/permissive-1-0/)"* — i.e. `CDLA-Permissive-1.0`, same as Serengeti.
- 31 categories, `id=0` is `"empty"`, `id=1` is `"human"` (exact match, not "person").
- 10,357 images, 10,402 annotations, 20 unique `location` values (sites).
- Annotations are per-image, not sequence-level-only: every annotation carries both `image_id` and `seq_id` (`"sequence_level_annotation": True` is present but doesn't prevent per-image `image_id`). 0 annotations had a missing/unmatched `image_id`; 0 images had zero annotations (so the "seq_id fallback" and "no_annotation" exclusion code paths exist for robustness but never triggered on the real file — `mapping_notes` in `build_info.json` is empty).
- `width`/`height` present on every image (2592x2000); `datetime` present and parseable on every image (0 null datetimes) — the plan's context estimate of "~10,222 images, 20 sites, ~76% empty" matched almost exactly (10,357 images, 20 sites, `empty` count 7886/10357 = 76.1%).
- No `bbox` key anywhere in the file, confirming `boxes: null` (no ground-truth boxes) for every Kgalagadi row.

## Kgalagadi Image-Access Probe (live, Task 3 Step A)

Probed with a handful of HEAD requests (no full-zip download):

| Candidate template | Result |
|---|---|
| `https://storage.googleapis.com/.../KGA/KGA_public/{file_name}` | **200, image/jpeg** |
| `https://lilawildlife.blob.core.windows.net/.../KGA/KGA_public/{file_name}` | **200, image/jpeg** |
| Same two URLs without `KGA_public/` | 404 |

**Chosen strategy: `url`** (per-image fetch under the `KGA_public/` prefix) for `snapshot_kgalagadi`, set as the default `SOURCE_STRATEGY` in `download_images.py`. The 10.56 GB `KGA_S1.lila.zip` season archive is **never downloaded** in the default path. Also confirmed the lowest-`sample_rank` Serengeti row returns 200 on both of its mirrors. The zip strategy (`HttpRangeFile` Range-GETs + thread-local `ZipFile`-per-source) is fully implemented and covered by `tests/data/test_download_images.py`, but its member-name-to-`file_name` mapping (`KGA_ZIP_MEMBER_PREFIX`) was never probed live since the URL strategy already worked — this is documented as an unverified guess in the README and in a code comment, to be confirmed before anyone relies on `--strategy snapshot_kgalagadi=zip` for real.

## Network Smoke-Test Outcome

Ran the opt-in network test once: `WILD_DATA_NET_TESTS=1 python -m pytest tests/data -q -m network` → **1 passed**. It downloaded the lowest-`sample_rank` row from each committed manifest (Serengeti via the `url` mirrors, Kgalagadi via the `url` `KGA_public/` mirrors) into a pytest temp directory and confirmed each file's sha256 matched the recorded checksum.

## Exact Colab Commands

```bash
# mount Drive, then:
python tools/data/download_images.py \
  --manifest data/manifests/serengeti_trainval.jsonl \
  --manifest data/manifests/kgalagadi_test.jsonl \
  --root /content/drive/MyDrive/wild_diff_icmh/images \
  --workers 16

# resume: rerun the same command (skips files whose size already matches the checksum record)

# Week-1 ~210-image slice (70 rows per manifest/split, already site/stratum-diverse):
python tools/data/download_images.py \
  --manifest data/manifests/serengeti_trainval.jsonl \
  --manifest data/manifests/kgalagadi_test.jsonl \
  --root /content/drive/MyDrive/wild_diff_icmh/images \
  --workers 16 --limit 70

# gate before any train/eval job:
python tools/data/split_check.py \
  data/manifests/serengeti_trainval.jsonl data/manifests/kgalagadi_test.jsonl \
  --list train=/content/drive/MyDrive/wild_diff_icmh/images/_lists/serengeti_trainval.train.list \
  --list val=/content/drive/MyDrive/wild_diff_icmh/images/_lists/serengeti_trainval.val.list \
  --build-info data/manifests/build_info.json
```

`.list` files land at `<root>/_lists/<manifest-stem>.<split>.list` (one absolute path per line); point a training config's `file_list` there — `utils/file.py:load_file_list` reads them directly (it does not expand `$DATA_DIR`, confirmed by reading the source).

## Decisions Made

- **Kgalagadi retrieval strategy = `url`**, not `zip`, based on a live HEAD-request probe (see above). Saves ~10.5 GB of unnecessary download per full Kgalagadi pass.
- **Kgalagadi license = `CDLA-Permissive-1.0`**, sourced from the LILA dataset page since the JSON's own `info` block states no license (verified via `curl`, not assumed).
- **`safe_join()` rewritten from `Path.resolve()+relative_to()` to a pure lexical join** — see Deviations below.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed a genuine Windows-only race in `safe_join()`**
- **Found during:** Task 1, running `tests/data/test_pipeline_e2e.py` repeatedly
- **Issue:** `safe_join()` used `Path(root).resolve()` and `dest.relative_to(root_path)` for containment checking. When `root` did not yet exist and multiple `ThreadPoolExecutor` workers concurrently called `safe_join()` while another thread's `dest.parent.mkdir(parents=True)` was creating the same directory tree, `Path.resolve()` intermittently misjudged containment on Windows, raising a flaky `ValueError: relative_path escapes root` (observed on roughly half of ~10 consecutive runs before the fix).
- **Fix:** Rewrote `safe_join()` to do a pure lexical join (`root_path.joinpath(*normalized.split("/"))`) after the existing absolute/drive-letter/`..` rejection checks, avoiding any filesystem access during path safety validation. Also made `run_download` create `root` up front (before spawning workers) as defense in depth.
- **Files modified:** `tools/data/download_images.py`
- **Verification:** Ran `tests/data/test_pipeline_e2e.py` and the full `tests/data` suite 5-10 times consecutively with zero failures after the fix (previously flaky ~50% of the time).
- **Committed in:** `6f31ad7` (Task 1 commit)

**2. [Rule 1 - Bug] Fixed a thread-local zip-reader cache collision across different zip locations for the same source**
- **Found during:** Task 3, writing `tests/data/test_download_images.py::test_zip_member_parent_dir_never_escapes_root`
- **Issue:** `_get_zip_reader()`'s thread-local cache was keyed only by `source` (e.g. `"snapshot_kgalagadi"`). When two `download_row()` calls on the same thread targeted the same source but different zip locations (a real scenario in tests exercising both a local zip and a remote one; also a latent risk if a caller ever changed `zip_locals`/`zip_urls` mid-run), the second call silently reused the first call's already-open `ZipFile` handle pointed at the wrong archive, causing spurious "member not found" failures.
- **Fix:** Changed the cache key to `(source, location)` where `location` is the resolved zip path/URL, so a changed location always opens a fresh reader.
- **Files modified:** `tools/data/download_images.py`
- **Verification:** `tests/data/test_download_images.py` passes reliably (verified 3 consecutive full-suite runs); the failing test now passes both in isolation and as part of the full suite.
- **Committed in:** `3d9b82b` (Task 3 commit)

---

**Total deviations:** 2 auto-fixed (both Rule 1 - genuine bugs found while proving the plan's own test behaviors, not scope changes).
**Impact on plan:** Both fixes are correctness/reliability fixes to code introduced in this same plan (not pre-existing repo issues), found via the plan's own required verification loop. No scope creep — no new files, no new dependencies, no architectural change.

## Issues Encountered

None beyond the two auto-fixed bugs above, which were caught and resolved during the plan's own verification steps before committing.

## User Setup Required

None - no external service configuration required. The Colab downloader runbook (see `data/manifests/README.md` and "Exact Colab Commands" above) is the only manual step, and it is a user-run script invocation, not account/credential setup (all LILA mirrors are public, no auth needed).

## Next Phase Readiness

- The frozen manifests, the leakage gate, and the resumable downloader are ready for the rest of Tuần 1 (CAMERA_TRAP_FINE_TUNING_PLAN.md items 4-6): TV-B's runtime inventory and `train.py` Lightning-2.x fix, and the two-person minimal end-to-end round-trip (image -> manifest/split -> short train -> checkpoint -> `inference_partition.py` -> `results.jsonl`).
- **Not yet wired:** `split_check.py` is not called automatically inside `train.py`/eval (explicitly out of scope for this plan per the orchestrator's locked decision — `train.py` is being reworked separately). The importable `split_check.assert_no_leakage` API is ready for that wiring to call.
- **Not yet downloaded:** actual image bytes. `data/raw/` only holds the metadata JSONs; running `tools/data/download_images.py` (commands above) on Colab/Drive is the next concrete step before any training run can read real pixels.
- `REQUIREMENTS.md` DATA-01 still names CCT as the held-out test set; its wording should be updated to Kgalagadi in a future pass (flagged as a Phase 2 concern per the orchestrator's locked decision, not fixed here to avoid touching an unrelated untracked file mid-task).
- `KGA_ZIP_MEMBER_PREFIX` (the zip-strategy fallback's member-name mapping) is an unverified guess — confirm it live before anyone actually depends on `--strategy snapshot_kgalagadi=zip`.

---
*Phase: quick-260916-sxk*
*Completed: 2026-09-16*

## Self-Check: PASSED

All 14 created files verified present on disk; all 3 task commits (`6f31ad7`, `4daf863`, `3d9b82b`) verified present in `git log --oneline --all`. Full `tests/data` suite (45 passed, 1 skipped) and the opt-in network test (1 passed) re-verified immediately before writing this summary. `data/manifests/{serengeti_trainval,kgalagadi_test}.jsonl` sha256 on disk matches `build_info.json`'s `output_sha256` exactly.
