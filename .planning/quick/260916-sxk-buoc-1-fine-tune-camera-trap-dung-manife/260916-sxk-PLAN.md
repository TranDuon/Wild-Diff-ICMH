---
phase: quick-260916-sxk
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
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
autonomous: true
requirements:
  - DATA-02
  - DATA-03
  - DATA-05
estimate:
  tokens: 130000
  raw_tokens: 130000
  tasks: 3
  confidence: low
must_haves:
  truths:
    - "Running build_manifests.py locally on the LILA JSONs produces serengeti_trainval.jsonl (4000 train + 400 val by default) and kgalagadi_test.jsonl (up to 400 non-empty images spread round-robin over every Kgalagadi site that has non-empty images, plus up to 50 empty_check images). Every row has every contract field, and every null field has an entry in null_reasons."
    - "Serengeti train and val share no site_id and no sequence_id. All Kgalagadi rows are split=test with KGA:-prefixed site ids. split_check.py exits 0 on the committed manifests and exits non-zero when a shared site or sequence is injected on purpose."
    - "Rebuilding with the same seed produces byte-identical manifests. build_info.json records source sha256s, counts, exclusions and output_sha256. split_check --build-info stops immediately if any manifest byte changed."
    - "download_images.py puts each selected image at <root>/<relative_path>. It falls back between mirrors, retries with backoff, and can resume: a second run fetches nothing. It records sha256 and bytes per image, and gets Kgalagadi images without downloading the whole 10.56 GB zip."
    - "Absolute-path .list files per manifest/split are written only after the split gate passes, and utils.file.load_file_list reads them directly."
    - "Each downloaded image gets width, height, is_grayscale and EXIF datetime recorded, and a first-100 EXIF report is printed, so the EXIF check can run on Colab."
    - "The whole tests/data suite runs on CPU with no network access by default. The only network test is opt-in via WILD_DATA_NET_TESTS=1."
  artifacts:
    - path: "tools/data/build_manifests.py"
      provides: "Site-disjoint, stratified, seeded manifest builder for Serengeti train/val and Kgalagadi test"
      contains: "build_kgalagadi_rows"
    - path: "tools/data/split_check.py"
      provides: "Importable + CLI leakage/contract gate (sites, sequences, image ids, required fields, list files, manifest sha256)"
      contains: "assert_no_leakage"
    - path: "tools/data/download_images.py"
      provides: "Resumable concurrent downloader with mirror fallback, remote-zip member extraction, checksums, gated list files"
      contains: "HttpRangeFile"
    - path: "data/manifests/serengeti_trainval.jsonl"
      provides: "Frozen Serengeti train/val manifest"
    - path: "data/manifests/kgalagadi_test.jsonl"
      provides: "Frozen Kgalagadi fixed test manifest"
    - path: "data/manifests/build_info.json"
      provides: "Provenance, counts, exclusions and output sha256 for the manifests"
      contains: "output_sha256"
    - path: "tests/data/test_split_check.py"
      provides: "Deliberate leak-injection cases proving the gate blocks"
      contains: "SplitLeakageError"
  key_links:
    - from: "tools/data/build_manifests.py"
      to: "tools/data/split_check.py"
      via: "builder imports REQUIRED_FIELDS and calls assert_no_leakage on its own outputs before returning 0"
      pattern: "assert_no_leakage"
    - from: "tools/data/download_images.py"
      to: "tools/data/split_check.py"
      via: "write_list_files calls assert_no_leakage before any .list file is written"
      pattern: "assert_no_leakage"
    - from: "download_images.py .list files"
      to: "utils/file.py load_file_list -> dataset/licdataset.py LICDataset"
      via: "one absolute image path per line, newline-terminated"
      pattern: "load_file_list"
    - from: "manifest relative_path"
      to: "download destination on disk"
      via: "safe_join(root, relative_path); zip member names never decide the destination"
      pattern: "safe_join"
    - from: "data/manifests/build_info.json output_sha256"
      to: "split_check.py --build-info"
      via: "sha256 recomputed per manifest file and compared"
      pattern: "output_sha256"
---

<objective>
Step 1 of camera-trap fine-tuning (CAMERA_TRAP_FINE_TUNING_PLAN.md, Tuần 1 items 1-3). Build three pieces: the frozen data manifests (Snapshot Serengeti train/val, Snapshot Kgalagadi fixed test set), a resumable image downloader that the user runs later on Colab/Google Drive, and the `split_check.py` leakage gate. All of it must be proven locally on CPU with no GPU and, by default, no network.

Purpose: Site/burst leakage is the project's first fatal risk (STATE.md Blockers). Both team members depend on the folder-based data contract (`data/manifests/`) to hand work to each other. Every later training/eval job consumes these files.
Output: `tools/data/{build_manifests,split_check,download_images}.py`, `tests/data/*`, and committed `data/manifests/*` (two JSONL manifests, build_info.json, README.md, .gitattributes) built from the real LILA JSONs.

Scope decisions (from the orchestrator's user-approved facts; they act as locked decisions):
- Snapshot Kgalagadi replaces Caltech Camera Traps as the held-out test set, per the user decision. REQUIREMENTS.md DATA-01 still names CCT; that wording is a Phase 2 concern and this plan does not change it.
- Default sizes: 4000 train + 400 val Serengeti images, 400 non-empty + 50 empty_check Kgalagadi images. All sizes are configurable. Fixed seed 20260916.
- Illumination comes from the datetime hour (day 06:00-18:59, else night) with `illumination_source: "datetime_hour_proxy"`. The downloader records an `is_grayscale` flag later.
- The 5.5 GB Serengeti species metadata JSON is NOT loaded. Serengeti `species` is null, with a reason.
- Layout is `tools/data/` (orchestrator suggestion; it matches the repo's existing `tools/` scripts). CAMERA_TRAP_FINE_TUNING_PLAN.md §5 mentions `src/data/`, but `src/` here only holds vendored recognize-anything code.
- Out of scope here: calling split_check automatically inside `train.py`/eval (DATA-03's train/eval hook). `train.py` is being reworked by the separate Phase 1 Lightning-2 item, and editing it would commit an untracked file this task does not own. This plan gives that work the importable `assert_no_leakage` API. The gate already runs automatically inside build_manifests.py and before download_images.py writes list files.
</objective>

<execution_context>
@D:/Diff_ICMH/.claude/gsd-core/workflows/execute-plan.md
@D:/Diff_ICMH/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@.planning/CAMERA_TRAP_FINE_TUNING_PLAN.md
@.claude/CLAUDE.md
@utils/file.py
@dataset/licdataset.py

Verified facts (do not re-derive):
- utils/file.py `load_file_list` only strips each line. It does NOT expand `$DATA_DIR` (datalists/train.list uses that prefix, but nothing expands it). So list files must hold absolute paths and be generated on the machine that holds the images (Colab).
- `data/raw/snapshot_serengeti/bboxes.json.zip` holds one member, `SnapshotSerengetiBboxes_20190903.json` (info.version "20190903"; its description says boxes under 400 px² were removed manually). Contents:
  - categories: 1 animal, 2 person, 3 group, 4 vehicle.
  - 82,938 images, all 2048x1536, seasons S1-S6. 78,029 have at least one annotation; 4,909 have zero boxes.
  - 146,359 annotations: 145,380 animal, 544 person, 435 group.
  - 225 locations (largest has 1,755 images, smallest has 4).
  - 34,166 seq_ids; no seq_id spans two locations.
  - 1,184 images have a null datetime. Non-null datetimes are all "YYYY-MM-DD HH:MM:SS". About 20.4% of images fall at night by the hour rule.
  - Image keys: id, file_name, seq_id, location, height, width, seq_num_frames, frame_num, season, datetime. Annotation keys: id, category_id, image_id, bbox [x,y,w,h] in absolute pixels.
- Serengeti per-image URLs return HTTP 200:
  - `https://storage.googleapis.com/public-datasets-lila/snapshotserengeti-unzipped/<file_name>`
  - `https://lilawildlife.blob.core.windows.net/lila-wildlife/snapshotserengeti-unzipped/<file_name>`
- Kgalagadi:
  - Metadata `https://storage.googleapis.com/public-datasets-lila/snapshot-safari/KGA/SnapshotKgalagadi_S1_v1.0.json.zip` (0.3 MB, HTTP 200).
  - Images are only confirmed as `https://storage.googleapis.com/public-datasets-lila/snapshot-safari/KGA/KGA_S1.lila.zip` (10.56 GB, HTTP 200).
  - About 10,222 images, 20 sites, ~76% empty.
- Local environment: Python 3.13.6 (Windows), pytest 9.1.1, Pillow 12.0.0. Colab runs Python 3.12 and has Pillow. Use only the standard library plus an optional lazy Pillow import; install no packages.
- git: `core.autocrlf=true` in this repo. `data/raw/` is NOT ignored today, and neither is anything else under `data/`. Almost every repo source file is untracked. Stage only the paths listed in files_modified, each named explicitly.
</context>

<tasks>

<task type="tracer">
  <name>Task 1: End-to-end tracer — synthetic Serengeti COCO-CT -> manifest -> split gate -> download -> .list read by load_file_list</name>
  <files>tools/data/build_manifests.py, tools/data/split_check.py, tools/data/download_images.py, tests/data/conftest.py, tests/data/test_pipeline_e2e.py</files>
  <read_first>utils/file.py, dataset/licdataset.py (lines 1-30), tools/verify_colab_t4_notebook.py (repo script style), .planning/CAMERA_TRAP_FINE_TUNING_PLAN.md section "Tuần 1" items 1-3</read_first>
  <action>
Build ONE thin path through all three scripts using the Serengeti source only; Kgalagadi, stratification and hardening come in Tasks 2-3. Tasks 2-3 must be able to fill in function bodies without changing these signatures.

Conventions for all three scripts:
- Standard library only (json, zipfile, hashlib, argparse, random, datetime, urllib, concurrent.futures, pathlib). Pillow is imported lazily inside functions only.
- Each script has a module docstring, a `main(argv=None) -> int`, and a `sys.exit(main())` guard.
- Scripts import each other by putting `Path(__file__).resolve().parent` on sys.path, so `python tools/data/<name>.py` works from the repo root on Windows and on Colab.
- Do not add `__init__.py` under tools/.

split_check.py (owns the data contract):
- Define `REQUIRED_FIELDS`, a tuple with: image_id, source, source_version, license, relative_path, source_file_name, site_id, sequence_id, frame_num, datetime, illumination, illumination_source, species, is_empty, boxes, width, height, split, subset, sample_rank, sha256, null_reasons. This is the CAMERA_TRAP_FINE_TUNING_PLAN.md item-2 contract plus provenance and ranking fields.
- Define `VALID_SPLITS = {"train", "val", "test"}` and a `SplitLeakageError(RuntimeError)`.
- `load_manifest_rows(paths)` reads JSONL rows.
- `find_violations(rows) -> list[str]` returns readable messages for:
  - a site_id present in more than one split;
  - a sequence_id present in more than one split;
  - a duplicate image_id;
  - a split value outside VALID_SPLITS.
- `assert_no_leakage(manifest_paths, list_files=None, build_info=None)` raises SplitLeakageError listing the first 20 violations plus the total count. The list_files and build_info parameters are accepted now and implemented in Task 2.
- CLI:
  - positional manifest paths;
  - on success, print "split_check: OK" with row/site/sequence counts and exit 0;
  - otherwise print each violation, then "split_check: FAILED", and exit 1.

build_manifests.py:
- `load_coco_ct(path)` accepts a .json file or a .zip holding exactly one .json member. It returns `(data, meta)`, where meta holds the repo-relative posix path, the sha256 of the file on disk, and the member name.
- `illumination_from_datetime(value)` returns `(label, reason)`:
  - It parses "YYYY-MM-DD HH:MM:SS" and also accepts a "T" separator.
  - Hour 6 through 18 inclusive gives "day"; any other hour gives "night".
  - A missing or unparseable value gives `(None, <reason text>)`.
- `make_serengeti_row(image, anns, category_names, source_version)` fills the fields as follows:

| Field | Value |
|---|---|
| image_id | "SER:" + id |
| source | "snapshot_serengeti" |
| license | "CDLA-Permissive-1.0" |
| relative_path | "snapshot_serengeti/" + file_name, forward slashes |
| source_file_name | file_name |
| site_id | "SER:" + location |
| sequence_id | "SER:" + seq_id |
| frame_num, datetime, width, height | copied from the image record |
| illumination | from `illumination_from_datetime` |
| illumination_source | "datetime_hour_proxy" |
| species | None. Reason: species labels live in the 5.5 GB SnapshotSerengeti_S1-11_v2.1.json, which is intentionally not loaded. |
| is_empty | None. Reason: the 20190903 bbox file removed boxes under 400 px², so zero boxes does not prove an empty frame. |
| boxes | list of objects with keys "category" (name) and "bbox_xywh" (absolute pixels), ordered by annotation id; an empty list when the image has no annotations |
| subset | "bbox_subset" |
| split, sample_rank | assigned later |
| sha256 | None. Reason: filled by download_images.py in the checksums file. |
| null_reasons | a dict with one reason for EVERY field whose value is None |

- `split_and_sample_serengeti(rows, *, seed, train_size, val_size, val_site_fraction, max_per_sequence)`, tracer version:
  1. Seed-shuffle the sorted unique site_ids. The first max(1, round(fraction x n_sites)) sites go to val; the rest go to train.
  2. For each split, keep at most max_per_sequence frames per sequence, seed-shuffle, and take the first N.
  3. Set split, and set sample_rank to the 0-based pick order within that split.
  - Use purpose-scoped RNGs, e.g. `random.Random(f"{seed}:val_sites")`, so draws added later do not shift existing ones.
- `write_jsonl(rows, path)`:
  - sort rows by (split, sample_rank, image_id);
  - write each row with `json.dumps(sort_keys=True, ensure_ascii=False)`;
  - open the file with `encoding="utf-8"` and `newline="\n"`, so the bytes are identical on Windows and Linux.
- CLI flags and defaults: `--serengeti-bbox-json` (data/raw/snapshot_serengeti/bboxes.json.zip), `--out-dir` (data/manifests), `--seed` (20260916), `--train-size` (4000), `--val-size` (400), `--val-site-fraction` (0.1), `--max-per-sequence` (1).
- Output file: `<out-dir>/serengeti_trainval.jsonl`. After writing, call `split_check.assert_no_leakage` on it. If it raises, print the error and return 1.

download_images.py:
- `DEFAULT_MIRRORS` holds the snapshot_serengeti templates: GCS first (Colab runs inside GCP), then Azure. The templates are exactly the two verified URLs above, with a `{file_name}` placeholder.
- `--mirror-template SOURCE=TEMPLATE` (repeatable) replaces the defaults for that source. Tests use it with file:// URLs.
- `safe_join(root, relative_path)` normalizes separators. It raises ValueError for absolute paths, drive-letter paths and any ".." segment, and otherwise returns the absolute path under root.
- `candidate_urls(row, mirrors)` fills each template with `urllib.parse.quote(source_file_name, safe="/")`.
- `fetch_url(url, timeout=60, max_bytes=50 MiB)` uses urllib.request, reads in chunks, and aborts once the body passes max_bytes.
- `download_row(row, root, mirrors, fetch=fetch_url)`:
  - In the tracer, it tries each candidate once.
  - It writes to dest + ".part", then calls os.replace into place.
  - It returns a record dict with image_id, relative_path, sha256, bytes, url and status.
- `load_checksums(path)` returns a dict keyed by image_id.
- Only the main thread appends to the checksums JSONL (default `<root>/_meta/checksums.jsonl`); worker threads only return records.
- A row is skipped when its record exists and the file exists with a matching size.
- `write_list_files(manifest_paths, rows, root, checksums, list_dir)`:
  1. Call `split_check.assert_no_leakage(manifest_paths)` first.
  2. For each (manifest file stem, split), write `<list_dir>/<stem>.<split>.list`. Each line is the absolute path of a row that has a checksum record, in manifest order, "\n"-terminated.
- `run_download(args) -> dict` returns summary counters: downloaded, skipped, failed. `main` wraps it and returns 1 if failed > 0.
- CLI flags and defaults: `--manifest` (repeatable, required), `--root` (required), `--checksums`, `--list-dir` (default `<root>/_lists`), `--workers` (16), `--splits` (default: all), `--limit N`, `--dry-run`.
  - `--limit N` keeps the N lowest sample_rank rows per (manifest, split).
  - `--dry-run` prints candidate URLs and creates nothing on disk.
- Downloads run through `ThreadPoolExecutor(max_workers=workers)`. Print progress every 100 rows and a one-line summary at the end.

tests/data/conftest.py:
- Insert the repo root and tools/data at the front of sys.path.
- `make_coco_ct` fixture: a factory that builds an in-memory COCO-CT dict and writes it into tmp_path as .json or .zip. Parameters: n_sites, seqs_per_site, frames_per_seq, the fraction of night-hour sequences, plus switches that add one null-datetime image, one zero-box image and one person-box image.
- `jpeg_mirror` fixture: writes small real Pillow JPEGs at mirror_dir/source_file_name for the given rows.
- Register a `network` marker in `pytest_configure`. Skip network-marked tests unless the env var WILD_DATA_NET_TESTS=1 is set.

tests/data/test_pipeline_e2e.py, using a synthetic zip with 12 sites x 4 sequences x 3 frames:
1. `build_manifests.main` with train_size 20, val_size 4 and val_site_fraction 0.25 returns 0.
2. `split_check.main` on the output returns 0.
3. `download_images.main` with `--mirror-template snapshot_serengeti=<mirror_dir.as_uri()>/{file_name}` returns 0.
4. The checksums file has one record per manifest row, and each sha256 equals hashlib's sha256 of the mirror file.
5. `serengeti_trainval.train.list` and `serengeti_trainval.val.list` load through `utils.file.load_file_list`. Their counts equal the manifest split counts, and every listed path exists.
6. A second `run_download` reports downloaded == 0.
  </action>
  <verify>
    <automated>python -m pytest tests/data/test_pipeline_e2e.py -q</automated>
  </verify>
  <done>With no network, the synthetic COCO-CT file goes through build_manifests, split_check, download_images (file:// mirror) and the .list files, which load_file_list reads with correct per-split counts and existing absolute paths. sha256 values match the source bytes, and a rerun downloads nothing.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Full manifest contract — stratified site-disjoint Serengeti sampling, Kgalagadi test set, hardened gate, real manifest build</name>
  <files>tools/data/build_manifests.py, tools/data/split_check.py, tests/data/test_build_manifests.py, tests/data/test_split_check.py, data/manifests/serengeti_trainval.jsonl, data/manifests/kgalagadi_test.jsonl, data/manifests/build_info.json, data/manifests/.gitattributes, data/raw/.gitignore</files>
  <read_first>tools/data/build_manifests.py, tools/data/split_check.py, tests/data/conftest.py (all from Task 1)</read_first>
  <behavior>
    - illumination_from_datetime: "2010-10-08 05:59:00" gives night, 06:00:00 gives day, 18:59:59 gives day, 19:00:00 gives night. None and "garbage" give (None, a non-empty reason).
    - Every built row has every REQUIRED_FIELDS key, and every None-valued field has a null_reasons entry.
    - Serengeti train and val site sets are disjoint and so are their sequence sets. No sequence has more than max_per_sequence frames. Val has at least one day row and one night row.
    - Images with person or vehicle boxes, and images with a null datetime, are excluded and counted in build_info.
    - The same seed gives byte-identical files in two separate output dirs. A different seed gives a different train selection.
    - Any sample_rank prefix mixes sites and day/night strata: the first 20 train rows cover at least min(20, n_train_sites) sites.
    - A train_size larger than the train pool raises ValueError.
    - Kgalagadi: every site with a non-empty candidate is covered when nonempty_size is at least that site count. empty_check rows have is_empty True and subset "empty_check". Images with a human-type label and images with no annotation are excluded. Every site_id starts with "KGA:" and every split is "test".
    - split_check: a clean manifest passes. The gate reports a violation in each of these cases:
      - one site shared between train and val. This is the deliberate leak case from plan item 3: find_violations is non-empty, assert_no_leakage raises SplitLeakageError, and a subprocess run of tools/data/split_check.py exits non-zero.
      - a sequence shared across splits even when the site ids differ;
      - a duplicate image_id;
      - an invalid split;
      - a missing field, or a None value with no reason;
      - a relative_path containing "..";
      - a list file that declares a val path as train;
      - a manifest byte tampered after build_info was written.
  </behavior>
  <action>
Step A — raw-data hygiene and the Kgalagadi JSON:
- Create data/raw/.gitignore with exactly two lines: `*` and `!.gitignore`. This keeps the 55 MB / 5.5 GB LILA zips and the Kgalagadi JSON from ever being staged.
- Download SnapshotKgalagadi_S1_v1.0.json.zip from the verified GCS URL into data/raw/snapshot_kgalagadi/ with urllib.
- Inspect the real schema: top-level keys; image and annotation keys; all category names, including the empty and human-type labels; whether annotations are image-level or sequence-level; the datetime format; whether width/height exist; the location count; and the info/license block.
- Write the findings into the SUMMARY and map fields from the keys actually found. Do not assume keys.

Step B — Serengeti sampling:
- Replace the tracer internals of `split_and_sample_serengeti`. Keep the signature and add the kwarg `night_fraction=None`.
- Exclusions, applied before sampling and counted in `build_info.excluded` with reasons:
  - illumination is None (null or unparseable datetime);
  - any box whose category is person or vehicle (privacy and off-domain).
  - Zero-box images stay in the pool, flagged by boxes [] and is_empty None.
- Val site selection: take sites from the seeded site order into val until ALL of these hold:
  - the val site count reaches max(1, round(val_site_fraction x n_sites));
  - the val pool (sequence-capped) holds at least val_size rows;
  - the val pool has at least one day candidate and at least one night candidate.
  The remaining sites go to train. If the train pool is smaller than train_size, raise ValueError stating the shortfall.
- `sample_balanced(candidates, n, rng, max_per_sequence, night_fraction)`:
  1. Pick at most max_per_sequence frames per sequence (seeded).
  2. The night quota is round(n x night_fraction). When night_fraction is None, use the pool's own night share (keeps the data's real distribution; this is the default choice).
  3. Within each stratum, go round-robin over sites. Sites are in seeded order, and each site's candidates are in seeded order.
  4. If one stratum runs out, fill the gap from the other.
  5. Assign sample_rank by interleaving the day and night picks in proportion to their counts, so any rank prefix stays diverse. `download_images.py --limit` depends on this for the ~200-image Week-1 slice.
- Keep every RNG purpose-scoped.

Step C — `build_kgalagadi_rows(coco, meta, *, seed, nonempty_size=400, empty_size=50)`:
- Field mapping:

| Field | Value |
|---|---|
| image_id | "KGA:" + id |
| source | "snapshot_kgalagadi" |
| source_version | from the JSON member name (e.g. SnapshotKgalagadi_S1_v1.0) |
| license | the JSON info block if it states one; otherwise the LILA Snapshot Kgalagadi dataset page (WebFetch is allowed); otherwise None with a reason |
| relative_path | "snapshot_kgalagadi/" + file_name |
| site_id | "KGA:" + location |
| sequence_id | "KGA:" + seq_id |
| datetime, illumination | through `illumination_from_datetime` |
| species | sorted unique label names on the image, excluding the empty label |
| is_empty | True when the empty label is the only label |
| boxes | None. Reason: Snapshot Kgalagadi has no bbox ground truth; MegaDetector boxes are produced at eval time. |
| width, height | from the JSON when present; otherwise None. Reason: absent from the LILA JSON; recorded from the downloaded file in the checksums file. |
| split | "test" |
| subset | "nonempty" or "empty_check" |

- Labels are matched to images by image_id. If the annotations are sequence-level only, match them by seq_id instead and write that into `build_info.mapping_notes`.
- Exclusions, each counted in build_info: any image with a label whose name contains "human"; images with no annotation at all (unlabeled is not empty); images whose illumination is None.
- Selection:
  - nonempty: at most 1 frame per sequence, round-robin across sites in seeded order;
  - empty_check: images from distinct sequences, round-robin across sites.
  - When fewer candidates exist than requested, take all of them and record the shortfall in build_info. Do not quietly loosen the exclusions.
  - sample_rank interleaves the two subsets in proportion to their sizes.
- New CLI flags and defaults: `--kgalagadi-json` (data/raw/snapshot_kgalagadi/SnapshotKgalagadi_S1_v1.0.json.zip), `--kga-nonempty-size`, `--kga-empty-size`, `--skip-kgalagadi`.
- Output: `<out-dir>/kgalagadi_test.jsonl`. The builder gates on both manifests together.

Step D — `<out-dir>/build_info.json`:
- Format: `json.dumps(indent=2, sort_keys=True)`, "\n" newlines, and no timestamps, so a rebuild is reproducible.
- Contents:
  - the generator script path;
  - the seed and every size argument;
  - sources: repo-relative posix path, sha256, member name, and the JSON info block;
  - per manifest and per split: images, sites, sequences, day, night, zero_box_images, total_boxes, subset counts, and species_counts for Kgalagadi;
  - excluded counts with reasons;
  - kga_sites_total and kga_sites_covered;
  - shortfalls and mapping_notes;
  - output_sha256, mapping each manifest file name to its sha256.

Step E — harden split_check.py:
- `find_violations` also reports:
  - rows missing any REQUIRED_FIELDS key;
  - None values with no null_reasons entry;
  - relative_path values that break the safety rule (absolute path, drive letter, or a ".." segment).
  build_manifests.py keeps importing REQUIRED_FIELDS, so the gate owns the contract.
- `assert_no_leakage` implements `list_files` (a dict mapping split to path). Each listed path is normalized to forward slashes and matched to a manifest row whose relative_path it ends with. Report paths that match no row, or that match a row whose split differs from the declared split.
- `assert_no_leakage` also implements `build_info`: recompute each manifest file's sha256 and report mismatches with output_sha256. This is the fail-fast manifest checksum from CAMERA_TRAP_FINE_TUNING_PLAN.md §2 rule 1.
- New CLI flags: `--list SPLIT=PATH` (repeatable) and `--build-info PATH`.

Step F — create data/manifests/.gitattributes with the lines `*.jsonl -text` and `*.json -text`. The repo has core.autocrlf=true, and a CRLF checkout would otherwise break the sha256 gate.

Step G — tests:
- Write tests/data/test_build_manifests.py and tests/data/test_split_check.py covering every behavior above. Write them first, confirm they fail, then implement.
- Build the synthetic Kgalagadi data with a small COCO-CT factory. Add it to conftest only if both test files need it; otherwise keep it local to the test file.

Step H — real build:
- Run `python tools/data/build_manifests.py` with the defaults.
- Then run `python tools/data/split_check.py data/manifests/serengeti_trainval.jsonl data/manifests/kgalagadi_test.jsonl --build-info data/manifests/build_info.json`.
- Put the actual counts in the SUMMARY: per split, sites, day/night, zero-box, exclusions, KGA non-empty/empty, KGA sites covered out of total, and any shortfalls.
  </action>
  <verify>
    <automated>python -m pytest tests/data -q && python tools/data/build_manifests.py && python tools/data/split_check.py data/manifests/serengeti_trainval.jsonl data/manifests/kgalagadi_test.jsonl --build-info data/manifests/build_info.json && python -c "import json,collections;r=[json.loads(l) for l in open('data/manifests/serengeti_trainval.jsonl',encoding='utf-8')];k=[json.loads(l) for l in open('data/manifests/kgalagadi_test.jsonl',encoding='utf-8')];c=collections.Counter(x['split'] for x in r);assert c['train']==4000 and c['val']==400,c;st={s:{x['site_id'] for x in r if x['split']==s} for s in c};assert not st['train']&st['val'];assert all(x['split']=='test' and x['site_id'].startswith('KGA:') for x in k);print(c,collections.Counter(x['subset'] for x in k),len({x['site_id'] for x in k}))" && git check-ignore -q data/raw/snapshot_serengeti/bboxes.json.zip</automated>
  </verify>
  <done>The committed manifests exist with 4000 train and 400 val Serengeti rows from disjoint sites and sequences, plus a Kgalagadi test set covering all sites that have non-empty images, with actual counts reported. build_info.json has output_sha256 and split_check --build-info passes. Every leak-injection and contract test fails the gate as intended. The raw LILA zips are git-ignored.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 3: Downloader hardening — retries/fallback/resume, Kgalagadi without the full zip, grayscale/EXIF metadata, Colab runbook</name>
  <files>tools/data/download_images.py, tests/data/test_download_images.py, data/manifests/README.md</files>
  <read_first>tools/data/download_images.py, tools/data/split_check.py, tests/data/conftest.py, data/manifests/build_info.json</read_first>
  <behavior>
    - An HTTP 503 injected twice on a path: the third attempt succeeds and the record shows 3 attempts.
    - First mirror returns 404: the second mirror is used and recorded, with no retries spent on the 404.
    - A second run makes zero GET requests.
    - An existing file with no record is hashed and recorded (status "recovered") with no GET.
    - A stale .part file is replaced.
    - A non-JPEG payload, or a payload above max_bytes: the row fails, no final file and no record are left.
    - safe_join rejects "../x.jpg", "/abs.jpg" and "C:/x.jpg".
    - Zip strategy over HttpRangeFile (served locally with Range support, member prefix set): the selected member is written with a sha256 equal to its source bytes, and unselected members are never written.
    - An archive member named with a parent-directory path causes no write outside root, because the destination comes only from the manifest.
    - A grayscale JPEG gives is_grayscale True; a colored one gives False. A JPEG saved with EXIF DateTimeOriginal gives a populated exif_datetime and exif_status "present".
    - Leaking manifests (a shared site injected): run_download returns non-zero and no .list file exists.
    - --dry-run creates nothing under root.
    - One opt-in network test downloads the lowest-rank Serengeti row and Kgalagadi row from the committed manifests into tmp_path and checks for a JPEG with a recorded sha256.
  </behavior>
  <action>
Step A — probe once over the network, using only a handful of HEAD or Range requests:
- Take 2 real source_file_name values from data/manifests/kgalagadi_test.jsonl.
- Send HEAD requests to these candidates:
  - `https://storage.googleapis.com/public-datasets-lila/snapshot-safari/KGA/KGA_public/{file_name}`
  - `https://lilawildlife.blob.core.windows.net/lila-wildlife/snapshot-safari/KGA/KGA_public/{file_name}`
  - the same two URLs without the `KGA_public/` segment.
- If any candidate returns 200 with an image content type, the snapshot_kgalagadi strategy is "url" with those templates.
- Otherwise the strategy is "zip":
  - Open `https://storage.googleapis.com/public-datasets-lila/snapshot-safari/KGA/KGA_S1.lila.zip` through HttpRangeFile. This reads only the central directory.
  - Work out how member names relate to file_name (identical, or a fixed prefix) and set `KGA_ZIP_MEMBER_PREFIX`.
- Also confirm that the lowest-rank Serengeti row returns 200 on both mirrors.
- Record the probe results (URLs tried, status codes, chosen strategy, member mapping) in the SUMMARY and in a short comment next to the constants.

Step B — hardening. Keep the Task 1 signatures.
- Strategy selection:
  - `SOURCE_STRATEGY` maps source to "url" or "zip", seeded from the probe.
  - `--strategy SOURCE=url|zip` overrides it.
  - `--zip-url SOURCE=URL` sets the archive URL.
  - `--zip-local SOURCE=PATH` uses a zip already copied to Colab local disk instead of range reads.
- Retries:
  - Each URL gets up to `--retries` attempts (default 4). Backoff starts at `--backoff` seconds (default 2.0), doubles each attempt, and adds up to 0.5 s of random jitter. Tests pass 0.
  - Retry on URLError, timeouts, HTTP 429 and 5xx, and bodies shorter than Content-Length.
  - On 403/404, move straight to the next mirror.
  - Record the winning URL and the attempt count.
- Integrity:
  - A payload must start with the JPEG SOI bytes (0xFF 0xD8). If Pillow imports, it must also pass `Image.open(...).verify()`.
  - Otherwise delete the .part file and count the row as failed.
  - Always overwrite stale .part files.
- Resume:
  - Skip a row when its record exists and the file size matches.
  - If the file exists but has no record, hash it, validate it, and record it as "recovered" without fetching.
  - `--verify-existing` re-hashes recorded files and downloads mismatches again.
  - After each record, the main thread appends it to the checksums JSONL, flushes, and calls os.fsync, so a killed Colab session loses only the rows in flight.
- `HttpRangeFile(io.RawIOBase)`:
  - It is seekable and readable. A HEAD request gives the Content-Length.
  - readinto sends a GET with a `Range: bytes=a-b` header under the same retry policy.
  - Raise a clear error if the server answers a Range request with 200 instead of 206.
  - Wrap it in `io.BufferedReader(buffer_size=1 MiB)` and pass that to zipfile.ZipFile. zipfile reads zip64 and checks CRC.
- Zip strategy:
  - One ZipFile per worker thread, via threading.local.
  - Member name = prefix + source_file_name, read with `ZipFile.read`. Never extract the whole archive.
  - The destination always comes from `safe_join(root, relative_path)`, never from the member name. This is the zip-slip mitigation.
- Post-download metadata on each record, using Pillow when available; when Pillow is missing, the fields are None and a reason is recorded:
  - width and height;
  - is_grayscale: downscale to 64x64 RGB; the image counts as grayscale when the per-pixel mean of max(|R-G|, |G-B|, |R-B|) is at most 2.0. This is the IR night check;
  - exif_datetime: Exif IFD DateTimeOriginal (tag 36867), else DateTime (tag 306), else None;
  - exif_status: "present", "absent" or "unreadable".
- `--exif-report N` (default 100): after a run, print three things for the first N recorded rows in manifest order:
  - how many have an exif_datetime;
  - how many have a manifest datetime;
  - how many is_grayscale values disagree with the manifest's illumination proxy.
  This is the CAMERA_TRAP_FINE_TUNING_PLAN.md item-2 / DATA-05 check. The real numbers come from the Colab run.
- `run_download` summary adds recovered, bytes and by_mirror.

Step C — write tests/data/test_download_images.py first, covering every behavior above:
- Local server only: `http.server.ThreadingHTTPServer` on 127.0.0.1, port 0, in a daemon thread.
- The handler serves an in-memory dict of paths to bytes, answers Range with 206, injects 503 a set number of times per path, and counts requests.
- Build JPEGs with Pillow, including an EXIF-bearing one via `Image.Exif`.
- Build zips in memory with zipfile.
- Mark the one real-download test `network`.

Step D — write data/manifests/README.md (English, concise), covering:
- the field table: meaning and null policy for each REQUIRED_FIELDS entry;
- the file inventory, and why .gitattributes exists;
- split rules: site-disjoint, sequences never split, Kgalagadi test-only, SER:/KGA: prefixes, sample_rank semantics;
- the rebuild command and seed;
- the Colab runbook:
  - mount Drive;
  - run `python tools/data/download_images.py --manifest data/manifests/serengeti_trainval.jsonl --manifest data/manifests/kgalagadi_test.jsonl --root /content/drive/MyDrive/wild_diff_icmh/images --workers 16`;
  - to resume, rerun the same command;
  - `--limit 70` gives the ~210-image Week-1 slice (70 per manifest/split);
  - the zip strategy flags;
  - where the .list files land (`<root>/_lists`) and how to point a config `file_list` at them;
- the gate command to run before any train/eval job: split_check with `--list` and `--build-info`;
- known limits:
  - Serengeti zero-box is not a verified empty;
  - illumination is an hour-based proxy until is_grayscale exists;
  - Serengeti species is null;
  - Kgalagadi has no bboxes;
  - Kgalagadi replaces CCT as the held-out set by user decision;
  - the gate is not yet called from inside train.py/eval; that wiring is pending in the Phase 1 train.py work.

Step E — run the opt-in network test once: `WILD_DATA_NET_TESTS=1 python -m pytest tests/data -q -m network`. It writes only into a pytest temp dir. Record the outcome in the SUMMARY. If there is no network, record that and continue.

Step F — commit:
- Stage each path in this plan's files_modified by name. Nothing else: no data/raw zips, no __pycache__, no other untracked repo files.
- Run `git status --short` and confirm that only those paths are staged.
  </action>
  <verify>
    <automated>python -m pytest tests/data -q && python tools/data/download_images.py --manifest data/manifests/serengeti_trainval.jsonl --manifest data/manifests/kgalagadi_test.jsonl --root /content/drive/MyDrive/wild_diff_icmh/images --dry-run --limit 2 && python tools/data/split_check.py data/manifests/serengeti_trainval.jsonl data/manifests/kgalagadi_test.jsonl --build-info data/manifests/build_info.json</automated>
  </verify>
  <done>All tests/data tests pass locally with no network. The dry run prints Serengeti URLs and Kgalagadi URLs or zip members for 2 rows per manifest/split and creates nothing. The Kgalagadi retrieval strategy is chosen from a real probe and documented. The README gives the Colab runbook and gate command. Only the listed files are committed.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| LILA mirrors (GCS/Azure HTTP) -> downloader | Untrusted remote bytes, status codes and lengths are written to Colab/Drive disk |
| Manifest JSONL / LILA JSON -> filesystem paths | file_name/relative_path strings decide where files are written |
| Remote zip archive -> extractor | Member names and sizes inside a 10.56 GB third-party archive |
| Git checkout -> training/eval jobs | Committed manifests may be altered (CRLF conversion, manual edits) before a job reads them |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-sxk-01 | Tampering | download_images.safe_join / split_check relative_path rule | high | mitigate | Reject absolute, drive-letter and ".." paths at build time (split_check) and again at write time (safe_join); tests cover all three forms |
| T-sxk-02 | Tampering | zip strategy (HttpRangeFile + zipfile) | high | mitigate | Read only the selected member with ZipFile.read, never extract the whole archive; destination comes only from the manifest via safe_join; zipfile checks CRC; a test with a parent-directory member name proves nothing is written outside root |
| T-sxk-03 | Denial of Service | fetch_url | medium | mitigate | max_bytes cap (50 MiB) with chunked reads, per-request timeout, bounded retries with backoff; test with a small max_bytes |
| T-sxk-04 | Tampering | downloaded image payloads | medium | mitigate | JPEG SOI check plus Pillow verify before os.replace from .part; sha256 + bytes recorded; --verify-existing re-hashes. LILA publishes no per-image hashes, so the hash is trust-on-first-download (accepted residual) |
| T-sxk-05 | Tampering | committed manifests | high | mitigate | build_info.output_sha256 checked by split_check --build-info (fail fast); .gitattributes -text prevents CRLF rewriting under core.autocrlf=true |
| T-sxk-06 | Information Disclosure | image selection | medium | mitigate | Exclude Serengeti images with person/vehicle boxes and Kgalagadi images with human-type labels; counts recorded in build_info |
| T-sxk-07 | Spoofing | mirror URLs | low | mitigate | Default templates are https only; the --mirror-template/--zip-url overrides are operator-supplied CLI input (accepted, documented in README) |
| T-sxk-08 | Information Disclosure | git staging | medium | mitigate | data/raw/.gitignore ignores the LILA zips; stage only files_modified paths, each named explicitly; `git check-ignore` in Task 2 verify |
| T-sxk-SC | Tampering | package installs | low | accept | No packages are installed; only the standard library plus the already-present Pillow/pytest are used, so the package-legitimacy gate does not apply |
</threat_model>

<verification>
- `python -m pytest tests/data -q` passes on Windows CPU with no network (the network test is skipped).
- `python tools/data/split_check.py data/manifests/serengeti_trainval.jsonl data/manifests/kgalagadi_test.jsonl --build-info data/manifests/build_info.json` exits 0. The leak-injection tests prove it exits non-zero on a shared site or sequence.
- Rebuilding the manifests with seed 20260916 reproduces output_sha256 exactly (covered by the determinism test and by the real rebuild in Task 2 verify).
- The .list files produced by download_images.py are read by utils.file.load_file_list (e2e test).
- The SUMMARY records: the Kgalagadi JSON schema findings, the probe results and chosen Kgalagadi strategy, the actual manifest counts and shortfalls, and the network smoke-test outcome.
</verification>

<success_criteria>
- Frozen, reproducible manifests for Serengeti train/val (site- and sequence-disjoint, stratified by site and day/night) and the Kgalagadi fixed test set are committed with provenance (build_info.json).
- The split gate is importable and runnable, runs automatically at build time and before list generation, and is proven to block a deliberately injected shared site.
- A Colab-ready downloader is resumable, falls back between mirrors, is safe against path traversal and zip-slip, records sha256/EXIF/grayscale per image, and gets Kgalagadi images without the full 10.56 GB zip.
- No GPU was used, and no network was used in the default test run.
</success_criteria>

<output>
Create `.planning/quick/260916-sxk-buoc-1-fine-tune-camera-trap-dung-manife/260916-sxk-SUMMARY.md` when done.
</output>
