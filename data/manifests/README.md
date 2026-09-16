# data/manifests/

Frozen, reproducible data contract for the Snapshot Serengeti train/val
corpus and the Snapshot Kgalagadi fixed test set. This is the folder-based
handoff between TV-A (data) and TV-B (training): both people read these
files, nobody re-derives them by hand.

Built by `tools/data/build_manifests.py` from the official LILA JSONs,
gated by `tools/data/split_check.py`, and consumed by
`tools/data/download_images.py` (which fetches the actual image bytes and
writes `.list` files for the training dataloader).

## Files

| File | Contents |
|---|---|
| `serengeti_trainval.jsonl` | 4000 train + 400 val rows, site- and sequence-disjoint, day/night stratified |
| `kgalagadi_test.jsonl` | 400 nonempty + 50 empty_check rows, `split="test"` only, covers every Kgalagadi site that has a non-empty candidate |
| `build_info.json` | Provenance: source file sha256/version, seed, exclusion counts, per-split stats, `output_sha256` for each manifest |
| `.gitattributes` | `*.jsonl -text` / `*.json -text` so `core.autocrlf` never rewrites these files (would break the sha256 gate) |

## Row contract (`REQUIRED_FIELDS` in `tools/data/split_check.py`)

Every row has every field below. Any field whose value is `null` has a
matching entry in `null_reasons` explaining why — nothing is silently
guessed.

| Field | Meaning | Null policy |
|---|---|---|
| `image_id` | `"SER:" + id` or `"KGA:" + id` | never null |
| `source` | `snapshot_serengeti` or `snapshot_kgalagadi` | never null |
| `source_version` | Source JSON's version string (e.g. `SnapshotSerengetiBboxes_20190903`) | never null |
| `license` | `CDLA-Permissive-1.0` for both sources (Serengeti: bbox JSON `info` block; Kgalagadi: stated on the LILA dataset page, its own JSON `info` block does not state one) | never null |
| `relative_path` | `<source>/<file_name>`, forward slashes, used by `download_images.py`/`safe_join` | never null |
| `source_file_name` | Original LILA `file_name`, used to build download URLs | never null |
| `site_id` | `"SER:" + location` or `"KGA:" + location` | never null |
| `sequence_id` | `"SER:" + seq_id` or `"KGA:" + seq_id` | never null |
| `frame_num` | Frame number within the sequence | null if missing in source |
| `datetime` | `"YYYY-MM-DD HH:MM:SS"` from the source JSON | null if missing in source |
| `illumination` | `"day"` (hour 6-18 inclusive) or `"night"`, derived from `datetime` | null if `datetime` is null/unparseable |
| `illumination_source` | Always `"datetime_hour_proxy"` | never null |
| `species` | Serengeti: always null (see below). Kgalagadi: sorted unique non-empty label names, `[]` when empty | Serengeti: species labels live in the 5.5 GB `SnapshotSerengeti_S1-11_v2.1.json`, intentionally not loaded |
| `is_empty` | Serengeti: always null (see below). Kgalagadi: `true` when the only label is `empty` | Serengeti: the 20190903 bbox file removed boxes under 400 px^2, so zero boxes does not prove an empty frame |
| `boxes` | Serengeti: list of `{"category", "bbox_xywh"}` (absolute pixels), `[]` when none. Kgalagadi: always null | Kgalagadi has no bbox ground truth; MegaDetector boxes are produced at eval time |
| `width`, `height` | From the source JSON | null if missing in source (then recorded from the downloaded file in the checksums file) |
| `split` | `train` / `val` (Serengeti) or `test` (Kgalagadi, always) | never null |
| `subset` | `bbox_subset` (Serengeti), `nonempty` / `empty_check` (Kgalagadi) | never null |
| `sample_rank` | 0-based pick order within `(manifest, split)`; a rank prefix stays site/stratum-diverse | never null |
| `sha256` | Checksum of the downloaded bytes | filled by `download_images.py` in the checksums file, null until then |
| `null_reasons` | `{field: reason}` for every null field above | n/a |

## Split rules

- Every Serengeti site belongs to exactly one of train/val; every sequence
  belongs to exactly one split; `max_per_sequence=1` by default, so no
  sequence contributes more than one frame.
- Serengeti images with a `person` or `vehicle` box, or a null/unparseable
  `datetime`, are excluded before sampling (counts in `build_info.json`
  under `excluded.snapshot_serengeti`).
- Kgalagadi is `split="test"` only — never used for train/val selection or
  hyperparameter choice. `KGA:`-prefixed site/sequence ids never collide
  with `SER:`-prefixed ones.
- `sample_rank` is assigned by interleaving strata (day/night for
  Serengeti, nonempty/empty_check for Kgalagadi) in proportion to their
  sizes, so any prefix of ranks (e.g. the `--limit` slice below) stays
  diverse instead of being all-day-then-all-night.

## Rebuild

```bash
python tools/data/build_manifests.py \
  --serengeti-bbox-json data/raw/snapshot_serengeti/bboxes.json.zip \
  --kgalagadi-json data/raw/snapshot_kgalagadi/SnapshotKgalagadi_S1_v1.0.json.zip \
  --out-dir data/manifests \
  --seed 20260916
```

The same seed reproduces byte-identical `.jsonl` files (`write_jsonl` sorts
rows deterministically and writes with `\n` newlines regardless of OS).
`build_manifests.py` calls `split_check.assert_no_leakage` on its own
output before returning 0, so a leaking build never gets written silently.

## Colab runbook

1. Mount Drive.
2. Run the downloader (resumable — rerunning the same command only fetches
   what is missing or changed):

   ```bash
   python tools/data/download_images.py \
     --manifest data/manifests/serengeti_trainval.jsonl \
     --manifest data/manifests/kgalagadi_test.jsonl \
     --root /content/drive/MyDrive/wild_diff_icmh/images \
     --workers 16
   ```

3. `--limit 70` gives the ~210-image Week-1 slice (70 rows per
   manifest/split, taken from the lowest `sample_rank`, which is already
   site/stratum-diverse):

   ```bash
   python tools/data/download_images.py \
     --manifest data/manifests/serengeti_trainval.jsonl \
     --manifest data/manifests/kgalagadi_test.jsonl \
     --root /content/drive/MyDrive/wild_diff_icmh/images \
     --workers 16 --limit 70
   ```

4. Retrieval strategy (probed live 2026-09-16, see `PLAN.md` Task 3 Step
   A): both sources default to per-image URL fetches
   (`SOURCE_STRATEGY = {"snapshot_serengeti": "url", "snapshot_kgalagadi":
   "url"}`) — Kgalagadi images are served individually under a
   `KGA_public/` prefix on both LILA mirrors, so the 10.56 GB
   `KGA_S1.lila.zip` season archive is never downloaded in the default
   path. The zip strategy (`HttpRangeFile` + range-read `zipfile`, reading
   only the selected member) is implemented and tested as a documented
   fallback — use it with:

   ```bash
   --strategy snapshot_kgalagadi=zip \
   --zip-url snapshot_kgalagadi=https://storage.googleapis.com/public-datasets-lila/snapshot-safari/KGA/KGA_S1.lila.zip
   ```

   The zip member-name-to-`file_name` mapping inside `KGA_S1.lila.zip` was
   never probed live (the URL strategy already returned 200, so the zip
   path was not needed) — `KGA_ZIP_MEMBER_PREFIX` in `download_images.py`
   is an unverified best guess; confirm the real member names before
   relying on `--strategy zip` for Kgalagadi.

5. `.list` files land at `<root>/_lists/<manifest-stem>.<split>.list`
   (e.g. `serengeti_trainval.train.list`, `kgalagadi_test.test.list`), one
   absolute image path per line. Point a training config's `file_list` at
   these; `utils/file.py:load_file_list` reads them directly (it does not
   expand `$DATA_DIR`, so the paths must already be absolute — the
   downloader writes them that way).
6. Run the gate before any train/eval job:

   ```bash
   python tools/data/split_check.py \
     data/manifests/serengeti_trainval.jsonl data/manifests/kgalagadi_test.jsonl \
     --list train=<root>/_lists/serengeti_trainval.train.list \
     --list val=<root>/_lists/serengeti_trainval.val.list \
     --build-info data/manifests/build_info.json
   ```

   `--exif-report N` (default 100, printed at the end of `download_images.py`)
   compares the downloaded EXIF `DateTimeOriginal`/`DateTime` against the
   manifest `datetime` and the `illumination`/`is_grayscale` proxies over
   the first N rows in manifest order — this is the Week-1 EXIF check
   (CAMERA_TRAP_FINE_TUNING_PLAN.md item 2 / DATA-05).

## Known limits

- Serengeti `boxes: []` is not a verified empty frame — the 20190903 bbox
  file manually removed sub-400px^2 boxes, so a zero-box image may still
  contain an animal. `is_empty` is left null for Serengeti for this reason.
- `illumination` is an hour-of-day proxy from `datetime`, not a measured
  property, until `is_grayscale` (from the downloaded IR/RGB image itself)
  is available in the checksums file.
- Serengeti `species` is always null (the 5.5 GB species JSON is
  intentionally not loaded).
- Kgalagadi has no bounding-box ground truth; MegaDetector boxes are
  produced at eval time, not stored in the manifest.
- Kgalagadi replaces Caltech Camera Traps (CCT) as the held-out test set,
  per user decision. `REQUIREMENTS.md` DATA-01 still names CCT; updating
  that wording is a Phase 2 concern, not part of this manifest build.
- The gate is not yet called automatically from inside `train.py`/eval —
  that wiring is pending in the separate Phase 1 `train.py` rework
  (Lightning 2.x migration). Run `split_check.py` manually (see above)
  before any train/eval job until that hook lands.
