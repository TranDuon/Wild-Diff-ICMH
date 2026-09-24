# Camera-trap manifests

`kgalagadi_site_split.jsonl` is the primary experiment contract. It contains
all 10,222 non-human Snapshot Kgalagadi images used by the comparison paper.
The 135 images labelled as human are excluded.

The paper does not publish a train/validation/test ratio, so this project fixes
the following reproducible protocol:

- 70% train, 15% validation, 15% test by **sequence**, separately inside each
  of the 20 camera sites;
- every frame from one trigger sequence remains in exactly one split;
- each site appears in all three splits, because a separate codec is fine-tuned
  and evaluated for each site, matching the comparison paper's setup;
- seed `20260916`; strata are day/night and empty/non-empty where possible.

Actual frozen counts are:

| Split | Images | Sequences | Sites |
|---|---:|---:|---:|
| train | 7,191 | 2,497 | 20 |
| validation | 1,499 | 531 | 20 |
| test | 1,532 | 538 | 20 |
| total | 10,222 | 3,566 | 20 |

`build_info.json` records the source checksum, exclusions, split statistics and
manifest SHA-256. `split_check.py` verifies the hash and prevents sequence
leakage before training starts. The older `kgalagadi_test.jsonl` and
`serengeti_trainval.jsonl` files are retained only for historical
reproducibility; they are not the primary H1/H2/H3 protocol.

## Row contract

Important fields are `image_id`, `relative_path`, `site_id`, `sequence_id`,
`datetime`, `illumination`, `species`, `is_empty`, `split`, and `sample_rank`.
Kgalagadi has no ground-truth boxes, so `boxes` is null; H2 and foreground SSIM
read MegaDetector boxes from a separate JSON/JSONL sidecar.

## Rebuild and verify

```bash
python tools/data/build_manifests.py \
  --kgalagadi-json data/raw/snapshot_kgalagadi/SnapshotKgalagadi_S1_v1.0.json.zip \
  --out-dir data/manifests \
  --seed 20260916 \
  --kga-train-fraction 0.70 \
  --kga-val-fraction 0.15 \
  --kga-test-fraction 0.15 \
  --skip-serengeti

python tools/data/split_check.py \
  data/manifests/kgalagadi_site_split.jsonl \
  --build-info data/manifests/build_info.json
```

## Download images

The downloader is resumable and downloads individual public LILA files, not
the whole season archive:

```bash
python tools/data/download_images.py \
  --manifest data/manifests/kgalagadi_site_split.jsonl \
  --root /content/drive/MyDrive/wild_diff_icmh/images \
  --workers 16
```

For training, copy the images from Drive to Colab's local SSD first. Reading
thousands of small JPEGs directly from mounted Drive is much slower and less
reliable.

## Evaluation role of Serengeti

Snapshot Serengeti is supplemental cross-dataset evaluation only. It must not
be used to select a Kgalagadi checkpoint, ROI weight, prompt, or training
schedule. The frozen Kgalagadi validation split is the only model-selection
split; the Kgalagadi test split is opened for final reporting.

## Known limitations

- Kgalagadi provides species/empty labels but no bounding boxes. MegaDetector
  boxes are pseudo-labels, not ground truth, and this must be stated in the
  report.
- `illumination` is inferred from the annotation timestamp. H3 transmits
  illumination plus Southern-Hemisphere season in one byte per image and
  includes that byte in the measured bitstream size.
- The comparison paper does not disclose its split. Results are therefore
  comparable in dataset and per-site training protocol, but not claimed to use
  the authors' undisclosed exact split.
