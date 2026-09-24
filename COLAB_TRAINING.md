# Fine-tune Kgalagadi trên Google Colab

Notebook sẵn dùng: `Wild_Diff_ICMH_Kgalagadi_Train.ipynb`.

## Protocol đã khóa

- Dataset chính: toàn bộ 10.222 ảnh Snapshot Kgalagadi không có người.
- Chia 70/15/15 theo sequence bên trong từng địa điểm; không tách các frame
  cùng một lần kích hoạt camera sang nhiều tập.
- Fine-tune một model riêng cho mỗi trong 20 địa điểm.
- Chọn checkpoint/siêu tham số bằng validation Kgalagadi; chỉ mở test để báo
  cáo cuối. Serengeti chỉ là đánh giá bổ sung.
- Thứ tự thí nghiệm: baseline checkpoint tác giả → H1 fine-tune thuần → H2
  từ chính checkpoint H1. H3 chỉ đổi tag/metadata lúc encode-decode trên cùng
  checkpoint H1 hoặc H2, không chạy thêm training.

## Đồng bộ code hiệu quả

Không tải ZIP code lên lại mỗi lần. Ở máy local:

```bash
git add <các-file-muốn-lưu>
git commit -m "mô tả thay đổi"
git push origin main
```

Trong Colab chỉ cần chạy cell clone/pull. Dữ liệu và checkpoint nằm ở Drive,
code nằm ở `/content` để đọc nhanh. Không commit dữ liệu hay checkpoint vào Git.

## Cấu trúc Drive

```text
MyDrive/wild_diff_icmh/
  images/snapshot_kgalagadi/KGA_S1/...
  checkpoints/
    sd2p1/v2-1_512-ema-pruned.ckpt
    ram/ram_plus_swin_large_14m.pth
    difficmh_models/.../model.ckpt
  detections/kgalagadi_megadetector.json
  runs/
    h1/A01/checkpoints/last.ckpt
    h1/A01/checkpoints/best.ckpt
    h1_control/A01/checkpoints/best.ckpt
    h2/A01/checkpoints/last.ckpt
```

Mỗi đầu session, copy ảnh cần dùng từ Drive sang SSD `/content/data`. Checkpoint
được lưu thẳng về Drive để mất runtime vẫn resume được.

## Cài đặt không phá Torch của Colab

```bash
pip install -q -r requirements-colab.txt
pip install -q --no-deps compressai==1.2.8
pip install -q --no-deps -e src/recognize-anything
```

Không cài lại `torch`, `torchvision` hoặc xFormers bằng một wheel tùy ý. Code
đã dùng `scaled_dot_product_attention` có sẵn trong PyTorch 2 nên xFormers
không còn là điều kiện bắt buộc.

## Chạy một địa điểm trước

RAM++ được chạy một lần trước training rồi cache tag. Việc này giữ nguyên text
conditioning của checkpoint tác giả nhưng giải phóng khoảng 3 GB VRAM khi
fine-tune:

```bash
python tools/precompute_ram_tags.py \
  --data-root /content/data/wild_diff_icmh/images \
  --checkpoint /content/drive/MyDrive/wild_diff_icmh/checkpoints/ram/ram_plus_swin_large_14m.pth \
  --site-id KGA:A01 \
  --output /content/drive/MyDrive/wild_diff_icmh/tags/KGA_A01.jsonl
```

Sau đó chạy smoke test 20 step trên `KGA:A01`:

```bash
export WILD_DATA_ROOT=/content/data/wild_diff_icmh/images
export KGA_SITE_ID=KGA:A01
export KGA_TAGS=/content/drive/MyDrive/wild_diff_icmh/tags/KGA_A01.jsonl
export BPP_WEIGHT=2
export WILD_RUN_DIR=/content/drive/MyDrive/wild_diff_icmh/runs/h1/A01

python train.py --config configs/train_kgalagadi_colab.yaml \
  --init-checkpoint /content/drive/MyDrive/wild_diff_icmh/checkpoints/difficmh_models/CNscale1.0_1_1_2_2_WTagGCM_bs16x1_lr0.00005_cfg7.0/model.ckpt \
  lightning.trainer.max_steps=20 \
  lightning.trainer.val_check_interval=10 \
  lightning.trainer.check_val_every_n_epoch=1 \
  lightning.trainer.limit_val_batches=2
```

Nếu runtime ngắt, chạy lại đúng lệnh. `resume_checkpoint: auto` sẽ nạp
`last.ckpt`, bao gồm model, optimizer và global step. Khi đã đo được thời gian
và VRAM, bỏ bốn override smoke-test; không nên chạy cả 20 site trước phép đo này.

Để chạy tuần tự một số site trong một session:

```bash
python tools/train_kgalagadi_sites.py \
  --config configs/train_kgalagadi_colab.yaml \
  --run-root /content/drive/MyDrive/wild_diff_icmh/runs/h1 \
  --init-checkpoint /content/drive/MyDrive/wild_diff_icmh/checkpoints/difficmh_models/CNscale1.0_1_1_2_2_WTagGCM_bs16x1_lr0.00005_cfg7.0/model.ckpt \
  --max-sites 1
```

`--max-sites 1` là van an toàn. Chỉ tăng sau khi biết một site tốn bao nhiêu
phút và compute unit. Lần chạy sau tự bỏ qua site đã có `best.ckpt`, nên
`--max-sites 1` sẽ chuyển sang site chưa hoàn thành tiếp theo.

`BPP_WEIGHT` phải luôn trùng checkpoint tác giả đã chọn. Làm pilot với 2 trước;
để dựng RD curve tối thiểu lặp protocol cho 2, 8 và 32. Chỉ mở rộng 4 và 16 sau
khi dự báo tổng compute vẫn nằm trong ngân sách.

## H2 và H3

H2 cần MegaDetector JSON/JSONL theo format chuẩn (`images[].file` và
`images[].detections`). File phải có một record cho mọi ảnh của site, kể cả
record có `detections: []`; preflight sẽ chặn nếu thiếu. Không có file này thì
không được giả lập ROI:

```bash
export KGA_DETECTIONS=/content/drive/MyDrive/wild_diff_icmh/detections/kgalagadi_megadetector.json
python tools/train_kgalagadi_sites.py \
  --config configs/train_kgalagadi_h2.yaml \
  --run-root /content/drive/MyDrive/wild_diff_icmh/runs/h2 \
  --init-checkpoint-template '/content/drive/MyDrive/wild_diff_icmh/runs/h1/{site}/checkpoints/best.ckpt' \
  --max-sites 1
```

Để kết luận H2 có tác dụng, phải chạy thêm đối chứng `h1_control` từ đúng
checkpoint H1, cũng 2 epoch nhưng giữ loss đều. So sánh H2 với đối chứng này,
không chỉ so với H1 trước khi train thêm:

```bash
python tools/train_kgalagadi_sites.py \
  --config configs/train_kgalagadi_h1_control.yaml \
  --run-root /content/drive/MyDrive/wild_diff_icmh/runs/h1_control \
  --init-checkpoint-template '/content/drive/MyDrive/wild_diff_icmh/runs/h1/{site}/checkpoints/best.ckpt' \
  --max-sites 1
```

H2 mặc định chỉ weight `L_dist` (V1) để tiết kiệm compute. Chỉ thử V2 weight
thêm `L_sem` tại encoder layer 9 nếu V1 đã có tín hiệu; V2 cần hai lượt trích
feature UNet nên đắt hơn rõ rệt.

H3 dùng chính checkpoint H1/H2, không fine-tune thêm. Nó thu vocabulary RAM++
còn 37 tag (6 bit/tag), nối metadata ngày/đêm và mùa từ manifest, rồi truyền
một byte metadata/ảnh. Site được biết từ chính model per-site đang giải mã nên
không tốn thêm bit. Nếu có bảng habitat đã kiểm chứng, truyền thêm
`--habitat-map /đường/dẫn/site_habitat.json` (JSON dạng `{"KGA:A01": "..."}`);
không được tự đoán habitat. Toàn bộ header, tag và metadata đều được tính vào BPP:

Trước hết decode H1 với tag RAM++ đầy đủ đã cache (không có metadata H3):

```bash
python inference_partition.py \
  --ckpt_sd checkpoints/sd2p1/v2-1_512-ema-pruned.ckpt \
  --ckpt_lc /content/drive/MyDrive/wild_diff_icmh/runs/h1/A01/checkpoints/best.ckpt \
  --config configs/model/diffeic.yaml \
  --input /content/data/wild_diff_icmh/images \
  --output /content/results/h1_A01 \
  --manifest data/manifests/kgalagadi_site_split.jsonl \
  --tag-cache "$KGA_TAGS" \
  --split test --site-id KGA:A01 --steps 50 --device cuda \
  params.c_cfg_scale=3.0
```

Sau đó decode lại đúng checkpoint/ảnh/seed nhưng bật hai tầng H3:

```bash
python inference_partition.py \
  --ckpt_sd checkpoints/sd2p1/v2-1_512-ema-pruned.ckpt \
  --ckpt_lc /content/drive/MyDrive/wild_diff_icmh/runs/h1/A01/checkpoints/best.ckpt \
  --config configs/model/diffeic.yaml \
  --input /content/data/wild_diff_icmh/images \
  --output /content/results/h3_A01 \
  --manifest data/manifests/kgalagadi_site_split.jsonl \
  --tag-cache "$KGA_TAGS" \
  --tag-vocabulary data/vocab/kgalagadi_wildlife_tags.txt \
  --domain-metadata \
  --split test --site-id KGA:A01 --steps 50 --device cuda \
  params.c_cfg_scale=3.0
```

## Đánh giá đúng với bài so sánh

Sau khi decode test, dùng bitstream thật và kích thước ảnh trước padding:

```bash
python tools/evaluate_kgalagadi.py \
  --manifest data/manifests/kgalagadi_site_split.jsonl \
  --data-root /content/data/wild_diff_icmh/images \
  --reconstruction-root /content/results/h1_A01 \
  --detections "$KGA_DETECTIONS" \
  --site-id KGA:A01 \
  --method H1 \
  --output /content/drive/MyDrive/wild_diff_icmh/results/h1_A01.jsonl
```

Kết quả gồm BPP, tỉ lệ nén RGB 24-bit, PSNR, SSIM và foreground SSIM. Muốn có
đường RD như bài so sánh phải lặp cùng protocol cho nhiều checkpoint BPP_WEIGHT
(2, 4, 8, 16, 32); không kết luận hơn/kém từ một điểm duy nhất.

## Quy tắc tiết kiệm tài nguyên

- Dùng L4 mặc định, crop 256, batch 1 và gradient accumulation 8.
- H1 mặc định 5 epoch/site; H2 và H1-control cùng 2 epoch/site. Đây là ngân
  sách khởi đầu, chỉ tăng nếu learning curve và compute unit của pilot cho phép.
- Validation chỉ 4 batch, sample 20 bước mỗi 2 epoch; checkpoint mỗi 50
  optimizer step và luôn lưu `last.ckpt`.
- Checkpoint dự án chỉ lưu phần trainable/control/codec; SD, VAE và RAM++ đóng
  băng không bị nhân bản vào mỗi file.
- Nếu phải dùng T4, override `lightning.trainer.precision=16-mixed`.
- Chỉ dùng A100 cho run cuối khi phép đo cho thấy tốc độ bù được compute unit.
