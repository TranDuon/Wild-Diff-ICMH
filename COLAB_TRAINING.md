# Lưu ý vận hành Colab

Notebook: `Wild_Diff_ICMH_Kgalagadi_Train.ipynb` (sinh từ `tools/build_colab_training_notebook.py`;
muốn sửa cell thì sửa file sinh rồi chạy lại nó, không sửa tay notebook).

File này **chỉ chứa lưu ý vận hành**. Giao thức thí nghiệm, phạm vi, ngân sách và tiêu chí
nghiệm thu nằm duy nhất trong khung GSD: `.planning/PROJECT.md`, `.planning/REQUIREMENTS.md`,
`.planning/ROADMAP.md`, `.planning/STATE.md`.

## Đồng bộ code

- Sửa code ở máy local → `git commit` → `git push origin main`. Không tải ZIP code lên Colab.
- Trong runtime đang chạy, khi có bản sửa mới: chạy **Bước 2A** rồi chạy lại đúng cell vừa lỗi.
  Bước 2A từ chối pull nếu thư mục `/content/Wild-Diff-ICMH` có sửa cục bộ — không sửa file
  trong đó.
- Sửa nội dung một cell ngay trên trình duyệt (ví dụ đổi tên run dir) không ảnh hưởng repo,
  nhưng sẽ mất khi mở lại notebook từ GitHub.
- Không commit dữ liệu, checkpoint, log hay kết quả vào Git.

## Cấu trúc Drive

```text
MyDrive/wild_diff_icmh/
  images/snapshot_kgalagadi/...        # ảnh gốc, chép sang /content/data mỗi runtime
  checkpoints/
    sd2p1/v2-1_512-ema-pruned.ckpt
    ram/ram_plus_swin_large_14m.pth
    difficmh_models/CNscale1.0_1_1_<BPP_WEIGHT>_2_.../model.ckpt
  tags/KGA_<site>.jsonl                # cache RAM++ (Bước 6), resumable
  detections/kgalagadi_megadetector.json
  runs/
    h1_v2/<site>/{config.yaml, config_model.yaml, checkpoints/last.ckpt, best.ckpt}
    h1_control/<site>/...
    h2/<site>/...
  logs/                                # log đầy đủ của mọi subprocess
  results/results.jsonl                # registry kết quả duy nhất
```

- `/content` bị xoá khi runtime ngắt; Drive thì không. Ảnh được chép sang SSD `/content/data`
  vì đọc hàng nghìn file nhỏ trực tiếp từ Drive rất chậm.
- **Không dùng `runs/h1/`**: checkpoint ở đó tạo trước bản vá entropy-bottleneck (commit
  a385eee), entropy model khởi tạo ngẫu nhiên. Resume sẽ bị chặn vì thiếu contract version.

## Cài đặt (Bước 3)

- Không cài lại `torch`, `torchvision` hay xFormers. Attention dùng `scaled_dot_product_attention`
  của PyTorch 2 nên không cần xFormers.
- Bước 3 bỏ pin NumPy/SciPy, ghim phiên bản NumPy/SciPy/Torch/Torchvision hiện có bằng
  constraints và dừng nếu chúng bị đổi sau khi cài.
- CompressAI cài bằng `--no-deps --no-build-isolation`; vì vậy `requirements-colab.txt` phải
  liệt kê đủ phụ thuộc không-phải-Torch của nó (`pytorch-msssim`, `torch-geometric`, ...).
  Trên Python 3.13 CompressAI được build từ source.
- RAM++ (`src/recognize-anything`) cài dạng package thường, không editable, để kernel đang chạy
  thấy ngay. **Sửa code trong `src/recognize-anything` thì phải chạy lại Bước 3.**
- Giữ `transformers<5` (BERT của RAM++ dùng API 4.x).

## RAM++ tags (Bước 6)

- Tag được tính một lần rồi cache trên Drive; training đọc cache thay vì giữ RAM++ (~3 GB) trên GPU.
- Cell tự bỏ qua nếu đủ tag, hoặc tiếp tục phần còn thiếu; dòng JSONL bị cắt cụt do runtime chết
  được tự sửa.

## Training, checkpoint và resume (Bước 7–8)

- `--init-checkpoint` = chỉ nạp trọng số tác giả (step về 0). `--resume` / `resume_checkpoint: auto`
  = khôi phục toàn trạng thái từ checkpoint của dự án. Có checkpoint resume thì init bị bỏ qua.
- `BPP_WEIGHT` phải trùng thư mục checkpoint tác giả đã chọn (`CNscale1.0_1_1_<BPP_WEIGHT>_2_...`),
  và config phải có `control_model_ratio: 1.0` (CNscale1.0). `train.py` chặn nếu lệch.
- Checkpoint tác giả dùng tên tham số entropy model của CompressAI cũ (`_matrixN`, `_biasN`,
  `_factorN`); loader tự đổi sang `matrices.N`, `biases.N`, `factors.N` và dừng nếu thiếu khóa
  hoặc xung đột. Log phải có `Migrated 14 legacy CompressAI entropy-bottleneck keys`.
  **Không được "sửa" bằng `strict=False`.**
- Nhịp checkpoint: rolling mỗi 50 optimizer step + `last.ckpt` cuối mỗi lượt `trainer.fit`
  (ghi `.part`, kiểm tra, rồi mới thay). Khi resume, checkpoint được chép về đĩa local trước khi
  nạp; file hỏng bị bỏ qua và dùng file hợp lệ cũ hơn.
- **Chạy lại Bước 7 vào run dir đã có checkpoint ≥ `max_steps`** thì Lightning dừng ngay, không
  train gì. Muốn đo lại thì dùng run dir mới.
- Bước 8 resume đúng checkpoint của Bước 7 (`--resume PROJECT_CKPT`) và yêu cầu step tăng.
  Nếu báo checkpoint hỏng: chạy lại Bước 7 một lần rồi mới chạy Bước 8.
- `tools/train_kgalagadi_sites.py --max-sites N` chạy tuần tự nhiều site và bỏ qua site đã có
  `best.ckpt`; giữ `--max-sites 1` cho tới khi biết một site tốn bao nhiêu CU.

## H2 và H3

- H2 cần file MegaDetector JSON/JSONL (`images[].file`, `images[].detections`) có **một record
  cho mọi ảnh của site**, kể cả `detections: []`; preflight chặn nếu thiếu. Không giả lập ROI.
- H2 và H1-control khởi động từ checkpoint H1 qua
  `--init-checkpoint-template '.../runs/h1_v2/{site}/checkpoints/best.ckpt'`.
- H3 không train. Decode cùng checkpoint/ảnh/seed hai lần: không có và có
  `--tag-vocabulary data/vocab/kgalagadi_wildlife_tags.txt --domain-metadata`.
- `--habitat-map` chỉ dùng với bảng site → habitat đã kiểm chứng (JSON `{"KGA:A01": "..."}`);
  không tự đoán habitat. Bitstream H3 không có version marker — decode phải dùng đúng cờ
  tag/metadata như lúc encode.

## Decode và đánh giá (Bước 9)

- `inference_partition.py`: truyền `--config <run>/config_model.yaml` (tự chọn nếu file nằm
  cạnh checkpoint) để dựng đúng kiến trúc lúc train.
- **Khi có `--manifest` mà không có `--crop-size`, decode và evaluate tự cắt ô 256×256 ở giữa
  ảnh** (để tránh hết VRAM: ảnh 2592×2000 cần ~26 GiB cho attention của VAE). Số liệu khi đó chỉ
  đại diện ~1,3% diện tích ảnh. Số báo cáo phải theo giao thức end-to-end ở độ phân giải gốc
  (REQUIREMENTS EVAL-11), sẽ được thêm vào code ở Phase 2.
- `tools/evaluate_kgalagadi.py` phải dùng cùng `--crop-size` với lúc decode; tính bpp từ kích
  thước file bitstream thật; `--results-registry` ghi/thay dòng cùng `exp_id` (idempotent).
- Bước 9 chỉ decode 2 ảnh với 5 bước DDIM: là kiểm tra end-to-end, **không phải số báo cáo**.

## Đóng Phase 1 (Bước 10)

- Bước 10 dùng biến trong bộ nhớ của Bước 7–9 (`smoke_hours`, `PROJECT_CKPT`, `resumed_step`,
  `decode_cu_estimate`, ...). **Runtime mới thì phải chạy lại Bước 1–9 trước**, không chạy riêng
  Bước 10 được.
- Dự báo chi phí 2K step trong closeout lấy tổng thời gian Bước 7 chia 20 step, gồm cả thời gian
  dựng model và validation — con số bị phóng đại. Tốc độ ổn định xem ở `it/s` trong
  `logs/train_smoke_*.log` (1 optimizer step = 8 batch).

## Compute unit

- Hằng số `COLAB_CU_PER_HOUR = 1.54` trong notebook chỉ để ước lượng. CU thật = chênh lệch
  "Available" trong Colab Resources trước và sau phiên; ghi lại mỗi phiên.
- Runtime GPU tốn CU cả khi đang cài đặt hoặc chép ảnh; ngắt runtime ngay khi xong việc.
- Chọn L4 thủ công (quyết định D-04); notebook không có nhánh riêng cho T4/A100.
