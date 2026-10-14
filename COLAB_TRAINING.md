# Lưu ý vận hành Colab

**Phase 3 H1 mới:** dùng `Wild_Diff_ICMH_Phase3_H1.ipynb` trên nhánh `main`.
Thứ tự và giới hạn ngân sách tại [docs/PHASE3_H1_COLAB.md](docs/PHASE3_H1_COLAB.md).
Pilot λ2/500 step, đánh giá dev30; mở rộng và full dev tắt mặc định.
Cell2A pull cố định cập nhật script; không chạy lại Phase2 đã hoàn thành.

Notebook:
- `Wild_Diff_ICMH_Kgalagadi_Train.ipynb` — training (sinh từ `tools/build_colab_training_notebook.py`);
- `Wild_Diff_ICMH_Phase2_Eval.ipynb` — Phase 2: nhãn ngày/đêm, tập dev, MegaDetector, B0, baseline,
  chấm điểm (sinh từ `tools/build_colab_phase2_notebook.py`, dùng lại Bước 1–5 của notebook training).

Muốn sửa cell thì sửa file sinh rồi chạy lại nó, không sửa tay notebook.

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
    phase1_calib/<site>/<YYYYmmdd-HHMMSS>/   # smoke Bước 7–8: checkpoint + throughput.json
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
- Bước 7 luôn tạo run dir mới `runs/phase1_calib/<site>/<giờ chạy>` vì chạy lại vào run dir đã
  có checkpoint ≥ `max_steps` thì Lightning dừng ngay, không train gì. Lượt train thật (H1/H2)
  dùng run dir cố định để tự resume.
- `ThroughputMonitor` (trong `train_kgalagadi_colab.yaml`) ghi `throughput.json` vào run dir:
  median giây/batch × `accumulate_grad_batches`, bỏ 8 batch khởi động và thời gian validation.
  Đây là số dùng để tính ngân sách, không dùng tổng thời gian cell.
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
- **Giao thức báo cáo (EVAL-11, từ Phase 2):** khi có `--manifest` mà không có `--crop-size`,
  decode thu nhỏ ảnh về cạnh dài 1024 (`--processing-long-side`), nén, giải mã, rồi phóng ảnh tái
  tạo về **kích thước gốc** 2592×2000 trước khi lưu; bpp tính trên số pixel ảnh gốc. Ảnh gốc đầy đủ
  cần ~26 GiB VRAM nên không nén thẳng ở độ phân giải gốc.
- `--crop-size 256` chỉ còn cho smoke test (Bước 9), không dùng cho số báo cáo; không dùng chung
  với `--processing-long-side`.
- Mỗi thư mục output decode là một kho lưu trữ (EVAL-12): `<ảnh>.png` (kích thước gốc),
  `data/<ảnh>` (bitstream), `decode_log.jsonl` (byte, kích thước, thời gian encode/decode từng ảnh)
  và `run_info.json` (giao thức, checkpoint, seed, commit). Phiên Colab bị ngắt thì chạy lại cùng lệnh
  với `--skip-existing`; mỗi ảnh có seed riêng nên kết quả không phụ thuộc ảnh nào đã bỏ qua.
- `tools/evaluate_kgalagadi.py` chỉ cắt khi có `--crop-size` (phải trùng lúc decode); mặc định so
  với ảnh gốc; tính bpp từ kích
  thước file bitstream thật; `--results-registry` ghi/thay dòng cùng `exp_id` (idempotent).
- Bước 9 chỉ decode 2 ảnh với 5 bước DDIM: là kiểm tra end-to-end, **không phải số báo cáo**.

## Đóng Phase 1 (Bước 10)

- Bước 10 dùng biến trong bộ nhớ của Bước 7–9 (`PROJECT_CKPT`, `STEADY_SECONDS_PER_STEP`,
  `resumed_step`, ...). **Runtime mới thì phải chạy lại Bước 1–9 trước**, không chạy riêng
  Bước 10 được.
- Điền `CU_AVAILABLE_AT_START` ở Bước 1 và `CU_AVAILABLE_NOW` ở Bước 10 (số "Available" trong
  Runtime → View resources). Thiếu một trong hai thì Bước 10 dừng. Quên điền ở Bước 1 thì gán
  lại ngay trong cell Bước 10 (dùng số đã ghi lúc bắt đầu phiên).
- Dự báo 2K step dùng tốc độ ổn định từ `throughput.json`; báo cáo vẫn ghi kèm số chia theo tổng
  thời gian (`wall_clock_seconds_per_step`) để đối chiếu.
- Gửi lại: `results/phase1_metadata_<site>.json`, `results/phase1_closeout_<site>.json` và
  `throughput.json` trong run dir.

## Phase 2 — hai notebook, mỗi notebook một luồng chạy từ trên xuống

- `Wild_Diff_ICMH_Phase2_Prepare.ipynb` (**đã xong 05–06/10**, chỉ để tái lập): nhãn ngày/đêm, tập dev,
  MegaDetector trên 10.222 ảnh gốc, thống kê miền, RAM++ tags cho 20 site, decode B0. Cần đủ ảnh và
  checkpoint (Bước 4–5). P2-7 (decode B0) chỉ chạy khi đặt `RUN_B0 = True`.
- `Wild_Diff_ICMH_Phase2_Eval.ipynb` (**chạy bây giờ**): Bước 1 → 11, không bỏ cell nào. Chỉ chép 300 ảnh dev,
  kiểm tra đầu ra của Prepare và in số ảnh B0 đã decode, rồi baseline → chấm điểm → đồ thị RD → tóm tắt → CU.
- Output ở `MyDrive/wild_diff_icmh/phase2/` (sidecar ngày/đêm, `kgalagadi_dev.txt`, `detections/`,
  `archive/B0_*`, `bitstreams/`, `eval/`, `rd_dev.png`, `sessions.jsonl`). Không ghi gì vào thư mục repo
  trên Colab.
- **Không ghi ảnh baseline lên Drive** (ghi hàng chục GB làm Colab bị chặn "Google Drive quota exceeded"):
  ảnh baseline nằm ở `/content/p2_baselines`, chấm xong thì chỉ gói bitstream thành `.tar` lên Drive.
- MegaDetector chạy trong venv riêng `/content/envs/detect` (virtualenv, `--system-site-packages`,
  `huggingface-hub` ghim theo bản của Colab), không cài vào môi trường codec.
- Runtime ngắt giữa chừng: chạy lại từ Bước 1; điểm đã chấm không nén/chấm lại.

## Phase 3 H1 — chạy tiếp theo phiên (bản sửa 13/10/2026)

- Notebook chính: `Wild_Diff_ICMH_Phase3_H1.ipynb`. `SESSION_HOURS = 1.5` là
  **ngân sách training mới cho mỗi lần `trainer.fit`**, không cộng thời gian training
  của phiên trước. Chuẩn bị dữ liệu, dựng model, chép checkpoint và đánh giá nằm ngoài
  giới hạn này; chúng vẫn có thể tiêu tốn CU khi runtime GPU còn kết nối.
- Nguyên nhân lượt 500 → 1000 dừng ở 862: Timer mặc định của Lightning khôi phục
  thời gian đã chạy trong checkpoint, nên mốc 90 phút bị tính cộng dồn. Helper H1
  nay truyền `--session-time-limit`: chỉ đặt lại thời gian Timer, giữ nguyên trọng số,
  optimizer, scheduler, vòng lặp và tổng số bước. Không sửa file checkpoint gốc hay
  cấu hình đã đóng băng. Lệnh `train.py` không có cờ này vẫn giữ hành vi cũ.
- Runtime còn hoạt động: chạy **2A**, rồi cell **11** với `RUN_EXTENSION = True`,
  `EXTEND_TO = 1000` nếu quyết định hoàn tất mốc này. Runtime mới: mở notebook hiện tại,
  chọn L4, chạy **1 → 6**, rồi 11; không cần train lại 7/8. Số bước là **tổng 1000**,
  không phải thêm 1000. Log in rõ bước bắt đầu, mục tiêu và thời gian phiên mới.
- Pull không thay các cell đang mở. Để dùng wrapper log mới, mở lại notebook từ
  GitHub/Colab; không cần chép đoạn sửa tạm. Helper in tiến trình chuẩn bị, đọc/copy
  và kiểm tra snapshot trước decode, rồi stream cả stdout/stderr và báo lỗi nonzero.
- Nếu muốn chấm checkpoint mới, chạy **9 → 10**; cell 9 chuẩn bị lại môi trường detector
  khi runtime mới. Không tự bật 12 (đánh giá toàn bộ) hoặc kéo dài 3000 bước. Giữ các
  snapshot 500/862 và kết quả cũ; sửa Timer không chứng minh chất lượng tốt hơn.
- Các kiểm thử CPU xác nhận hành vi resume, không thay cho xác nhận chạy thật trên Colab.

## H2 ROI-V1 pilot200 — tiếp tục Recovery trong cùng phiên (14/10/2026)

- Đã có H1 Recovery200 + dev30 thì không train/chấm H1 lại. Thử đúng một H2
  foreground2/background1, semantic tắt; cùng trọng số tác giả, optimizer mới,
  LR/data/crop/seed/200 bước như H1. Không dùng H1 đã train làm khởi tạo chỉ riêng
  H2; mục so sánh đồng-duration này khác các thí nghiệm legacy H2 nêu phía trên.
- Giữ **Wild_Diff_ICMH_H1_Recovery.ipynb và L4 đang kết nối**, không mở phiên mới
  hoặc reset. Thêm một cell continuation duy nhất theo [H2_ROI_PILOT.md](docs/H2_ROI_PILOT.md)
  nếu notebook đã mở trước lúc cập nhật. Pull không thay cell đang mở; lần sửa sau
  chỉ dùng2A và chạy lại continuation, không copy bản vá mới.
- Bảng điều khiển mặc định không train/copy/install. Lần lượt chuẩn bị/xem10
  overlay thật → tick xác nhận đúng hash → train H2 200 → chấm H2 dev30 → đọc
  trạng thái. Tận dụng local files/env, chỉ chuẩn bị ảnh cần cho mỗi thao tác.
  Training dùng union train/val; đánh giá chỉ30 originals. Môi trường detector
  dùng lại `/content/envs/h1-recovery-detect`, không tự dựng mới.
- H2 có run/checkpoint/CSV riêng. Không chạm H1/B0 artifacts; không chạy Xie,
  alpha sweep/full-dev/test/extensions. Kết quả30 ảnh là pilot, không phải hoàn
  tất Phase3/4 hay chứng minh ưu thế nộp báo.

## Compute unit (chi phí toàn phiên)

- Hằng số `COLAB_CU_PER_HOUR = 1.54` trong notebook chỉ để ước lượng. CU thật = chênh lệch
  "Available" trong Colab Resources trước và sau phiên; ghi lại mỗi phiên (Bước 1 và Bước 10
  có ô để điền).
- Runtime GPU tốn CU cả khi đang cài đặt hoặc chép ảnh; ngắt runtime ngay khi xong việc.
- Chọn L4 thủ công (quyết định D-04); notebook không có nhánh riêng cho T4/A100.
