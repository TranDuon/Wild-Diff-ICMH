# Phase 3 — H1 pooled pilot, tiết kiệm tài nguyên

Notebook: `Wild_Diff_ICMH_Phase3_H1.ipynb`, nhánh `main`.

## Mục tiêu và giới hạn

Huấn luyện một mô hình Diff-ICMH trên tập train của cả 20 site Kgalagadi,
khởi tạo từ checkpoint tác giả λ=2. Đây là H1 (fine-tuning thích ứng miền),
chưa phải H2 (ưu tiên con vật bằng loss) hay H3 (điều kiện theo miền).
Không tái hiện Xie-SGC; chỉ dùng số đã công bố khi thảo luận văn học liên quan.
Chưa tuyên bố Phase 3 hoàn thành hay H1 tốt hơn B0 chỉ vì code đã kiểm thử.

- Cạnh dài xử lý 512, crop train 256 ngẫu nhiên đều; full-width control ratio 1.
- Smoke 20 optimizer steps trong run riêng, decode 4 ảnh validation × DDIM5.
- Pilot 500 optimizer steps; một step tích lũy 8 batch × 1 ảnh.
- Validation trong training: chỉ loss, 2 batch/100 optimizer steps; không sinh
  ảnh, không tính LPIPS lúc train. Loss là tín hiệu theo dõi, không phải kết
  quả chất lượng tái tạo hoặc tiêu chí khẳng định hơn B0.
- Checkpoint rolling mỗi 100 step + checkpoint đầy đủ cuối phiên. Không train
  lại khi đã đạt mục tiêu. Không tự bỏ qua checkpoint lỗi để bắt đầu lại.
- Sau pilot: dev30 cố định, phân tầng từ dev202 đã đóng băng ở Phase 2.
  Tập nhỏ có chủ đích, không đại diện tỷ lệ tự nhiên và không đủ để nộp báo.
- Quick/full evaluation: cùng DDIM50, seed231, guidance3, cạnh dài512; ảnh
  tái tạo được so với ảnh gốc ở độ phân giải gốc, kể cả số byte của bitstream.
- Đo LPIPS và chất lượng ảnh, MegaDetector trên ảnh tái tạo. Tái sử dụng hộp
  ảnh gốc và B0 Phase2. MegaDetector là nhãn tự động, không phải ground truth.
- Không SpeciesNet/DISTS/đánh giá test trong pilot. Không chạy lại baseline.

## Thao tác trên Colab

1. Chọn runtime L4 đã dùng thành công. Điền số CU hiện tại (không dùng số dư
   cũ trong kế hoạch), tốc độ CU/giờ đang hiển thị, giới hạn thời gian phiên.
2. Chạy Bước 1→6 để mount Drive, lấy main, cài an toàn và chép dữ liệu cục bộ.
   Chỉ copy file thiếu; không tải lại dataset/checkpoint từ Internet.
3. Chạy 7 để kiểm tra cấu hình mới một lần. Run smoke tách khỏi pilot.
4. Chạy 8: mục tiêu tổng500. Nếu hết giới hạn phiên trước khi đạt500, chạy
   lại chính cell8. Nếu runtime bị thu hồi, chạy1→6 rồi8 để resume.
5. Chạy9→10: dev30, xem `comparison.json`, `throughput.json` và `crop_stats.json`.
   Kiểm tra loss không bất thường, dung lượng/chất lượng và con vật còn được
   giữ. Không tự quyết định tăng training từ một chỉ số trên30 ảnh.
6. Chỉ sau review: bật11 để kéo dài tổng1000/3000; chạy9→10 lại. Chỉ bật12
   để chấm dev202 của checkpoint được chọn. Các bước nặng tắt mặc định.
7. Khi đã xong việc: ghi CU cuối phiên và bật ngắt runtime ở13 để tránh giữ
   GPU nhàn rỗi. CU ước tính job không bao gồm toàn bộ chi phí setup/idle.

Nếu có lỗi: gửi log `logs/p3_*.log`. Sau khi được báo đã push bản sửa, chạy
**2A**, rồi đúng cell bị lỗi. Không copy cell pull tạm, không reset/xóa repo.
Pull cập nhật script/config, không thay cell notebook đang mở; chỉ khi có thay
đổi cấu trúc notebook mới cần mở lại notebook từ link GitHub/Colab.

## Tài sản và kết quả

Đầu vào đã có: `MyDrive/wild_diff_icmh/images/`, `checkpoints/`,
`tags/KGA_all.jsonl`, `phase2/kgalagadi_dev_b0eval.txt`,
`phase2/kgalagadi_illumination.jsonl`, `phase2/detections/originals.jsonl`,
`phase2/archive/B0_ls512_ddim50/lambda_2/`.

Đầu ra trên `MyDrive/wild_diff_icmh/`:

| Đường dẫn | Ý nghĩa |
|---|---|
| `runs/h1_pooled_pilot_ls512/lambda_2/checkpoints/last.ckpt` | Resume đầy đủ optimizer/global step |
| `runs/h1_pooled_smoke_ls512/lambda_2/` | Smoke riêng, không dùng làm kết quả pilot |
| `phase3/protocol.json`, `dev30.txt`, `smoke4.txt` | Protocol và mẫu đóng băng |
| `phase3/snapshots/<run>/step_*/` | Checkpoint cố định cho từng lần đánh giá |
| `phase3/archive/step_*_<scope>_ddim*/` | Bitstream, ảnh tái tạo, thông tin decode |
| `phase3/eval/step_*_quick/comparison.json` | B0 và H1 trên cùng30 ảnh |
| `phase3/eval/B0_quick/` | Chấm B0 nhỏ một lần, dùng lại sau khi kéo dài train |
| `phase3/eval/step_*_full/` | Kết quả dev202 nếu đã chủ động bật |
| `phase3/resource_usage.jsonl`, `logs/p3_*.log` | Thời gian/chi phí ước tính/log lỗi đầy đủ |
| `results/results.jsonl` | Registry chung, ID Phase3 riêng cho quick/full, không ghi đè số Phase2 |
| `runs/.../throughput.json`, `crop_stats.json` | Tốc độ ổn định và tỷ lệ crop chứa con vật |

Các snapshot gắn với step và run, không bị thay bằng checkpoint mới. Protocol,
tập mẫu và cấu hình run không được tự ghi đè. Khi có thay đổi thiết kế thí
nghiệm thực sự, dừng và quyết định migration/run mới, không trộn kết quả.

## Điều còn phải xác nhận

Kiểm thử CPU/syntax không thay thế chạy GPU. Cần Colab xác nhận smoke→pilot500,
resume, quick evaluation đủ30 ảnh, thời gian/CU thực và review chất lượng.
Sau đó mới quyết định tăng bước/λ tiếp theo và xác nhận Phase3; không tự đánh
dấu xong toàn bộ ba điểm λ2/8/32 từ một pilot λ2.
