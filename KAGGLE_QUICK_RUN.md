# Diff-ICMH inference trên Kaggle

File chạy chính: Diff_ICMH_Kaggle_Quick_Run.ipynb.

Notebook chuyển phần **Quick Start / Inference** trong README.md thành 8 bước
chạy tuần tự trên Kaggle:

1. Clone repo TranDuon/Wild-Diff-ICMH.
2. Cài PyTorch 2.4.1/CUDA 12.1 và các dependency inference.
3. Kiểm tra GPU, phiên bản thư viện và tối thiểu 15 GiB dung lượng trống.
4. Tải SD 2.1, RAM++ và đúng checkpoint Diff-ICMH của RD point đã chọn.
5. Stage ảnh Kodak hoặc ảnh từ một Kaggle Dataset.
6. Gọi inference_partition.py bằng đúng dotlist override trong README.
7. Kiểm tra ảnh tái tạo, bitstream và bpp.txt, rồi hiển thị ảnh so sánh.
8. Đóng gói output thành ZIP để tải về.

## Cách chạy

Trong **Notebook options** của Kaggle:

- Chọn **Accelerator = GPU**.
- Bật **Internet**.
- Chạy các cell từ trên xuống.

Lần đầu có thể lâu vì phải tải các checkpoint lớn. Nếu Kaggle yêu cầu restart
sau cell cài dependency, restart session, chạy lại cell clone, bỏ qua cell cài
đặt và tiếp tục từ preflight.

## Cấu hình quan trọng

| Biến | Mặc định | Ý nghĩa |
|---|---:|---|
| BPP_WEIGHT | 2 | RD point, nhận 2, 4, 8, 16 hoặc 32 |
| CONTROL_MODULE_SCALE | 1.0 | Tỉ lệ control module như README |
| CFG_SCALE | 5.0 | Classifier-free guidance scale như README |
| SMOKE_TEST | True | Chạy 1 ảnh với 10 diffusion steps |
| SEED | 231 | Seed của script inference |
| SOURCE_DIR | Kodak trong repo | Có thể đổi sang /kaggle/input/... |
| MAX_IMAGES | 1 khi smoke test | Đặt None để chạy toàn bộ ảnh |

Sau khi smoke test thành công, đặt SMOKE_TEST = False. Notebook khi đó dùng
STEPS = 50, đúng với lệnh inference trong README.

## Output hợp lệ

Mỗi lần chạy thành công phải có:

- ảnh PNG tái tạo trong thư mục output;
- bitstream nén trong thư mục con data/;
- bpp.txt chứa bpp, PSNR, SSIM và LPIPS cho từng ảnh cùng giá trị trung bình;
- file ZIP được tạo ở cell cuối.

Notebook đã được kiểm tra định dạng JSON và cú pháp tĩnh của toàn bộ cell code.
Inference GPU thực tế cần được chạy trên Kaggle vì máy local không có CUDA và
không tải sẵn các checkpoint lớn.
