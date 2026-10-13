# Kiểm tra H1 bằng kết quả đã lưu

Đã tính lại đúng cùng 30 ảnh kiểm chứng; không huấn luyện, giải mã hay chạy bộ phát hiện mới.

| Điểm | Bit/pixel ảnh gốc ↓ | Chất lượng điểm ảnh ↑ | Sai khác cảm nhận ↓ | Độ khớp phát hiện ↑ | Ảnh mất phát hiện ↓ |
|---|---:|---:|---:|---:|---:|
| B0_2 | 0.005990 | 22.3352 | 0.5378 | 0.3944 | 0.3000 |
| B0_8 | 0.002863 | 19.9337 | 0.5759 | 0.2716 | 0.5000 |
| B0_32 | 0.000561 | 16.2368 | 0.6524 | 0.1659 | 0.6000 |
| H1_500 | 0.004320 | 21.9450 | 0.5466 | 0.3603 | 0.3500 |
| H1_1000 | 0.004217 | 21.8798 | 0.5458 | 0.3267 | 0.4500 |

Tên cột tương ứng: số bit trên mỗi pixel; điểm chất lượng tái tạo (PSNR); sai khác cảm nhận (LPIPS); độ chính xác phát hiện trung bình (mAP). Mũi tên cho biết chiều tốt hơn.

## Kết luận dễ hiểu

H1 ở 1.000 bước dùng 29.6% ít bit hơn B0 λ=2. Chưa thể gọi đây là tiến bộ toàn diện: phải xem khả năng giữ con vật cùng với dung lượng.
Không so điểm số ở hai dung lượng khác nhau rồi kết luận toàn bộ phương pháp tốt/xấu hơn. Ba mức B0 đã được tính trên đúng cùng 30 ảnh để đối chiếu.

- H1_500: khoảng dung lượng B0_8 ↔ B0_2; B0 dùng không nhiều bit hơn mà có điểm phát hiện không thấp hơn: không có điểm đã đo. Không có điểm lấn át không có nghĩa H1 thắng cả đường cong.
- H1_1000: khoảng dung lượng B0_8 ↔ B0_2; B0 dùng không nhiều bit hơn mà có điểm phát hiện không thấp hơn: không có điểm đã đo. Không có điểm lấn át không có nghĩa H1 thắng cả đường cong.

Thông số giải mã thực tế: đã xác minh. Quyết định: `conditional_low_lr_trial`.
Nếu được phép thử: một run mới từ trọng số tác giả λ=2, optimizer mới, tốc độ học chính 0,00001; tốc độ học phụ vẫn 0,001. Dừng ở 200 bước rồi chấm đúng 30 ảnh. Không tự kéo lên 3.000 bước hoặc làm H2.

## Chưa thể khẳng định

Đây là đối chiếu với nhãn tự động từ ảnh gốc, không phải hộp con vật do người gán. Tỉ lệ mất phát hiện tính trên số ảnh gốc có phát hiện con vật; không phải số con vật bị mất. Các ứng viên ‘con vật ảo’ phải kiểm tra ảnh thủ công.
Bằng chứng thành phần hàm lỗi/tốc độ học: `unavailable`. Chưa xác định tốc độ học là nguyên nhân; giảm tốc độ học chỉ là giả thuyết thử có kiểm soát.
Ảnh minh họa: chỉ liệt kê các trường hợp cần xem, không tự tải/sinh ảnh và chưa đưa ra kết luận bằng mắt.

## Khoảng 10 trường hợp cần xem

- `KGA:KGA_S1/A05/A05_R1/KGA_S1_A05_R1_IMAG0453` — `snapshot_kgalagadi/KGA_S1/A05/A05_R1/KGA_S1_A05_R1_IMAG0453.JPG`; ảnh gốc chưa có trong phiên kiểm tra này. Xem cùng ảnh B0 và H1 đã lưu.
- `KGA:KGA_S1/A08/A08_R1/KGA_S1_A08_R1_IMAG0007` — `snapshot_kgalagadi/KGA_S1/A08/A08_R1/KGA_S1_A08_R1_IMAG0007.JPG`; ảnh gốc chưa có trong phiên kiểm tra này. Xem cùng ảnh B0 và H1 đã lưu.
- `KGA:KGA_S1/B02/B02_R1/KGA_S1_B02_R1_IMAG0371` — `snapshot_kgalagadi/KGA_S1/B02/B02_R1/KGA_S1_B02_R1_IMAG0371.JPG`; ảnh gốc chưa có trong phiên kiểm tra này. Xem cùng ảnh B0 và H1 đã lưu.
- `KGA:KGA_S1/B04/B04_R1/KGA_S1_B04_R1_IMAG0546` — `snapshot_kgalagadi/KGA_S1/B04/B04_R1/KGA_S1_B04_R1_IMAG0546.JPG`; ảnh gốc chưa có trong phiên kiểm tra này. Xem cùng ảnh B0 và H1 đã lưu.
- `KGA:KGA_S1/B05/B05_R1/KGA_S1_B05_R1_IMAG0019` — `snapshot_kgalagadi/KGA_S1/B05/B05_R1/KGA_S1_B05_R1_IMAG0019.JPG`; ảnh gốc chưa có trong phiên kiểm tra này. Xem cùng ảnh B0 và H1 đã lưu.
- `KGA:KGA_S1/B06/B06_R1/KGA_S1_B06_R1_IMAG0211` — `snapshot_kgalagadi/KGA_S1/B06/B06_R1/KGA_S1_B06_R1_IMAG0211.JPG`; ảnh gốc chưa có trong phiên kiểm tra này. Xem cùng ảnh B0 và H1 đã lưu.
- `KGA:KGA_S1/B06/B06_R1/KGA_S1_B06_R1_IMAG1760` — `snapshot_kgalagadi/KGA_S1/B06/B06_R1/KGA_S1_B06_R1_IMAG1760.JPG`; ảnh gốc chưa có trong phiên kiểm tra này. Xem cùng ảnh B0 và H1 đã lưu.
- `KGA:KGA_S1/B06/B06_R1/KGA_S1_B06_R1_IMAG2955` — `snapshot_kgalagadi/KGA_S1/B06/B06_R1/KGA_S1_B06_R1_IMAG2955.JPG`; ảnh gốc chưa có trong phiên kiểm tra này. Xem cùng ảnh B0 và H1 đã lưu.
- `KGA:KGA_S1/B07/B07_R1/KGA_S1_B07_R1_IMAG0042` — `snapshot_kgalagadi/KGA_S1/B07/B07_R1/KGA_S1_B07_R1_IMAG0042.JPG`; ảnh gốc chưa có trong phiên kiểm tra này. Xem cùng ảnh B0 và H1 đã lưu.

## Dấu vết kiểm chứng

Mã băm dữ liệu đã chọn: `6135e89c5085569d41e41701cb41ebea8131530032d4e8291a1918ed3fbf04d8`. Mã băm danh sách 30 ảnh: `f1f175c1df398f624ec101565fdd12a382cd618d0b4f22ad7fff411fbe9609e8`.
JSON đi kèm giữ nguồn từng điểm, thông số giải mã, mẫu số và mã băm bản ghi. Chưa đánh dấu Phase 3 hoàn thành.
