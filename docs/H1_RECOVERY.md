# H1: đọc kết quả trước, thử nhỏ sau

Notebook mới: [mở H1 Recovery trên Colab](https://colab.research.google.com/github/TranDuon/Wild-Diff-ICMH/blob/main/Wild_Diff_ICMH_H1_Recovery.ipynb).
Notebook pilot cũ và thư mục `runs/h1_pooled_pilot_ls512/lambda_2` được giữ nguyên.

## Đã có gì?

H1 cũ đã chạy đến **1.000 bước cập nhật optimizer**, có kết quả trên 30 ảnh validation
cố định ở bước 500, 862 và 1.000. Lần tiếp tục từ 862 đến 1.000 đã chạy được; đây không
còn là vấn đề timer/resume. H1 ở 1.000 bước dùng khoảng 29,6% ít bit hơn B0 λ = 2,
nhưng giữ/phát hiện con vật kém hơn tại hai điểm này. Đó là **đánh đổi ở dung lượng
khác nhau**, chưa chứng minh fine-tuning luôn kém hơn.

[Báo cáo từ cache](../results/h1_cached_review_261013.md) tính lại B0 λ = 2/8/32 và
H1 bước 500/1.000 trên đúng **cùng 30 ảnh**, không lấy trung bình 202 ảnh của B0
để so với 30 ảnh H1. Báo cáo nêu các điểm gần dung lượng, ảnh bị mất phát hiện,
bằng chứng về protocol và dữ liệu còn thiếu. Không chạy lại baseline hay Xie-SGC.

Các điểm cần đọc đúng:

- Bit trên mỗi pixel cho biết dung lượng nén; nhỏ hơn nghĩa là ít dữ liệu hơn.
- Điểm phát hiện trung bình đánh giá khả năng khớp hộp con vật, nhưng hộp tham chiếu
  ở đây do MegaDetector trên ảnh gốc tạo ra, **không phải nhãn chuẩn do người gán**.
- Tỷ lệ bỏ sót là tỷ lệ **ảnh** có phát hiện con vật ở bản gốc nhưng bản tái tạo không
  có; không phải tỷ lệ từng con vật bị mất.
- Chất lượng gần ảnh gốc và chất lượng cảm nhận là hai góc nhìn bổ sung, không đồng
  nghĩa giữ đúng con vật. Một ảnh trông đẹp vẫn có thể mất đặc điểm quan trọng.
- 30 ảnh đã chọn phân tầng chỉ giúp chọn hướng thử; chưa đủ số liệu để kết luận cho
  toàn bộ dữ liệu, chứng minh ý nghĩa thống kê hay báo cáo thắng một bài khác.
- So các điểm đã lưu hoặc nội suy giữa chúng chỉ mang tính mô tả; không phải đã đo
  hai mô hình ở đúng cùng dung lượng, không được gọi là chiến thắng đường cong nén.

Chưa xác nhận tốc độ học là nguyên nhân. Nếu chưa đọc được ảnh gốc/tái tạo hoặc log
thành phần loss/tốc độ học, báo cáo phải ghi thiếu bằng chứng, không suy luận thành lỗi.
Tỷ lệ crop chứa hộp con vật thấp cũng chưa chứng minh lỗi crop: nhiều ảnh training vốn
trống và H1 cố ý dùng crop đều, không crop ưu tiên hộp.

## Cách dùng notebook mới

1. **Chỉ đọc kết quả:** runtime CPU, chạy **1 → 2 → 3 → 4**, rồi dừng. Bước 3 chỉ đọc
   cache và chấm điểm hộp đã có; không cài detector, chép corpus hay mở checkpoint lớn.
2. **Nếu được chọn thử:** đọc báo cáo, chọn L4 và Lưu/kết nối phiên mới; chạy 1–4 lại.
   Ở Bước 4 đổi `RUN_TRIAL = True`, `ACKNOWLEDGE_REVIEW = REVIEW_SHA256` rồi chạy
   **5 → 6 → 7**. Báo cáo thiếu hợp đồng so sánh/không cho phép thử thì cổng vẫn đóng.
3. **Đánh giá lượt thử:** đủ 200 bước thì đổi `RUN_TRIAL = False`,
   `RUN_EVALUATION = True`; chạy lại **4 → 5 → 6 → 8 → 9**. Chỉ đánh giá 30 ảnh của
   checkpoint mới và dùng B0 đã lưu. Không đánh giá toàn bộ validation/test.
4. **Ngắt giữa chừng:** runtime mới chạy 1–6 với cùng hash/đường dẫn đã xác nhận,
   rồi Bước 7. Nó chỉ resume trạng thái của run mới đã kiểm tra, không cộng thêm 200.
5. **Có bản sửa code:** sau khi được báo đã push, chạy **2A** rồi cell lỗi. Không copy
   cell tạm. Nếu thay cấu trúc notebook, mở lại link canonical trên một lần.
6. Kết thúc thì ghi số compute unit còn lại và chủ động ngắt runtime ở Bước 10.

Mọi hành động nặng tắt mặc định để chạy toàn bộ notebook không tự tiêu tài nguyên.
Hash gắn quyết định với đúng file báo cáo đã đọc. Nếu dữ liệu/báo cáo thay đổi, đọc lại
và xác nhận lại; không chuyển một quyết định cũ sang thí nghiệm khác.

## Thử nghiệm nhỏ thay đổi đúng một yếu tố

Run riêng: `MyDrive/wild_diff_icmh/runs/h1_low_lr_recovery_ls512/lambda_2`.
Artifacts riêng: `MyDrive/wild_diff_icmh/phase3/h1_low_lr_recovery`.

| Yếu tố | Lượt thử mới |
|---|---|
| Điểm khởi đầu | Trọng số checkpoint tác giả λ = 2, optimizer mới |
| Tốc độ học chính | 0,00001 thay vì 0,0001 |
| Tốc độ học phụ | Giữ 0,001 |
| Tổng bước | Tối đa 200, không tự tăng lên 1.000/3.000 |
| Loss, crop, dữ liệu, seed | Giữ nguyên H1 pilot; chưa bật ROI loss |
| Đánh giá | Cùng 30 ảnh validation, cạnh dài 512, 50 bước lấy mẫu, seed 231, guidance 3; chấm ở kích thước gốc |

Không lấy checkpoint H1 cũ để resume rồi chỉ sửa tốc độ học trong YAML: optimizer
trong checkpoint có thể khôi phục tốc độ học cũ. Lượt mới nạp **trọng số** tác giả;
các phiên tiếp theo chỉ resume full-state checkpoint của **chính run mới** và kiểm
tra tốc độ học trong optimizer. Run cũ, hợp đồng cấu hình cũ, ảnh và báo cáo cũ không đổi.

Ước tính riêng training theo tốc độ đo 5,65 giây/bước: khoảng **18,8 phút cho 200
bước**, khoảng **0,48 compute unit** nếu Colab tiêu 1,54 unit/giờ. Đây không phải hóa
đơn: chưa cộng cài môi trường, chép tài sản, nạp mô hình, staging/lưu checkpoint và
đánh giá. Giới hạn phiên mặc định 0,5 giờ training, tối đa 2 giờ; nếu chưa đủ 200 bước,
chạy lại từ checkpoint, không kéo dài đích. Chỉ cài detector khi chủ động bật đánh giá.

Chọn hướng tiếp theo dựa trên **dung lượng + giữ con vật + chất lượng ảnh + log thành
phần loss**, không chỉ một con số hoặc loss tổng giảm. Nếu lượt thử vẫn kém, dừng
để chẩn đoán, không mặc định huấn luyện lâu hơn. Lượt thử 200 bước chưa đủ để kết luận
về hội tụ cuối hoặc đóng góp khoa học, kể cả số đo nhanh tốt hơn.

Đây là lượt thăm dò ngắn, chưa có nhánh tốc độ học cũ dừng đúng 200 bước để đối chứng.
So nó với checkpoint 500/1.000 bước còn khác số bước học; không quy mọi khác biệt
cho tốc độ học. Không mở thêm lượt đối chứng GPU chỉ để biến thăm dò thành kết luận.

## H2 tiếp theo: một cặp thử có kiểm soát, chưa tự chạy

Chỉ sau khi review và chọn một nguồn trọng số đáng tin cậy mới lên cặp H1/H2:

- Hai nhánh bắt đầu từ **cùng checkpoint nguồn**, optimizer khởi tạo giống nhau.
- Cùng tốc độ học chính/phụ, seed, danh sách ảnh, resize/crop, cách lấy batch,
  số bước cập nhật, ngân sách và protocol đánh giá.
- Nhánh đối chứng dùng loss đều; nhánh H2 chỉ đổi **trọng số loss vùng có con vật**.
  Không đồng thời đổi cách crop ưu tiên hộp, tags hay kiến trúc khác.
- Trước train xem khoảng 10 overlay hộp/mask sau resize, flip và crop; kiểm tra
  tọa độ, hộp nhỏ, ảnh đêm và ảnh trống. Hộp/mask tự động phải được ghi rõ là nhãn giả.
- Dùng cùng 30 ảnh cho quyết định rẻ trước. Chỉ mở rộng tập đánh giá sau khi có dấu
  hiệu tốt và đồng ý ngân sách; không chọn mô hình bằng tập test.
- Đóng góp H2 cần giữ con vật/chất lượng vùng động vật tốt hơn **nhánh đối chứng
  tương ứng tại dung lượng so sánh được**, kèm chi phí và giới hạn. Không chỉ so H2
  với H1 cũ có tốc độ học/số bước/nguồn optimizer khác.

Tài liệu này chỉ định nghĩa cặp đối chứng; chưa triển khai hay train H2 và không
đánh dấu Phase 3 hoàn thành. Không tái hiện phương pháp Xie; số công bố của workshop
là tham chiếu văn học với điều kiện/dataset/chỉ số được công bố rõ, không trộn vào
phép so trực tiếp cache H1/B0 trên 30 ảnh hiện tại.
