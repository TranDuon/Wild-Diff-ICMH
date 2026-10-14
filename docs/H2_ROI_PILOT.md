# Một lượt H2 ưu tiên vùng con vật, tiếp tục ngay trong Recovery

H1 Recovery đã train 200 bước và chấm đúng 30 ảnh. Kết quả này là đối chứng đã
lưu, không chạy H1/B0/Xie-SGC lại. H2 là **một lượt duy nhất**: vùng con vật có
trọng số 2, nền 1, semantic ROI tắt. Giữ nguyên tác giả lambda2, optimizer mới,
tốc độ học chính 0,00001/phụ 0,001, pooled20 site, cạnh dài512, crop256 ngẫu
nhiên và seed. Khởi đầu từ checkpoint H1 đã train sẽ không còn cùng200 bước,
nên lượt H2 này phải dùng cùng trọng số tác giả như lượt H1 Recovery.

## Đối chứng H1 đã xác nhận ngày 14/10/2026

H1 tốc độ học thấp đã **train đủ200 và đánh giá xong**, không còn là lượt thử
chưa chạy. Trên cùng30 ảnh đã chốt:

| Kết quả | Bit/pixel (ít hơn là nhỏ hơn) | Điểm phát hiện (cao hơn tốt hơn) | Ảnh bỏ sót |
|---|---:|---:|---:|
| Diff-ICMH gốc lambda2 | 0,00599017 | 0,394393 | 6/20 |
| H1 tốc độ học thấp200 | 0,00535252 | 0,408349 | 6/20 |

H1 dùng ít hơn khoảng10,65% bit và điểm phát hiện cao hơn ở **điểm đã đo này**;
độ gần pixel gốc hơi giảm (22,335→22,245), sai khác cảm nhận hơi giảm
(0,53776→0,53657). Cả hai bỏ sót đúng cùng6 ảnh theo tham chiếu tự động. Đây là
tín hiệu đủ để chọn H1 làm đối chứng pilot, chưa chứng minh thắng toàn đường
cong hoặc tốc độ học thấp là nguyên nhân (H1 cũ có số bước khác).

Bằng chứng đã đọc: `phase3/h1_low_lr_recovery/eval/step_000200_quick/comparison.json`,
`done.json`, hợp đồng run và CSV loss. Mã băm hợp đồng H1:
`73061ee145b3f2010c692daa43587ccf281edb5a9fedcb1fa8b51a3cdaa1dbe0`.
Runner H2 kiểm lại các bằng chứng này trên Drive trước khi cho chạy.

## Notebook đã mở: giữ phiên L4 và dữ liệu đang có

Không ngắt/reset runtime, không mở notebook khác, không chạy lại các cell train
H1. Git pull đổi file trong repository nhưng **không đổi cell notebook đã mở**.
Vì vậy, chỉ lần đầu cần thêm **một cell ở cuối chính notebook Recovery đang chạy**:

```python
RUN_TRIAL = False
RUN_EVALUATION = False
safe_pull()
import runpy
H2_CONTINUATION = runpy.run_path(
    str(REPO / 'tools/colab_h2_continuation.py'), run_name='__main__',
    init_globals={'REPO': REPO, 'DRIVE_ROOT': DRIVE_ROOT, 'CU_RATE': CU_RATE},
)
```

Không cần Ctrl+S vào GitHub để chạy cell. Để lưu cell cho lần sau, có thể lưu
bản notebook trong Drive của chính bạn; **không chuyển runtime để làm việc đó**.
File notebook canonical vẫn là `Wild_Diff_ICMH_H1_Recovery.ipynb`, được cập nhật
ngay tại chỗ. Cell trên chỉ hiện bảng điều khiển, không tự chạy GPU hoặc cài gói.
Nếu đang mở phiên mới sau khi đã tự xóa runtime thì `/content` cũ không còn; không
có cách phục hồi RAM bằng git pull. Nhưng **phiên đang có không bị thay/xóa** bởi
cell continuation này.

## Bấm lần lượt trong bảng điều khiển

1. **Chuẩn bị + xem ROI:** kiểm tra nguồn H1 đã hoàn tất, dùng tối đa10 ảnh train
   để vẽ mask bằng đúng resize/crop/flip của dataset. Ảnh local đúng kích thước
   file được dùng lại; không chép toàn bộ corpus chỉ để xem10 ảnh. Đọc báo cáo và
   xem panel trước khi tick xác nhận. Đây là hộp MegaDetector tự động, không phải
   nhãn segmentation do người vẽ. Crop ngẫu nhiên có thể không chứa con vật và
   vùng nhỏ có thể biến mất ở latent32×32; báo cáo không sửa seed để che điều này.
2. **Tự tick xác nhận:** xác nhận gắn với mã băm đúng preflight và overlay hiện tại.
   Nếu nguồn/báo cáo thay đổi phải xem và xác nhận lại, không dùng quyết định cũ.
3. **Train H2 tới200:** chỉ H2, giới hạn tổng200 optimizer steps; checkpoint mới
   nằm ở `runs/h2_roi_v1_alpha2_ls512/lambda_2/`. Phiên bị ngắt thì bấm lại Train
   để tiếp tục checkpoint H2 đúng hợp đồng, không học từ đầu hay dùng H1 optimizer.
   Đủ200 thì không train thêm. Training chuẩn bị đúng union train/val, bỏ qua
   ảnh/checkpoint local đã đủ. Không đổi Torch/NumPy hoặc dựng lại detector.
4. **Chấm H2 30 ảnh:** chỉ khi đủ200, giải mã đúng DDIM50/seed231/hướng dẫn3/cạnh
   dài512, chấm ở kích thước ảnh gốc. Chỉ chép30 ảnh còn thiếu, dùng môi trường
   detector đã có tại `/content/envs/h1-recovery-detect`. Môi trường thiếu/hỏng
   sẽ báo lỗi để xử lý, không tự cài lại hoặc reset phiên. Các stage hoàn tất được
   kiểm tra và dùng lại. Không tạo ảnh hay chạy phát hiện lại H1/B0.
5. **Đọc trạng thái:** gửi `preflight.json`, `eval/step_000200_quick/comparison.json`,
   `throughput.json` và `loss_components.csv` mới. Chưa kéo dài, quét alpha hay
   chấm toàn bộ validation/test. Không tự đánh dấu Phase3/4 hoàn thành.

Nếu lỗi: giữ nguyên phiên, chạy cell **2A** pull, chạy lại cell continuation,
chọn lại đúng thao tác. Script chạy trong child mới để không dùng module cũ đã
import. Log đầy đủ nằm ở `phase3/h2_roi_v1_alpha2/logs/`; không copy cell vá lỗi
khác mỗi lần. Bảng mới yêu cầu đọc/xác nhận lại trước thao tác nặng.

Nếu detector bị ngắt giữa lúc ghi dòng cuối, H2 giữ bản `.bak` nguyên trạng,
khôi phục các dòng hoàn chỉnh rồi chỉ làm tiếp ảnh thiếu. Dòng hỏng ở giữa hoặc
nguồn khác bị chặn, không tự bỏ qua. Kiểm tra chống lẫn split vẫn đọc manifest
đầy đủ; riêng kiểm tra tài sản H2 chỉ yêu cầu train/val, không cần ảnh test.

## Ngân sách và cách đọc kết quả

Tốc độ H1 đã đo khoảng5,64 giây/optimizer step: phần200 bước khoảng19 phút,
~0,48 compute unit nếu Colab đang báo1,54/giờ. **Không gồm** nạp/lưu checkpoint,
chép dữ liệu, setup hoặc chấm điểm. Đánh giá trước đây gồm giải mã khoảng9 phút,
detector khoảng17 phút; lần này có thể nhanh hơn khi cache mô hình còn nhưng
không được coi là bảo đảm. Phiên L4 vẫn tiêu unit cả khi chép/chờ.

So H2 với **H1_low_lr_200 đã lưu**: dung lượng thực trên mỗi pixel, độ gần ảnh
gốc, sai khác cảm nhận, điểm phát hiện, số ảnh bỏ sót và phát hiện giả. Ghi thêm
loss vùng con vật/nền, phần mask và tốc độ học. Thiếu log là chưa có bằng chứng,
không thay bằng0. Cả hai phải cùng30 ID/protocol; bitrate đạt được có thể khác,
nên một điểm không chứng minh thắng cả đường cong ở cùng dung lượng.

30 ảnh chỉ là lượt chọn hướng. Điểm phát hiện dùng MegaDetector trên ảnh gốc làm
tham chiếu tự động, không phải nhãn người gán. “Phát hiện giả” chỉ là ứng viên cần
nhìn lại ảnh, không đủ khẳng định diffusion tạo con vật. Kiểm thử CPU xác nhận
mã và hợp đồng; ROI trên ảnh thật/huấn luyện GPU vẫn cần bạn xác nhận trong phiên.
