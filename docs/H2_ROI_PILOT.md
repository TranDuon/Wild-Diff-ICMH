# Một lượt H2 ưu tiên vùng con vật trong Recovery — runtime L4 mới

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

## Phiên L4 mới: đi tuyến H2 trong một notebook duy nhất

Runtime cũ đã bị xóa: `/content` cũ đã mất, kể cả packages, symlink, hai
checkpoint local và detector. H1 đã lưu trên Drive là đối chứng **chỉ đọc**;
không cần chạy lại H1 để tái tạo môi trường. Mở bản mới nhất của **cùng một**
`Wild_Diff_ICMH_H1_Recovery.ipynb` trên Colab, chọn GPU **L4**, không dùng Run All.
Không chạy H1 4–8 hoặc các cell train/evaluate lịch sử. Làm tuần tự:

1. **Bước 1 — Drive:** mount đúng `DRIVE_ROOT` đang có H1 recovery200, H2 assets
   và kết quả B0. Ghi CU còn lại nếu cần tính chi phí phiên.
2. **Bước 2 — main:** clone hoặc safe-pull `main` vào `REPO`. Pull từ GitHub phải
   thành công và đúng commit mới trước khi setup. Nếu notebook đã mở trước khi
   bản mới được push, reload bản canonical; git pull không sửa cell đang mở.
3. **H2-A — cờ setup:** cả `RUN_H2_DEPENDENCIES` và `RUN_H2_DETECTOR` mặc định
   `False`. Đổi riêng cờ dependency thành `True`, chạy H2-A rồi H2-B. Cell H2-B
   kiểm tra đúng L4 trước cài đặt, giữ NumPy/SciPy/Torch/Torchvision, cài closure
   CompressAI/RAM và `pycocotools==2.0.11`, kiểm tra import training, và tạo
   link `./checkpoints` có kiểm tra đích. Nó không chép ảnh/checkpoint.
4. **H2-C — detector riêng:** trả `RUN_H2_DEPENDENCIES = False`, bật
   `RUN_H2_DETECTOR = True` trong H2-A, chạy lại H2-A rồi H2-C. Cell kiểm tra
   L4, dựng `/content/envs/h1-recovery-detect` với Colab system packages,
   giữ phiên bản huggingface-hub tương thích, import PytorchWildlife và xác
   nhận `.ready`. Nó chỉ chuẩn bị môi trường, **không chấm H1 hay H2**.
5. **H2-D — bảng điều khiển:** trả cả hai cờ H2-A về `False`, chạy H2-D để hiện
   các nút. Hiện nút không cài, copy, train, decode hay inference. Bấm theo thứ
   tự **preflight → xem overlay → tự xác nhận hash → train200 → eval30 → status**.
   Mỗi nút là hành động chủ động, không tự chuyển sang bước tiếp theo.

Đường dẫn `./checkpoints` chỉ trỏ tới `/content/data/wild_diff_icmh/checkpoints`;
runner H2 mới stage checkpoint tác giả lambda2 và SD2.1 lúc train/eval cần tới,
không tải hay sao chép toàn bộ kho. Nếu đã có link trỏ sai hoặc đường dẫn thường,
H2-B dừng thay vì ghi đè. Không chạy Step 3/5/8 của H1 để bù tài sản phiên mới.

### Tuyến cùng phiên cũ (ghi nhận lịch sử, không áp dụng sau khi xóa runtime)

Trước khi phiên cũ bị xóa, cell continuation từng có thể dùng lại packages,
checkpoint và detector đã nằm trong `/content`. Tuyến này **không còn đúng cho
runtime mới**. Git pull không khôi phục RAM hay local assets; dùng H2-A/B/C/D ở
trên. File canonical vẫn là Recovery notebook, không tạo notebook thứ hai.

## Bấm lần lượt trong bảng điều khiển

1. **Chuẩn bị + xem ROI:** kiểm tra nguồn H1 đã hoàn tất trên Drive (chỉ đọc), dùng tối đa10 ảnh train
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
   ảnh/checkpoint local đã đủ. Chỉ stage ảnh gốc thuộc train/val cùng checkpoint
   tác giả lambda2 và SD2.1; không stage ảnh test hay toàn bộ corpus. Không đổi
   Torch/NumPy hoặc dựng lại detector.
4. **Chấm H2 30 ảnh:** chỉ khi đủ200, giải mã đúng DDIM50/seed231/hướng dẫn3/cạnh
   dài512, chấm ở kích thước ảnh gốc. Chỉ chép ảnh gốc của frozen dev30 còn
   thiếu, dùng môi trường
   detector đã có tại `/content/envs/h1-recovery-detect`. Môi trường thiếu/hỏng
   sẽ báo lỗi để xử lý, không tự cài lại hoặc reset phiên. Các stage hoàn tất được
   kiểm tra và dùng lại. Không tạo ảnh hay chạy phát hiện lại H1/B0.
5. **Đọc trạng thái:** gửi `preflight.json`, `eval/step_000200_quick/comparison.json`,
   `throughput.json` và `loss_components.csv` mới. Chưa kéo dài, quét alpha hay
   chấm toàn bộ validation/test. Không tự đánh dấu Phase3/4 hoàn thành.

Nếu lỗi do script đã được sửa và repo vẫn sạch: giữ nguyên phiên, chạy cell
**2A** safe-pull rồi chạy lại H2-D. Nếu Git báo có thay đổi local, dừng và gửi
log; không force-pull hay xóa tài sản để đi tiếp. Sau đó
chọn lại đúng thao tác. Script chạy trong child mới để không dùng module cũ đã
import. Log đầy đủ nằm ở `phase3/h2_roi_v1_alpha2/logs/`; không copy cell vá lỗi
khác mỗi lần. Bảng mới yêu cầu đọc/xác nhận lại trước thao tác nặng.

Nếu detector bị ngắt giữa lúc ghi dòng cuối, H2 giữ bản `.bak` nguyên trạng,
khôi phục các dòng hoàn chỉnh rồi chỉ làm tiếp ảnh thiếu. Dòng hỏng ở giữa hoặc
nguồn khác bị chặn, không tự bỏ qua. Kiểm tra chống lẫn split vẫn đọc manifest
đầy đủ; riêng kiểm tra tài sản H2 chỉ yêu cầu train/val, không cần ảnh test.

## Ngân sách và cách đọc kết quả

Tốc độ H1 đã đo khoảng5,64 giây/optimizer step: phần200 bước khoảng19 phút,
~0,48 compute unit nếu Colab đang báo1,54/giờ. Đây chỉ là **ước tính phần train**,
không gồm cài H2-B/C, nạp/lưu checkpoint, chép dữ liệu, preflight hay chấm điểm.
Setup và eval cũng tiêu CU từ lúc gắn L4. Đánh giá trước đây gồm giải mã khoảng9 phút,
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
