# Phase 1: Khung xương end-to-end tối thiểu + vá lỗi chặn - Context

**Gathered:** 2026-09-27
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase này đóng gói một lát cắt Colab có thể tái lập từ manifest Kgalagadi đến train ngắn, checkpoint/resume toàn trạng thái, decode và ghi metric. Mục tiêu là loại hết lỗi chặn và đo được chi phí thật trước khi tiêu GPU cho baseline toàn corpus hoặc H1/H2/H3. Toàn corpus, ROI mask, machine-task evaluation và training nghiên cứu dài vẫn thuộc các phase sau.

</domain>

<decisions>
## Implementation Decisions

### Cổng hoàn tất Phase 1
- **D-01:** Giữ lần chạy Colab đã thành công (20 optimizer step, resume thêm một step, decode hai ảnh và ghi `results.jsonl`) làm bằng chứng end-to-end. Không chạy lại toàn notebook chỉ để tái tạo cùng bằng chứng.
- **D-02:** Phase 1 chỉ bổ sung các bằng chứng còn thiếu: leakage negative test, audit metadata 100 ảnh, kiểm tra checkpoint contract và một phép đo throughput/CU có giới hạn. Mỗi cổng phải tạo artifact hoặc log máy đọc được, không nghiệm thu chỉ bằng ảnh chụp màn hình.
- **D-03:** Smoke metric hai ảnh và DDIM 5 phải mang nhãn `smoke/non-report`; không được dùng làm số liệu khoa học hay so sánh mô hình.

### Chính sách GPU và ngân sách
- **D-04:** L4 là tier vận hành chuẩn cho notebook training. Người dùng chọn L4 thủ công; không thêm nhánh T4, không tự động thay batch/precision để cứu runtime T4, và không mặc định A100.
- **D-05:** Đo throughput theo kiểu tăng dần: dùng log smoke hiện có làm ước lượng ban đầu, chạy một cửa sổ calibration hữu hạn trên L4, rồi ngoại suy chi phí 2K step. Chỉ chạy đủ 2K khi dự báo vẫn nằm trong ngân sách Phase 1; nếu không, lưu số đo và sửa dự toán roadmap thay vì đốt CU mù quáng.
- **D-06:** Mọi con số compute phải lưu kèm GPU, elapsed time, optimizer steps và hệ số CU/hour đã dùng; `cu_estimate` là ước lượng, không được trình bày như số CU do Colab cung cấp trực tiếp.

### Resume và checkpoint
- **D-07:** `--init-checkpoint` là warm-start weights-only từ checkpoint tác giả; `--resume` là khôi phục checkpoint dự án đầy đủ. Hai đường không được trộn ý nghĩa.
- **D-08:** Resume chỉ đạt khi checkpoint có `optimizer_states`, `global_step` sau lần chạy tiếp lớn hơn trước, checkpoint contract đúng phiên bản và log ghi rõ đường dẫn được khôi phục.
- **D-09:** Checkpoint crash-recovery phải compact, giữ toàn bộ trạng thái cần cho Lightning resume nhưng loại state đông cứng của SD 2.1/VAE/RAM++; chỉ duy trì rolling `last.ckpt` ở nhịp phù hợp với Colab.

### Dữ liệu mẫu và kết quả
- **D-10:** KGA:A01 là lát cắt Phase 1. Site được phép hiện diện ở cả train/val/test vì protocol là per-site; sequence/burst mới là đơn vị cấm rò rỉ.
- **D-11:** Kiểm tra split phải có cả đường thành công và một fixture cố ý rò sequence để chứng minh preflight thực sự chặn job.
- **D-12:** Audit 100 ảnh đầu ưu tiên EXIF; nếu EXIF bị strip thì dùng `datetime` và `location` từ metadata JSON LILA, đồng thời ghi tỷ lệ nguồn metadata sử dụng được.
- **D-13:** `results.jsonl` trên Drive là registry duy nhất. Việc chạy lại cùng `exp_id` phải upsert/idempotent, không tạo bản ghi trùng.
- **D-14:** Cell cập nhật Git cố định là đường nhận bản sửa. Trong cùng runtime chỉ pull rồi chạy lại cell vừa lỗi và các cell phụ thuộc phía sau; runtime mới thì chạy lại các bước bootstrap theo notebook.

### the agent's Discretion
- Chọn độ dài cửa sổ calibration nhỏ nhất cho ước lượng throughput ổn định.
- Chọn định dạng artifact kiểm tra metadata và checkpoint miễn là máy đọc được và có test.
- Tổ chức các plan nhỏ để ưu tiên CPU/local gate trước GPU gate.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phạm vi và yêu cầu
- `.planning/PROJECT.md` — Core Value, ràng buộc compute và các quyết định kiến trúc không được phá.
- `.planning/REQUIREMENTS.md` — PRE-01..06, DATA-02/03/05, INFRA-03..05 và EVAL-01/08 thuộc Phase 1.
- `.planning/ROADMAP.md` — goal, ngân sách và success criteria của Phase 1; ranh giới với Phase 2/3.

### Luồng Colab và cấu hình
- `Wild_Diff_ICMH_Kgalagadi_Train.ipynb` — notebook người dùng trực tiếp chạy.
- `tools/build_colab_training_notebook.py` — nguồn sinh notebook; mọi sửa cell bền vững phải đi qua đây.
- `configs/train_kgalagadi_colab.yaml` — trainer, callback, run directory và resume policy.
- `requirements-colab.txt` — dependency contract quanh Torch/NumPy/SciPy có sẵn của Colab.

### Hợp đồng dữ liệu, checkpoint và kết quả
- `train.py` — phân tách warm-start/resume và entrypoint Lightning.
- `model/diffeic.py` — compact checkpoint hooks và trainable/frozen model boundary.
- `utils/checkpoint_contract.py` — migration/validation contract cho checkpoint dự án.
- `tools/data/split_check.py` — sequence leakage gate.
- `tools/evaluate_kgalagadi.py` — metric smoke và ghi registry.
- `utils/results_registry.py` — schema/upsert của `results.jsonl`.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `tools/build_colab_training_notebook.py`: đã sinh đủ Bước 1–9, log chi tiết lên Drive, permanent pull cell, smoke train/resume/decode/evaluate.
- `tests/test_phase1_completion_contract.py`: static contract cho notebook, checkpoint migration và completion path.
- `tests/test_checkpoint_contract.py`: test migration, compact checkpoint và resume validation.
- `tests/test_results_registry.py`: test schema và idempotent upsert.
- `tools/data/split_check.py` cùng `tests/data/test_split_check.py`: có sẵn positive/negative split contract.

### Established Patterns
- Notebook được sinh từ Python builder rồi kiểm tra syntax; không sửa thủ công notebook làm nguồn chân lý.
- JSONL là giao diện ổn định cho manifest, RAM tags và result registry.
- Các preflight rẻ chạy local/CPU trước; model/checkpoint gate thật chỉ chạy trên Colab L4.
- Lỗi subprocess được stream vào log trên Drive và cell ném lỗi ngắn trỏ tới log đầy đủ.

### Integration Points
- `train.py` gọi preflight, dựng model, warm-start/resume và `Trainer.fit`.
- `configs/train_kgalagadi_colab.yaml` điều khiển checkpoint cadence, precision và `default_root_dir` trên Drive.
- `inference_partition.py` nhận checkpoint project từ Bước 7/8 và tạo reconstruction/bitstream cho Bước 9.
- `tools/evaluate_kgalagadi.py` chuyển reconstruction thành per-image result và registry canonical.

</code_context>

<specifics>
## Specific Ideas

- Quy trình vận hành phải đơn giản: code sửa và push lên GitHub, Colab chạy permanent pull cell, sau đó chạy lại đúng cell lỗi.
- Không tạo thêm folder code cho mỗi lần sửa; checkpoint/data/log/results trên Drive tồn tại độc lập với clone code trong `/content`.
- Không đưa lại các thay đổi tự động cứu T4; người dùng sẽ chủ động chọn L4.

</specifics>

<deferred>
## Deferred Ideas

- Toàn bộ 10.222 ảnh, shard local, ROI pseudo-mask và eval MegaDetector/SpeciesNet — Phase 2.
- H1 dài, rate-overlap và catastrophic-forgetting checks — Phase 3.
- ROI-weighted loss và domain-aware TGM — Phase 4/5.

</deferred>

---

*Phase: 1-Khung xương end-to-end tối thiểu + vá lỗi chặn*
*Context gathered: 2026-09-27*
