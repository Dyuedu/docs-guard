# Intent 003 — DBNet text detection v1 (Phase 2.1)

Status: Accepted
Date: 2026-11-12
Owner: hoangduy20407
Proposal ref: docs/proposal.md §5, §6, §7; docs/plan.md giai đoạn 2.1

## Problem
Pipeline OCR cần bước **tìm dòng chữ** trước khi CRNN đọc. Hiện repo mới có:
- phần quanh model: `P.det_preprocess` (letterbox 960, căn trên-trái), `P.det_postprocess` (prob map → box) và `serving/model_repository/text_det/config.pbtxt` (`image [3,960,960]` → `prob [1,960,960]`);
- dữ liệu train: 20k + 1k hóa đơn tổng hợp có polygon cho mọi dòng (intent 002).

Còn thiếu model, vòng train, metric detection và checkpoint. DBNet là đường găng của giai đoạn 2:
- DBNet v1 → pseudo-label MC-OCR (1.5) → DBNet v2.
- Baseline B0 của `text_det` cũng cần checkpoint này.

## Proposed outcome
- `src/docguard/models/dbnet.py`: MobileNetV3-large + FPN + DB head. Khi train trả về prob, threshold và binary map. Khi export **chỉ trả `prob`**, đúng tên và shape trong `config.pbtxt`.
- Phần train dùng chung cho các model sau (`src/docguard/train/`, `scripts/train.py --config configs/train/text_det.yaml`):
  - dataset sinh shrink map và threshold map từ polygon;
  - loss BCE (OHEM 3:1) + L1 threshold + dice binary;
  - AMP, giữ checkpoint tốt nhất theo H-mean val, log MLflow.
- `src/docguard/metrics/detection.py`: P/R/H-mean theo IoU 0.5.
- **DBNet v1** train trên hóa đơn tổng hợp, có MLflow run id, và đạt các mức sau:
  - H-mean trên synth val đạt ngưỡng (xem Open questions);
  - đo recall trên box GT 4 trường của MC-OCR val, chạy qua đúng `P.det_preprocess`/`P.det_postprocess` như serving.
- Test `export_onnx` + parity (PyTorch vs ONNX Runtime) viết **cùng lúc** với model, đúng ràng buộc của plan §2.

**Ngoài phạm vi** (intent sau): pseudo-label MC-OCR (1.5), fine-tune DBNet v2, build TensorRT, INT8, pruning.

## Affected components
- Mới:
  - `src/docguard/models/{__init__,dbnet}.py`;
  - `src/docguard/train/` (dataset det, loss, vòng train chung);
  - `src/docguard/metrics/detection.py`;
  - `scripts/train.py`;
  - test tương ứng.
- Sửa:
  - `configs/train/text_det.yaml`: epochs, kích thước crop khi train, backbone pretrained (xem Open questions);
  - có thể sửa `src/docguard/export/onnx_export.py` nếu cần hook `export_mode`.
- Sửa (bổ sung 2026-11-12, owner duyệt ở spec, Flagged concern 1 phương án A):
  - `src/docguard/pipeline/processing.py`: `det_postprocess` unclip chính xác (nghịch đảo công thức shrink);
  - `configs/pipeline.yaml` (`unclip_ratio` → `shrink_ratio`) và `serving/model_repository/pipeline/1/model.py`;
  - `.github/workflows/ci.yml`: cài torch CPU cho test model.
- Đọc, không sửa:
  - `src/docguard/pipeline/processing.py` (`det_preprocess`);
  - `serving/model_repository/text_det/config.pbtxt`;
  - `data/synth_ocr/receipts/`, `data/processed/{annotations.jsonl,splits.json}`.
- Môi trường: `.venv` hiện **chưa có torch**, cần `pip install -e ".[train,export]"`.

## Constraints
- **I/O cố định:** input `image` FP32 `[N,3,960,960]` (letterbox căn trên-trái, chuẩn hóa như `to_chw`), output `prob` FP32 `[N,1,960,960]` đã qua sigmoid. ONNX opset 17. Batch động đến 8.
- **Chỉ dùng op export được** sang ONNX/TensorRT. Ví dụ, upsample dùng `nearest` hoặc `bilinear` với `scale_factor` cố định, không dùng op tùy biến. INT8 làm ở giai đoạn 4, nhưng không được chọn kiến trúc chặn đường INT8.
- **Không rò rỉ dữ liệu:**
  - v1 chỉ train trên dữ liệu tổng hợp;
  - mọi đánh giá trên MC-OCR dùng split val trong `splits.json`;
  - test split chỉ dùng cho B0 ở giai đoạn 3.
- **MC-OCR chỉ gán nhãn 4 trường:** precision đo trên MC-OCR sẽ sai, vì các dòng chữ không có nhãn bị tính là false positive. Trên MC-OCR chỉ báo **recall**. P/R/H-mean đầy đủ chỉ đo trên synth val.
- **Phần cứng:** train trên RTX 3090 24 GB thuê (15.000 đ/giờ, AMP bật), ngân sách khoảng 2–3,5 giờ cho 100k iteration. Laptop RTX 3060 6 GB chỉ dùng cho test và debug ngắn. AMP phải bật/tắt được qua config.
- **Metric cần run id:** theo CLAUDE.md, không báo metric nào mà không kèm MLflow run id. Đây là checkpoint làm B0 nên chưa có gì để so với B0.
- **Giấy phép:** chỉ port code từ repo Apache-2.0 và ghi attribution trong comment và NOTICE. Repo gốc `MhLiao/DB` **không có file LICENSE**, nên chỉ tham khảo thuật toán (bài báo), không chép code.

## Research (đã làm)
| Repo | License | Dùng được gì |
|---|---|---|
| [PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) | Apache-2.0 | **Tham khảo chính về cấu hình:** `configs/det/det_mv3_db.yml` dùng đúng MobileNetV3 + DB, với shrink 0.4, thresh 0.3–0.7, OHEM 3, α=5, β=10, Adam lr 1e-3. Có `ppocr/data/imaug/{make_shrink_map,make_border_map,random_crop_data,iaa_augment}.py`, `losses/det_db_loss.py`, `necks/db_fpn.py` và `metrics/eval_det_iou.py`. Code viết bằng Paddle, phải port sang PyTorch, nhưng các file sinh label chỉ dùng numpy/cv2/pyclipper. |
| [open-mmlab/mmocr](https://github.com/open-mmlab/mmocr) | Apache-2.0 | DBNet viết bằng PyTorch (`DBHead`, `DBModuleLoss`, `FPNC`, có `asf`). Phụ thuộc mmcv/mmengine nên không dùng thẳng, chỉ đối chiếu cách viết loss và head bằng PyTorch. |
| [WenmuZhou/DBNet.pytorch](https://github.com/WenmuZhou/DBNet.pytorch) | Apache-2.0 | Bản PyTorch gọn, gần với PaddleOCR (tác giả cũng đóng góp cho PaddleOCR). Ngừng cập nhật từ 2022. Dùng để đối chiếu vòng train và cách export. |
| [MhLiao/DB](https://github.com/MhLiao/DB) | không có | Code gốc của bài báo DB (AAAI 2020). Chỉ đọc để đối chiếu thuật toán. |
| torchvision `mobilenet_v3_large` | BSD-3 | Backbone có sẵn weight ImageNet, export ONNX được. Feature lấy ở stride 4/8/16/32. |

Hướng dự kiến: port label generation, loss và metric từ PaddleOCR sang PyTorch; backbone lấy từ torchvision; neck và head viết theo PaddleOCR `DBFPN` + `DBHead`. Dependency mới cần duyệt: `pyclipper` (MIT), dùng để shrink polygon khi sinh label.

## Open questions
- [x] **Backbone pretrained ImageNet** (torchvision) hay train từ đầu? CRNN đã chốt train từ đầu, nhưng PaddleOCR và mmocr đều dùng backbone pretrained cho DBNet, và dữ liệu tổng hợp ít đa dạng hơn ảnh thật. **Chốt (2026-11-12): pretrained.**
- [x] **Kích thước khi train:** random crop 640×640 như PaddleOCR hay letterbox 960 như serving? **Chốt: crop 640 khi train, còn val chạy letterbox 960 qua `P.det_preprocess`.**
- [x] **Số epoch.** `text_det.yaml` đang để 300 epoch × 20k ảnh = 6M ảnh (750k iteration ở batch 8). Trên máy P40 7.000 đ/giờ, ước lượng thô là 44–86 giờ (310k–600k đ). **Chốt (2026-11-12): 100k iteration ở batch 8 (khoảng 0,8M ảnh, 40 epoch), early stop theo H-mean val.** Trước khi train thật, chạy thử 15 phút để đo tốc độ.
- [x] **Ngưỡng nghiệm thu v1.** **Chốt:**
  - H-mean trên synth val ≥ 0,90;
  - recall trên 668 box GT 4 trường của MC-OCR val ≥ 0,80 (SELLER 119, ADDRESS 202, TIMESTAMP 139, TOTAL_COST 208);
  - cả hai đo qua `det_postprocess`, khớp 1–1 theo IoU ≥ 0,5.
- [x] **IoU theo polygon hay box thẳng?** **Chốt: box thẳng.** Lý do: `det_postprocess` (serving) trả box thẳng. Polygon GT được quy về bbox bao ngoài; polygon chỉ dùng để sinh label khi train.
- [x] **Train ở đâu:** **Chốt: máy thuê.** RTX 3060 chỉ dùng chạy test và debug ngắn.
- [x] **Loại GPU của máy thuê.** Máy 7.000 đ/giờ (24 GB / 3840 CUDA core / 12 TFLOPS) khớp **Tesla P40** (Pascal), không phải T4. P40 gần như không có FP16 nên không dùng được AMP. **Chốt: nếu máy đó không phải T4 thì train trên RTX 3090 (24 GB, 15.000 đ/giờ), bật AMP.** Ước lượng thô cho 100k iteration trên 3090: 80–150 ảnh/s, khoảng 2–3,5 giờ kể cả val và setup, tức khoảng **30k–55k đ**. Như vậy 3090 vẫn rẻ hơn P40 (6–12 giờ, 42k–84k đ). Benchmark B0–B5 (giai đoạn 3–5) vẫn phải đo trên T4 thật như proposal; GPU dùng để train không ảnh hưởng checkpoint.
