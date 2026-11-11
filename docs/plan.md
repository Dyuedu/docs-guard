
# Kế hoạch triển khai E2E — doc-guard

## Context

Repo đã có phần khung: tiền/hậu xử lý, KIE rule, metrics, export ONNX/INT8, Triton BLS, gateway, Locust (35 test pass). Phần còn thiếu:

- dữ liệu,
- code model và code train,
- trọng số,
- baseline B0,
- harness đánh giá,
- kết quả B0–B5.

Mục tiêu cuối: `docker compose up` chạy được pipeline thật trên T4, kèm bảng B0–B5, biểu đồ accuracy–latency và báo cáo Locust. Các mục tiêu số nằm ở proposal §2.

Các quyết định đã chốt:

- **Tự train từ đầu** DBNet, CRNN và CAT-Net-lite (MobileNetV3) bằng PyTorch.
- **Thuê T4** cho cả train lẫn benchmark. Local RTX 3060 chỉ dùng để dev/debug với dữ liệu nhỏ.

Rủi ro chính đã biết:

1. MC-OCR chỉ gán nhãn 4 trường, không gán toàn bộ dòng chữ, và chỉ khoảng 1.1k ảnh train có nhãn. Tập test public không có nhãn KIE. Vì vậy cần **dữ liệu tổng hợp** và **pseudo-label**.
2. Nhánh DCT của CAT-Net phải export được ONNX, nên DCT được làm bằng conv cố định thay vì đọc hệ số từ file JPEG.

---

## Giai đoạn 0 — Nền móng (0,5 tuần)

- `git init` commit đầu tiên. Thêm vào `.gitignore`: `data/`, `artifacts/`, `mlruns/`, `results/`, `*.plan`, `.venv`.
- Thêm `src/docguard/config.py` để load `configs/pipeline.yaml` và `configs/experiments.yaml` dùng chung.
- Thêm `configs/train/*.yaml`, mỗi model một file: hyperparams, đường dẫn data, seed.
- Viết CI tối thiểu (GitHub Actions): `pytest -q` trên CPU.
- Viết `scripts/setup_cloud.sh` cho máy T4: driver, docker, nvidia-container-toolkit, clone repo, tải data từ bucket hoặc Drive, `pip install -e ".[train,export,dev]"`.

## Giai đoạn 1 — Dữ liệu (1,5 tuần)

**1.1 MC-OCR**

- Tải về `data/raw/mcocr/`.
- Kiểm tra tên cột CSV với `convert_mcocr_csv` ([annotations.py](src/docguard/data/annotations.py)), sửa nếu lệch.
- Viết test với 2–3 dòng CSV mẫu.

**1.2 Split chuẩn dùng chung (mới)**

- Viết `scripts/make_splits.py` để sinh `data/processed/splits.json` (train/val/test theo `img_id`, 80/10/10, seed cố định).
- Chỉ lấy từ tập train có nhãn của MC-OCR.
- Sửa [make_tamper_dataset.py](scripts/make_tamper_dataset.py) để đọc `splits.json` thay vì tự shuffle. Như vậy OCR/KIE và tampering dùng **cùng một split**, không bị rò rỉ dữ liệu.

**1.3 Ảnh giả mạo**

- Chạy `make_tamper_dataset.py`, dùng font có dấu tiếng Việt (Roboto/Arial Unicode) đặt trong `data/fonts/`.
- Kiểm tra thủ công khoảng 50 cặp ảnh/mask.
- Thống kê tỉ lệ từng phép biến đổi và diện tích mask.

**1.4 Dữ liệu text tổng hợp cho OCR (mới)** — `src/docguard/data/synth_text.py`, `scripts/make_synth_ocr.py`

- **Hóa đơn tổng hợp toàn trang** cho DBNet:
  - render các dòng (tên cửa hàng, địa chỉ, món, giá, tổng, ngày) bằng font hóa đơn (mono/thermal) lên nền giấy;
  - augment: blur, nhiễu, perspective, nhăn giấy, đổ bóng, JPEG;
  - ghi polygon cho **mọi** dòng chữ. Mục tiêu khoảng 20k ảnh.
- **Ảnh dòng chữ** cho CRNN, 32×320:
  - corpus tiếng Việt gồm tên món, số tiền, ngày, địa chỉ;
  - charset phải khớp `configs/vocab_vi.txt` (221 ký tự).
  - Mục tiêu khoảng 1–2M dòng, sinh on-the-fly trong DataLoader để khỏi tốn đĩa.
- Tùy chọn: VinText (scene text tiếng Việt, giấy phép nghiên cứu) để tăng đa dạng cho DBNet và CRNN.

**1.5 Pseudo-label MC-OCR**

- Sau khi có DBNet v1, sinh box cho mọi dòng chữ trên MC-OCR rồi hợp nhất với box GT của 4 trường. Đây là tập fine-tune cho DBNet v2.
- Phần text cho CRNN chỉ lấy từ crop GT, không dùng pseudo-label.

## Giai đoạn 2 — Model và training (3–4 tuần, trên T4)

Code chung: `src/docguard/models/`, `src/docguard/train/`, `scripts/train.py --config configs/train/<model>.yaml`.

- Dùng AMP, checkpoint theo metric trên tập val, log MLflow (params, metrics, artifact `.pt`).
- **Ràng buộc export:** mọi model phải có đúng tên và shape I/O như các file `serving/model_repository/*/config.pbtxt`, và có test `export_onnx` cộng parity ngay khi viết model.

**2.1 text_det — DBNet (MobileNetV3-large + FPN + DB head)** — `models/dbnet.py`

- Dataset:
  - shrink polygon sinh prob map, threshold map;
  - random crop/scale, input 960×960 letterbox như `P.det_preprocess`.
- Loss: BCE (OHEM 3:1) + L1 threshold + dice binary.
- Đầu ra khi export: **chỉ prob map** `prob [N,1,960,960]`, khớp với `P.det_postprocess`.
- Lộ trình train:
  1. v1 trên hóa đơn tổng hợp;
  2. pseudo-label (1.5);
  3. v2 fine-tune trên MC-OCR cộng tổng hợp.
- Metric val: P/R/H-mean theo IoU 0.5. Viết thêm `metrics/detection.py`.

**2.2 text_rec — CRNN + CTC** — `models/crnn.py`

- Kiến trúc: backbone CNN (VGG-lite hoặc MobileNetV3-small) → BiLSTM ×2 → Linear(C = len(vocab)+1, blank = 0).
- Output `logits [N,T,C]`, khớp `P.ctc_greedy_decode`.
- BiLSTM export ONNX với opset 17 là được. Nếu TensorRT INT8 gặp vấn đề với LSTM thì có phương án dự phòng: thay bằng Conv1D hoặc Transformer encoder nhỏ.
- Lộ trình train:
  1. pretrain trên dòng chữ tổng hợp;
  2. fine-tune trên crop MC-OCR (4 trường) cộng 30% dữ liệu tổng hợp để tránh quên.
- Metric val: CER (`metrics/ocr.py:cer`).

**2.3 tamper_seg — CAT-Net-lite** — `models/catnet_lite.py`

- **Nhánh RGB:** MobileNetV3-large pretrained ImageNet (timm hoặc torchvision), lấy feature đa tầng.
- **Nhánh DCT:**
  - chuyển sang Y channel;
  - conv cố định 64 filter 8×8, stride 8 (cơ sở DCT, không train);
  - clip |coef| ≤ 20;
  - one-hot hoặc embedding histogram;
  - vài block conv nhẹ.
  - Toàn bộ đều là op conv thường nên export ONNX/TensorRT được.
- **Fusion:** concat ở stride 8/16, decoder kiểu FPN/LR-ASPP → `mask_logits [N,1,512,512]`.
- **Tùy chọn:** thêm SRM 3 filter cố định vào nhánh RGB (ablation).
- **Ràng buộc lưới 8×8:**
  - `patch 512 / stride 384` đã chia hết cho 8;
  - ảnh bị downscale bởi `max_long_side` sẽ mất căn lưới, nên cần log tỉ lệ ảnh bị scale trong đánh giá.
- **Dataset:**
  - patch 512 ở độ phân giải gốc, sample ưu tiên patch có mask (khoảng 50%);
  - augment **không resize**, chỉ flip;
  - JPEG re-compress đã có sẵn trong dữ liệu.
- **Loss:** BCE + Dice, có pos_weight vì vùng giả mạo nhỏ.
- **Metric val:** pixel-F1, IoU, image AUC tính trên **ảnh đầy đủ** qua `P.tile`/`P.stitch`, giống hệt cách serving chạy.
- **Tùy chọn:** pretrain trên DocTamper (chỉ nghiên cứu).

## Giai đoạn 3 — Đánh giá offline và baseline B0 (1 tuần)

**3.1 `src/docguard/eval/` + `scripts/evaluate.py --variant Bx --split test`**

- Backend runner dạng plugin, cùng một interface `run(model_name, batch) -> np.ndarray`:
  - `TorchRunner` (B0),
  - `OrtRunner` (B1),
  - `TrtRunner` (B2–B5, dùng `tensorrt` hoặc Triton client).
- Tái sử dụng **nguyên** logic của `pipeline/1/model.py`. Tách phần điều phối thành `src/docguard/pipeline/orchestrator.py`, nhận vào một hàm `infer(model, batch)`. Kết quả là BLS, eval offline và B0 dùng chung một code path, nên chênh lệch chỉ đến từ runtime.
- Đầu ra: pixel-F1, IoU, AUC, CER (trên crop GT), Field EM, det H-mean. Ghi `results/<variant>/metrics.json` và log MLflow.

**3.2 Baseline B0 serving** — `serving/baseline/app.py`

- FastAPI + PyTorch FP32, **cùng API `/v1/verify`** và cùng schema `VerifyResponse` ([schemas.py](src/docguard/schemas.py)).
- Dùng `orchestrator` với `TorchRunner`. Thêm service `baseline` (port 8081) vào docker-compose.

**3.3 Chốt ngưỡng**

- Tune `decision.tampered_area_ratio` và `tampered_score` trên tập val, rồi cập nhật `configs/pipeline.yaml`.
- Chốt ngưỡng p95 E2E tuyệt đối sau khi đo B0, như proposal yêu cầu.

## Giai đoạn 4 — Export và tối ưu B1–B5 (1,5 tuần)

- `scripts/export_all.py`:
  - chạy `export_onnx` cho 3 model (`dynamic_batch=True`, opset 17);
  - kiểm tra `max_abs_diff` so với PyTorch;
  - ghi `artifacts/{name}.onnx`.
- **B1:** ORT FP32. Đổi platform sang `onnxruntime_onnx`. Thêm thư mục variant `serving/variants/B*/` cùng `scripts/switch_variant.sh` để copy config và engine vào `model_repository`.
- **B2:** TensorRT FP16 cho cả 3 model qua `build_trt.sh`. Tham số hóa script theo variant: `build_trt.sh B2|B3|B4|B5`.
- **B3:** INT8 PTQ.
  - Lấy 500 ảnh calibration **từ tập train** qua `quantize_int8_static` (QDQ), rồi build TensorRT `--int8`.
  - Viết `src/docguard/export/calib.py`, sinh batch đã tiền xử lý đúng như serving.
  - Nếu CAT-Net INT8 mất quá nhiều độ chính xác, thử giữ nhánh DCT/đầu ra ở FP16 (exclude nodes).
- **B4:** pruning 30% bằng torch-pruning trên backbone, fine-tune ngắn (khoảng 10% số epoch), sau đó INT8. Viết `scripts/prune.py`.
- **B5:** tamper FP16 + OCR INT8. Đây đúng là cấu hình mặc định hiện tại của `build_trt.sh`.
- Với mỗi variant, chạy `evaluate.py` (accuracy) và `bench/latency.py` (p50/p95, peak VRAM qua `nvidia-smi`/pynvml, kích thước file), log MLflow theo `configs/experiments.yaml`.

## Giai đoạn 5 — Serving E2E và benchmark hệ thống (1 tuần)

- **Triton:**
  - `docker compose up` với engine B5;
  - smoke test gRPC trên 20 ảnh test, so sánh JSON với `evaluate.py` (phải khớp field và gần khớp bbox);
  - viết `tests/test_e2e_triton.py`, mark `gpu`, mặc định skip.
- **Integration:**
  - kiểm tra gateway → Triton với ảnh thật;
  - kiểm tra xử lý lỗi (ảnh hỏng, ảnh > 2048, không tìm thấy text).
- **Locust:**
  - concurrency 1/8/32/64;
  - so sánh 3 cấu hình: B0 FastAPI+PyTorch, Triton có dynamic batching, Triton không dynamic batching (thêm config variant);
  - ghi QPS, p95, p99, GPU util (Prometheus :8002);
  - lưu CSV vào `results/locust/`.
- Ghi rõ phiên bản driver, CUDA, TensorRT (24.08) và Triton.

## Giai đoạn 6 — Báo cáo (0,5 tuần)

- `scripts/report.py`: đọc MLflow hoặc `results/`, sinh bảng B0–B5 (markdown), biểu đồ accuracy–latency (pixel-F1 và Field EM theo latency), biểu đồ Locust.
- Trả lời câu hỏi nghiên cứu: INT8 làm pixel-F1 giảm bao nhiêu so với CER (B3 vs B5).
- Cập nhật README: kết quả, giới hạn, cách tái lập. Đánh dấu Roadmap.

**Tổng thời gian:** khoảng 9–10 tuần. Thời gian thuê T4 ước tính 60–100 giờ, tập trung ở giai đoạn 2 và 4.

---

## File mới hoặc cần sửa (tóm tắt)

- **Mới:**
  - `src/docguard/{config.py, models/{dbnet,crnn,catnet_lite}.py, train/, eval/, data/synth_text.py, pipeline/orchestrator.py, export/calib.py, metrics/detection.py}`
  - `scripts/{make_splits,make_synth_ocr,train,export_all,prune,evaluate,report,setup_cloud}.py|sh`
  - `serving/baseline/`, `serving/variants/`, `configs/train/`
- **Sửa:**
  - `scripts/make_tamper_dataset.py` (đọc splits.json)
  - `serving/model_repository/pipeline/1/model.py` (gọi orchestrator)
  - `scripts/build_trt.sh` (tham số variant)
  - `docker-compose.yml` (service baseline)
  - `pyproject.toml` (timm, pynvml, tensorrt tùy chọn)
- **Tái sử dụng:**
  - toàn bộ `pipeline/processing.py`, `kie/rules.py`, `metrics/*`, `export/onnx_export.py`, `bench/latency.py`, `data/synth_tamper.py`, `schemas.py`.

## Verification theo từng mốc

1. **Data:** `pytest` cho converter, splits (không trùng `img_id` giữa các split), synth_text (charset ⊆ vocab). Xem trực quan ảnh/mask mẫu.
2. **Model:** mỗi model có unit test forward shape, `export_onnx` và parity < 1e-3. Overfit 10 mẫu trên local 3060 trước khi train trên T4.
3. **Train:** đường cong MLflow. Ngưỡng tạm để đi tiếp: det H-mean ≥ 0.85, CER ≤ 5% trên crop GT, pixel-F1 ≥ 0.7 trên tập synth test.
4. **Eval:** `evaluate.py --variant B0` cho ra đủ metric. Kết quả B1 lệch so với B0 dưới 0,1 điểm (kiểm tra parity runtime).
5. **Serving:** `curl -F image=@x.jpg localhost:8080/v1/verify` trả JSON đúng schema. Kết quả Triton khớp `evaluate.py`.
6. **Mục tiêu cuối:** đối chiếu bảng B0–B5 và Locust với ngưỡng ở proposal §2: latency giảm ≥ 50%, VRAM giảm ≥ 40%, ΔF1 và ΔEM ≤ 1,5 điểm, throughput ≥ 2× tại 32 users.
