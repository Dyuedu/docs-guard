# doc-guard

Microservice phát hiện chỉnh sửa (tampering localization) và bóc tách thông tin (OCR + KIE) cho ảnh hóa đơn tiếng Việt. Mô hình được tối ưu bằng ONNX / TensorRT FP16 / INT8 và phục vụ trên Triton Inference Server qua gRPC.

Đề xuất đầy đủ (bài toán, mục tiêu, thiết kế thí nghiệm): [docs/proposal.md](docs/proposal.md)

```
client ──REST──► FastAPI gateway ──gRPC──► Triton
                                            └─ pipeline (Python BLS, CPU)
                                                 ├─ tamper_seg  (patch 512², FP16)   ┐ chạy
                                                 └─ text_det → crop → text_rec (INT8) ┘ song song
                                                 → KIE rules → decision → JSON
```

## Cấu trúc

| Đường dẫn | Nội dung |
|---|---|
| `src/docguard/data/` | Chuyển MC-OCR sang jsonl chuẩn hóa; sinh giả mạo copy-move / splicing / inpaint kèm mask pixel |
| `src/docguard/pipeline/processing.py` | Tiền/hậu xử lý dùng chung giữa Triton và đánh giá offline (tiling, stitch, DB postprocess, CTC decode) |
| `src/docguard/kie/rules.py` | Rule KIE: `seller`, `timestamp`, `total_amount` |
| `src/docguard/metrics/` | Pixel-F1, IoU, AUC, CER, Field Exact Match |
| `src/docguard/export/` | Export ONNX, chuyển FP16, INT8 static PTQ (QDQ), kiểm tra parity |
| `src/docguard/bench/latency.py` | Đo latency p50/p95/p99 ở mức model, log lên MLflow |
| `serving/model_repository/` | Config Triton cho 3 model + model `pipeline` dùng BLS |
| `serving/gateway/` | FastAPI gateway (validate upload, gọi gRPC, ép schema output) |
| `scripts/` | Sinh dataset giả mạo, build TensorRT engine |
| `loadtest/locustfile.py` | Bắn tải hệ thống |
| `configs/` | Ngưỡng pipeline, ma trận thí nghiệm B0–B5, vocab tiếng Việt |

## Bắt đầu

```bash
python -m venv .venv && .venv/Scripts/activate      # Linux: source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

### 1. Dữ liệu

```bash
# Tải MC-OCR 2021 vào data/raw/mcocr/, rồi:
python -c "from docguard.data.annotations import convert_mcocr_csv as c; \
  print(c('data/raw/mcocr/mcocr_train_df.csv', 'data/raw/mcocr/train_images', 'data/processed/annotations.jsonl'))"

# Split chuẩn theo ảnh gốc, dùng chung cho OCR/KIE và tampering
python scripts/make_splits.py --ann data/processed/annotations.jsonl --out data/processed/splits.json

# Sinh tập giả mạo (font: 1 file .ttf hoặc thư mục để xoay vòng; cần đủ glyph tiếng Việt)
python scripts/make_tamper_dataset.py --ann data/processed/annotations.jsonl \
  --splits data/processed/splits.json --images data/raw/mcocr/train_images \
  --out data/tamper --font data/fonts --tampered-per-image 2 --tampered-per-image-train 4

# Kiểm định: rò rỉ split, mask hợp lệ, tỉ lệ phép biến đổi, diện tích mask, ảnh preview (exit 1 nếu FAIL)
python scripts/tamper_report.py --root data/tamper --splits data/processed/splits.json \
  --out results/data/tamper --preview 50
```

Converter đã được kiểm tra với bản public MC-OCR (cột `img_id`, `anno_polygons`, `anno_texts`, `anno_labels`). MC-OCR chỉ được dùng cho nghiên cứu, cần đăng ký với VNDAG (xem `data/raw/mcocr/README_VNDAG.md`).

### 2. Train → Export → Optimize

```bash
pip install -e ".[train,export]"
# train models (chưa có, xem Roadmap) → artifacts/*.onnx
# INT8: docguard.export.onnx_export.quantize_int8_static(...)
docker compose run --rm triton bash /workspace/scripts/build_trt.sh
```

### 3. Serving và benchmark

```bash
docker compose up --build
curl -F image=@receipt.jpg -F request_id=req_1 http://localhost:8080/v1/verify

locust -f loadtest/locustfile.py --host http://localhost:8080 --users 32 --spawn-rate 8 \
  --run-time 3m --headless --csv results/triton_u32
```

Cần GPU NVIDIA và NVIDIA Container Toolkit. Colab/Kaggle không chạy được Docker, nên phải thuê GPU T4 theo giờ để có số liệu benchmark.

## Roadmap

- [x] Đề xuất, schema, rule KIE, metrics, sinh dữ liệu giả mạo
- [x] Tiền/hậu xử lý, Triton BLS pipeline, FastAPI gateway, Docker Compose, Locust
- [ ] Train tamper_seg (CAT-Net-lite / SegFormer-B0 + SRM), DBNet, CRNN
- [ ] Baseline B0: FastAPI + PyTorch FP32 (cùng API) để so sánh
- [ ] Ma trận B1–B5, bảng kết quả và biểu đồ accuracy–latency
- [ ] Ablation: Torch-Pruning cho backbone
- [ ] Mở rộng: CCCD tổng hợp + đối chiếu QR, phát hiện ảnh chụp lại màn hình

## Giới hạn

- Dữ liệu giả mạo là tự sinh, chưa kiểm chứng trên gian lận thật.
- Không dùng dữ liệu cá nhân thật (Nghị định 13/2023/NĐ-CP).
- DocTamper (nếu dùng để pretrain) chỉ có giấy phép nghiên cứu.
