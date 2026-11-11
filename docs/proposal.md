# Đề tài: Tối ưu hóa Pipeline phát hiện chỉnh sửa và bóc tách thông tin hóa đơn với ONNX/TensorRT và Triton Inference Server

## 1. Pain Point

Trong luồng giải ngân online, khách hàng tải lên ảnh hóa đơn / chứng từ thu nhập. Có hai vấn đề:

- **Gian lận**: kẻ gian sửa con số (số tiền, ngày) bằng copy-move, splicing hoặc inpainting.
- **Chi phí hạ tầng**: pipeline phát hiện chỉnh sửa + OCR chạy tuần tự trên PyTorch FP32 tốn VRAM, latency cao và không tận dụng được batching khi lượng request tăng vọt.

## 2. Problem Statement

Xây dựng microservice nhận ảnh hóa đơn, làm hai việc:

1. Khoanh vùng các vùng chữ bị chỉnh sửa ở mức pixel.
2. Bóc tách các trường có cấu trúc: tên cửa hàng (`seller`), ngày (`timestamp`), tổng tiền (`total_amount`).

Mô hình được tối ưu bằng FP16/INT8 quantization (structured pruning là ablation), xuất sang ONNX/TensorRT và phục vụ trên Triton qua gRPC.

**Mục tiêu** — so với baseline PyTorch FP32 do chính đề tài đo trên cùng GPU T4:

| Mục tiêu | Ngưỡng |
|---|---|
| Model latency (tổng các mô hình) | giảm ≥ 50% |
| Peak VRAM | giảm ≥ 40% |
| Pixel-F1 (tampering) | giảm ≤ 1,5 điểm % |
| Field Exact Match (OCR + KIE) | giảm ≤ 1,5 điểm % |
| Throughput tại 32 concurrent users | ≥ 2× so với FastAPI + PyTorch thuần |

Ngưỡng tuyệt đối cho p95 end-to-end chỉ chốt **sau khi đo baseline**, không đặt trước.

### Phạm vi

- **Trong phạm vi**: hóa đơn tiếng Việt (MC-OCR 2021), giả mạo tự sinh, ảnh đầu vào có cạnh dài ≤ 2048 px.
- **Ngoài phạm vi** (mở rộng sau): CCCD (kèm đối chiếu mã QR), phát hiện ảnh chụp lại màn hình, ảnh 4K.

## 3. Input / Output

**Input**: gRPC (Triton) hoặc REST `multipart/form-data` qua FastAPI. Gồm `image` (JPEG/PNG bytes) và `request_id`.

**Output**:

```json
{
  "request_id": "req_98234",
  "is_authentic": false,
  "decision_reason": "tampered_area_ratio > 0.002",
  "tampering_result": {
    "tampered_score": 0.91,
    "tampered_area_ratio": 0.0041,
    "tampered_bboxes": [[120, 340, 280, 390]],
    "overlaps_fields": ["total_amount"]
  },
  "extracted_fields": {
    "seller": {"value": "CO.OP MART", "confidence": 0.97},
    "timestamp": {"value": "15/08/2021", "confidence": 0.95},
    "total_amount": {"value": "125000", "confidence": 0.88}
  },
  "latency_ms": {"preprocess": 6.1, "tampering": 14.2, "ocr": 18.7, "total": 41.3}
}
```

`overlaps_fields` cho biết vùng bị sửa đè lên trường nào — tín hiệu nghiệp vụ quan trọng nhất.

## 4. Dữ liệu

- **OCR/KIE**: MC-OCR 2021, chia train/val/test theo nguyên bản.
- **Tampering**: tự sinh từ MC-OCR (`scripts/make_tamper_dataset.py`):
  - copy-move chữ số trong cùng ảnh;
  - splicing chữ số từ ảnh khác;
  - inpainting xóa chữ rồi render chữ mới bằng font tương tự;
  - re-compress JPEG chất lượng 70–95 để mô phỏng thực tế.

  Mask pixel được sinh tự động trong quá trình tạo.
- **Pretrain tampering** (tùy chọn): DocTamper — giấy phép chỉ cho nghiên cứu.
- **Tập test tách theo ảnh gốc**: cùng một hóa đơn không xuất hiện ở cả train và test.
- Không dùng dữ liệu cá nhân thật (Nghị định 13/2023/NĐ-CP).

## 5. Kiến trúc

| Thành phần | Lựa chọn | Lý do |
|---|---|---|
| Tampering | CAT-Net-lite (MobileNetV3) hoặc SegFormer-B0 + nhánh nhiễu SRM | Học artifact cấp thấp, không học ngữ nghĩa |
| Text detection | DBNet (MobileNetV3) | Nhẹ, export ONNX ổn định |
| Text recognition | CRNN + CTC | Non-autoregressive, dễ INT8 và dynamic batching |
| KIE | Rule + regex theo vị trí (baseline); LayoutLMv3 (tùy chọn) | |
| Serving | Triton: Python BLS điều phối tiền xử lý → tamper_seg / text_det → crop → text_rec | BLS xử lý số crop thay đổi theo ảnh |
| Gateway | FastAPI → Triton gRPC | |
| MLOps | MLflow, pytest, Locust, Docker Compose | |

Nhánh tampering chạy trên **patch ở độ phân giải gốc** (không resize) để giữ dấu vết chỉnh sửa.

## 6. Thiết kế thí nghiệm

| # | Biến thể |
|---|---|
| B0 | PyTorch FP32 (baseline) |
| B1 | ONNX Runtime FP32 |
| B2 | TensorRT FP16 |
| B3 | TensorRT INT8 (PTQ, calibration 500 ảnh) |
| B4 | Pruning 30% + INT8 |
| B5 | INT8 cho OCR, FP16 cho tampering (mixed) |

Mỗi biến thể ghi: dung lượng file, model latency p50/p95, peak VRAM, pixel-F1, CER, Field EM.

**Câu hỏi nghiên cứu chính**: INT8 và resize ảnh hưởng đến pixel-F1 tampering nhiều hơn CER của OCR bao nhiêu? B5 trả lời câu này.

**Benchmark hệ thống**: Locust với concurrency 1/8/32/64 — FastAPI + PyTorch vs Triton (bật/tắt dynamic batching); ghi QPS, p95, p99, GPU utilization.

## 7. Metrics

| Nhóm | Metric | Định nghĩa |
|---|---|---|
| Tampering | Pixel-F1, IoU | Mask nhị phân ngưỡng 0,5, trung bình theo ảnh |
| Tampering | Image-level AUC | Phân loại thật/giả theo `tampered_score` |
| OCR | CER | Edit distance / độ dài ground truth, trên crop đúng vị trí |
| KIE | Field Exact Match | Khớp tuyệt đối sau chuẩn hóa (bỏ khoảng trắng, chuẩn hóa số) |
| Hệ thống | p50/p95/p99, QPS, peak VRAM | Trên T4; ghi rõ phiên bản driver, TensorRT, Triton |

## 8. Sản phẩm bàn giao

- Repo chạy được bằng `docker compose up`.
- Bảng kết quả B0–B5 và biểu đồ accuracy–latency.
- Báo cáo Locust.
- README ghi rõ giới hạn: dữ liệu giả mạo tự sinh, chưa kiểm chứng trên gian lận thật.
