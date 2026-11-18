# Plan 003 — DBNet text detection v1

Status: Approved
Date: 2026-11-12
Spec: ./spec.md

## Files to change
| File | Thay đổi |
|---|---|
| `NOTICE` (mới) | Attribution PaddleOCR và mmocr (Apache-2.0), kèm danh sách file được port |
| `pyproject.toml` | Thêm `pyclipper>=1.3` vào extra `train` |
| `.github/workflows/ci.yml` | Thêm bước cài `torch`/`torchvision` CPU (`--index-url https://download.pytorch.org/whl/cpu`), `onnx`, `onnxruntime`, `pyclipper` (Flagged concern 2) |
| `src/docguard/pipeline/processing.py` | `det_postprocess(prob, ratio, bin_threshold, box_threshold, shrink_ratio, image_shape)`: unclip chính xác bằng nghiệm dương của (8 − 4k)d² + 2(1 − k)(w′ + h′)d − k·w′h′ = 0, với k = 1 − r²; lọc kernel `< 2` px (Flagged concern 1A) |
| `configs/pipeline.yaml` | `unclip_ratio: 1.5` → `shrink_ratio: 0.4` |
| `serving/model_repository/pipeline/1/model.py` | Truyền `d["shrink_ratio"]` |
| `configs/train/text_det.yaml` | Theo spec §Config: `iters`, `pretrained`, crop/scale/rotate, `loss_weights.prob: 5.0`, `eval_every`, `early_stop_patience`, `num_workers: 16` |
| `src/docguard/models/{__init__,dbnet}.py` (mới) | `DBNet`: backbone torchvision, DBFPN và DBHead port từ PaddleOCR |
| `src/docguard/train/__init__.py`, `det_labels.py`, `det_dataset.py`, `det_loss.py`, `eval_det.py`, `loop.py` (mới) | Sinh label (port Paddle), dataset và augment, `DBLoss` (theo mmocr), `evaluate_det`, vòng train chung (AMP, poly LR, eval, checkpoint, early stop, MLflow) |
| `src/docguard/metrics/detection.py` (mới) | `box_iou`, `match_boxes`, `detection_prf` |
| `scripts/train.py`, `scripts/eval_det.py` (mới) | CLI như spec |
| `tests/test_processing.py` | Thêm test unclip chính xác; sửa lời gọi `det_postprocess` hiện có cho đúng tham số mới |
| `tests/test_dbnet.py`, `test_det_labels.py`, `test_det_loss.py`, `test_detection_metric.py`, `test_eval_det.py`, `test_train_det.py` (mới) | AC-1 → AC-9 |
| `README.md` | Mục Train: lệnh train và eval DBNet, trỏ tới plan này |

## Order of work
Mỗi bước: viết test trước (chạy thấy FAIL), rồi mới implement. Expected values trong test lấy từ spec, không lấy từ output của code.

1. **Môi trường.** Cài torch CUDA vào `.venv` (`pip install -e ".[train,export,dev]"` cùng torch cu12x), thêm `pyclipper`. Chạy `pytest -q` để chắc bộ test cũ vẫn xanh.
2. **Unclip chính xác (concern 1A).**
   - Test: shrink map GT của hình chữ nhật 20×200 và 30×600 đưa vào `det_postprocess` phải cho box IoU ≥ 0,95 với GT; kernel cao 2 px không bị lọc. Chạy thấy FAIL.
   - Sửa `processing.py`, `pipeline.yaml`, `model.py` của BLS, test cũ.
   - Commit riêng.
3. **Metric** (AC-6): `test_detection_metric.py` (FAIL) → `metrics/detection.py`.
4. **Label** (AC-3, AC-4): `test_det_labels.py` (FAIL) → port `MakeShrinkMap`/`MakeBorderMap` → `det_labels.py`. Ghi comment nguồn ở đầu file; tạo `NOTICE`.
5. **Dataset** (AC-9): port `EastRandomCropData`, augment scale/rotate, đọc jsonl synth (polygon) và MC-OCR (bbox).
6. **Model** (AC-1, AC-2): `test_dbnet.py` (FAIL) → `dbnet.py`. Test export đọc `config.pbtxt`, chạy parity với ORT CPU. Kiểm index feature của torchvision bằng shape.
7. **Loss** (AC-5): `test_det_loss.py` (FAIL) → `det_loss.py`.
8. **Eval end-to-end** (AC-7): `test_eval_det.py` dùng model giả trả shrink map GT (FAIL) → `eval_det.py` + `scripts/eval_det.py`.
9. **Vòng train** (AC-8a): `test_train_det.py` smoke 2 iteration trên CPU, MLflow ghi vào `tmp_path` (FAIL) → `loop.py` + `scripts/train.py`.
10. **CI:** sửa `ci.yml`, đo thời gian test thêm (≤ 60 s). Push, kiểm CI xanh.
11. **Overfit local** (AC-8b): RTX 3060, 10 ảnh synth, ≤ 1.000 iteration, H-mean ≥ 0,90. Output và MLflow ghi vào thư mục scratch (không ghi `artifacts/`, `mlruns/` của repo). Ghi kết quả và run id vào Deviations.
12. **Train thật trên RTX 3090 thuê — owner chạy** theo các lệnh trong mục bên dưới:
    - chạy thử 15 phút, đo ảnh/s; nếu < 37 ảnh/s thì dừng và báo owner;
    - chạy đủ 100k iteration (early stop);
    - `eval_det.py` trên `synth_val` và `mcocr_val`;
    - chép `best.pt` và `mlruns/` về máy.
13. **Ghi kết quả AC-10** (H-mean, recall theo trường và theo nhóm chiều cao, run id) vào plan. Chạy verifier, rồi mở PR từ `feat/003-dbnet-text-det`.

### Lệnh trên máy thuê (bước 12)
```bash
bash scripts/setup_cloud.sh            # driver, docker, clone repo, pip install -e ".[train,export,dev]"
# chép data/synth_ocr/receipts (~1,6 GB), data/raw/mcocr và data/processed/ lên máy
nvidia-smi                             # xác nhận RTX 3090
python scripts/train.py --config configs/train/text_det.yaml --set train.iters=3000 train.eval_every=1000   # chạy thử ~15 phút
python scripts/train.py --config configs/train/text_det.yaml                                               # chạy thật
python scripts/eval_det.py --ckpt artifacts/train/text_det/<run>/best.pt --data synth_val  --out results/text_det/v1_synth_val.json  --mlflow-run-id <id>
python scripts/eval_det.py --ckpt artifacts/train/text_det/<run>/best.pt --data mcocr_val --out results/text_det/v1_mcocr_val.json --mlflow-run-id <id>
```

## Proof
- **Test:** `pytest -q` toàn bộ xanh ở local và trên CI (CI có torch CPU, nên test AC-1/2/5/8a chạy thật, không bị skip).
- **Overfit (AC-8b):** H-mean trên 10 ảnh, MLflow run id.
- **Eval (AC-10):** H-mean synth val ≥ 0,90; recall MC-OCR val ≥ 0,80 trên 668 box; recall theo trường và theo nhóm chiều cao; MLflow run id; tốc độ train (ảnh/s), VRAM tối đa, số tham số.
- **Không có so sánh B0:** checkpoint này chính là đầu vào để đo B0 ở giai đoạn 3 (trên T4).

## Risks
- **Throughput bị nghẽn ở CPU:** sinh border map bằng Python trên Xeon E5 v4 có thể không theo kịp 3090. Giảm thiểu: 16 worker, `persistent_workers`, `pin_memory`; đo ở lần chạy thử 15 phút. Nếu vẫn chậm, vector hóa `make_border_map` (vẫn giữ test AC-4).
- **Domain gap synth → real:** AC-10(b) có thể không đạt 0,80 ở v1. Ghi kết quả và để owner quyết (spec, concern 6).
- **Đổi chữ ký `det_postprocess`:** BLS và test gọi theo vị trí tham số. Phải sửa cùng commit, và grep toàn repo.
- **CI chậm hơn:** cài torch CPU mất khoảng 1–2 phút; có pip cache.
- **Tải weight ImageNet:** cần mạng khi train; trong test dùng `pretrained=False`.
- **Windows DataLoader:** worker dùng `spawn`, nên dataset phải pickle được (không giữ handle file mở). Smoke test chạy với `num_workers=0`, overfit chạy với `num_workers=2`.

## Deviations (điền khi làm xong)

### D1 — Ngưỡng test unclip (bước 2), owner duyệt 2026-11-18
- Plan ghi IoU ≥ 0,95 cho box 20×200 và 30×600. Khi raster hóa kernel GT bằng `fillPoly` (giống PaddleOCR `MakeShrinkMap`), ngưỡng này không đạt được do sai số làm tròn đỉnh polygon.
- Hai quy ước đã thử:
  - **bw** (kích thước tính bằng số pixel): 20×200 → 0,948; 30×600 → 0,850.
  - **bw−1** (kích thước theo tọa độ đỉnh contour): 20×200 → 0,891; 30×600 → 1,000.
- Mô phỏng 3.000 box ngẫu nhiên (cao 12–40, rộng 80–600, seed 0):
  - bw−1: IoU trung bình 0,921, p5 0,79, min 0,64;
  - bw: trung bình 0,834–0,866;
  - không box nào có IoU < 0,5.
- Chốt: dùng **bw−1** và làm tròn đến số nguyên gần nhất. Test kiểm từng box có IoU ≥ 0,85 và IoU trung bình ≥ 0,85, đúng ngưỡng AC-7.
