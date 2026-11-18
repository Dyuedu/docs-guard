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

### D2 — Đường dẫn data trong `text_det.yaml`
Dữ liệu synth (intent 002) nằm ở `data/synth_ocr/receipts/{train,val}/index.jsonl`, với đường dẫn ảnh tương đối theo thư mục chứa file jsonl. Config cũ trỏ tới `receipts/train.jsonl`, file này không tồn tại. Đã sửa đường dẫn và đặt `image_root: null`, nghĩa là ảnh được tìm theo thư mục của từng file jsonl.

### D3 — `export_onnx` dùng exporter TorchScript (`dynamo=False`)
Torch 2.11 mặc định dùng exporter dynamo, exporter này cần thêm gói `onnxscript`. Exporter TorchScript đã ổn định với `dynamic_axes` và TensorRT. AC-2 (khớp pbtxt, batch động, parity < 1e-3) đạt với exporter này.

### D4 — MLflow file store
MLflow 3.18 từ chối backend thư mục `./mlruns`. Spec quy định dùng thư mục này để owner chép về từ máy thuê. `loop.py` đặt `MLFLOW_ALLOW_FILE_STORE=true` (cờ opt-out chính thức của MLflow), và chỉ đặt khi tracking URI là thư mục.

### D5 — RNG augment theo (seed, index), không dùng `set_epoch`
Với `persistent_workers`, gọi `set_epoch` ở main process không tới được worker. Thay vào đó, `EpochOffsetSampler` sinh index `epoch·N + i`, và dataset dùng `index % N` để chọn ảnh, dùng `(seed, index)` làm seed RNG. AC-9 vẫn đạt. Ở chế độ eval, dataset index chặt để vòng lặp dừng đúng ở `len()`; lỗi này được phát hiện khi smoke test bị treo và đã có test kèm theo.

### D6 — Thứ tự test-first
- Bước 2–4: viết test trước và đã thấy FAIL.
- Bước 5–9: viết test cùng lúc với code trong khi chờ tải torch (2,7 GB), nên không quan sát được lần chạy FAIL trước khi implement.
- Expected values vẫn lấy từ spec: D, IoU 5320/7080, P/R/H của ca (b), BCE OHEM tính tay, pbtxt.

### D7 — Môi trường
- Đã cài torch 2.11.0+cu128, torchvision 0.26, onnx 1.23, onnxruntime 1.32 (CPU), mlflow 3.18, pyclipper 1.4.
- Cài `pip install -e . --no-deps` để `scripts/*.py` import được `docguard`.

### D8 — Unclip từ `minAreaRect` (owner duyệt 2026-11-18; spec §Out of scope cần owner cập nhật)
- Lần overfit đầu (run `6f365670695d43c9be8bb799457554f5`) đạt H-mean 0,79. Số box dự đoán khớp GT, nhưng box cao khoảng 1,4 lần.
- Nguyên nhân: chữ synth nghiêng (trung vị 3–5°). Bbox thẳng của một kernel mảnh, dài và nghiêng cao hơn nhiều so với độ dày thật, nên d bị phóng to.
- Đưa shrink map GT vào postprocess cũ (200 ảnh synth val, 3.965 box): H-mean 0,80, IoU trung bình 0,68. Nghĩa là AC-10(a) không thể đạt dù model hoàn hảo.
- Sửa: lấy kích thước kernel từ `cv2.minAreaRect(contour)` (giống PaddleOCR `DBPostProcess`), unclip hình chữ nhật xoay, rồi trả bbox thẳng của nó. Hợp đồng serving (box thẳng) không đổi. Với kernel thẳng trục, cách này cho kết quả trùng D1 (rw = bw − 1).
- Sau khi sửa, cùng phép thử GT cho H-mean 0,9995, IoU trung bình 0,876; checkpoint overfit cũ đạt H-mean 0,99.
- Test mới `test_det_postprocess_skewed_line_recovers_gt_bbox` (30×300, nghiêng 3° và −5°, IoU ≥ 0,85): trước khi sửa FAIL (0,58 và 0,53), sau khi sửa PASS.

### Kết quả AC-8b — overfit trên RTX 3060 Laptop (2026-11-18)
- Thiết lập: 10 ảnh synth train đầu tiên, 1.000 iteration, batch 8, warmup 100, ImageNet pretrained, AMP, 2 worker. Tham số còn lại theo `text_det.yaml`. MLflow ghi vào thư mục scratch (không ghi `mlruns/` của repo).
- **MLflow run `ecc59bb76a4e4da4abdfbb4e20a00dd9`**, H-mean trên chính 10 ảnh đó qua `evaluate_det`:

  | Iteration | 250 | 500 | 750 | 1.000 |
  |---|---|---|---|---|
  | H-mean | 0,729 | 0,953 | 0,982 | **0,990** |

  Tại iteration 1.000: P = 0,985, R = 0,995, IoU trung bình 0,757. **Đạt** (≥ 0,90).
- Thông tin thêm:
  - 4,18 triệu tham số; `best.pt` 16,2 MB;
  - VRAM tối đa 2,7 GB;
  - khoảng 37 ảnh/s trên 3060 Laptop. Con số này chỉ mang tính tham khảo; tốc độ trên 3090 sẽ đo ở bước 12.
- Lần chạy trước khi có D8 (run `6f365670695d43c9be8bb799457554f5`, postprocess cũ): H-mean 0,79.
