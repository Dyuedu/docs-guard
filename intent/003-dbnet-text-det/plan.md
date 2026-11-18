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

### Đo throughput trước khi thuê 3090 (2026-11-18, máy local i5-10500H + RTX 3060 Laptop)
- **Dataloader, 1 worker** (150 ảnh synth train, cấu hình `text_det.yaml`): 52,6 ms/ảnh, tức khoảng 19 ảnh/s mỗi worker.
  - `make_border_map`: 43%;
  - crop, resize, rotate: 33%;
  - đọc ảnh: 21%;
  - `make_shrink_map`: 3%.
- **GPU thuần** (batch giả 8×640, AMP, forward + backward + Adam): 43 ảnh/s trên 3060 Laptop, VRAM 2,7 GB.
- Kết luận cho bước 12:
  - Lần overfit 37 ảnh/s bị nghẽn ở CPU (2 worker × 19 ảnh/s), chưa phải giới hạn GPU.
  - Ước tính 3090 nhanh hơn 3060 Laptop khoảng 2,5–3 lần, tức 110–130 ảnh/s. Muốn GPU không phải chờ thì cần khoảng 10 worker Xeon E5 v4, giả định mỗi worker đạt 11–13 ảnh/s (đây là ước tính chưa đo). 16 worker cho dư địa vừa phải, nên **chưa cần vector hóa `make_border_map`**.
  - Lần chạy thử 15 phút sẽ xác nhận. Nếu `imgs_per_s` dừng ở khoảng 16 × (tốc độ một worker) và GPU-util thấp, thì nghẽn ở CPU.
  - Ở 110 ảnh/s, 100k iteration (800k ảnh) mất khoảng 2 giờ, dưới ngưỡng 6 giờ.

### D9 — Sau verifier (2026-11-18)
- **Major:** `scripts/eval_det.py` crash khi log vào `./mlruns` mặc định, vì thiếu phần thiết lập file store của D4.
  - Commit test tái hiện trước (`a38b12d`, thấy FAIL), rồi mới sửa: tách `setup_mlflow()` thành hàm dùng chung cho train và eval.
- **Minor:**
  - AC-1 thêm ca (N = 2, S = 960);
  - thêm test cho `record_polys`: text rỗng hoặc `###` bị bỏ qua, MC-OCR dùng 4 góc bbox;
  - `NOTICE` liệt kê đủ các file MMOCR và DBNet.pytorch;
  - smoke test giới hạn `MLFLOW_ALLOW_FILE_STORE` trong phạm vi test.
- **Ngoài bảng "Files to change":** thay đổi ở `docs/plan.md` (train trên 3090) và comment trong `scripts/setup_cloud.sh` thuộc commit duyệt intent `8775e80` của owner, có từ trước khi implement.

### D10 — Train trên RTX 5060 Ti thuê thay cho 3090 (owner chọn 2026-11-18)
**Máy và cách chọn**
- 3090 hết máy. Owner chọn máy ezyclouds: RTX 5060 Ti 16 GB (Blackwell), i5-14600KF 18 nhân, 27 GB RAM, Ubuntu 24.04, 7.000 đ/giờ.
- Môi trường: torch 2.11.0+cu128 (Blackwell cần cu128 trở lên). Trên máy này: 134 test pass, 1 skip (test chỉ chạy trên Windows).
- Không chạy `setup_cloud.sh`, vì máy không có sudo và việc train không cần Docker.

**Chạy thử 3.000 iteration** (16 worker, run `8664395e3e434f2bbdb142e9e0036346`)
- Khoảng 73 ảnh/s, GPU-util 98%, CPU khoảng 17%: GPU là chỗ nghẽn, đạt ngưỡng 37 ảnh/s.
- H-mean synth val: 0,817 (iteration 1.000) → 0,915 (2.000) → 0,946 (3.000).
- Xem trước MC-OCR val với checkpoint chạy thử: recall 0,719. Theo trường: TOTAL_COST 0,861, ADDRESS 0,718, SELLER 0,697, TIMESTAMP 0,525.

**Hai bug phát hiện khi chạy trên Linux**, đều đã có test tái hiện commit trước và FAIL trước khi sửa:
1. `imgs_per_s` tính gộp thời gian eval vào cửa sổ đo, ra 14 thay vì 73 ảnh/s, có thể kích hoạt nhầm luật dừng "< 37 ảnh/s".
   - Test `65f7fc1`, sửa `f2d991c`: reset cửa sổ đo sau eval và lưu checkpoint.
2. **Deadlock của DataLoader dùng fork.**
   - Lần chạy đầy đủ đầu tiên (run `4d91dbdd81964c51a913b8c55c23fdee`, đã dừng tay) treo ngay từ đầu: mọi thread của worker kẹt ở `futex`, thread pin của tiến trình chính chờ socket, GPU 0%.
   - Nguyên nhân: fork từ một tiến trình đã có khoảng 42 thread (torch, OpenCV, MLflow). Lỗi ngẫu nhiên, nên lần chạy thử không gặp; trên Windows không gặp vì Windows luôn dùng spawn.
   - Test `3babc94`, sửa `d09e0d9`: worker khởi động bằng `spawn` trên mọi hệ điều hành; mỗi worker đặt `cv2.setNumThreads(0)` và `torch.set_num_threads(1)` (theo mmengine `set_multi_processing`).

**Lần train đầy đủ:** run `a27386451d79430d8b0594d914f4c4df` (`v1_100k_r2`), bắt đầu 22:59 ngày 2026-11-18.

## Kết quả AC-10 — train 100k iteration trên RTX 5060 Ti (2026-11-19)
**MLflow run `a27386451d79430d8b0594d914f4c4df`** (`v1_100k_r2`)
- Chạy đủ 100k iteration (early stop không kích hoạt), mất 3,21 giờ; khoảng 73 ảnh/s; VRAM tối đa 2,67 GB.
- Model 4,18 triệu tham số; `best.pt` 16,2 MB, là checkpoint ở **iteration 90.000**.
- H-mean synth val theo các lần eval: 0,968 (5k) → 0,991 (15k) → 0,998 (50k) → 0,9992 (90k) → 0,9988 (100k).

| Tiêu chí | Ngưỡng | Kết quả | |
|---|---|---|---|
| (a) H-mean synth val (1.000 ảnh, 19.814 box) | ≥ 0,90 | **0,9992** (P 0,9989, R 0,9994, IoU trung bình 0,816) | **Đạt** |
| (b) Recall MC-OCR val (668 box, 4 trường) | ≥ 0,80 | **0,720** (481/668; IoU trung bình 0,745) | **Không đạt** |

Recall MC-OCR val theo trường và theo nhóm chiều cao (normal: cao ≤ 1,6 lần trung vị chiều cao của trường đó; tall: cao hơn):

| Trường | Recall | n | normal | tall |
|---|---|---|---|---|
| TOTAL_COST | 0,846 | 208 | 0,870 (184) | 0,667 (24) |
| SELLER | 0,782 | 119 | 0,788 (104) | 0,733 (15) |
| ADDRESS | 0,767 | 202 | 0,786 (173) | 0,655 (29) |
| TIMESTAMP | **0,410** | 139 | 0,408 (120) | 0,421 (19) |

Precision MC-OCR (0,137) không có ý nghĩa, vì chỉ 4 trường được gán nhãn.

**Phân tích các box bị trượt** (cùng checkpoint):
- Chỉ 17/668 box thật sự không được phát hiện (IoU tốt nhất < 0,1).
- 170 box có được phát hiện nhưng IoU nằm trong khoảng 0,1–0,5. Box dự đoán rộng khoảng gấp 2,09 lần và cao khoảng gấp 1,72 lần box GT (trung vị).
- Kiểm tra trực quan cho thấy ba nguyên nhân:
  1. dòng in sát nhau bị gộp theo chiều dọc (ví dụ "Tổng số" + "Tổng cộng", "Ngày bán" + "Quầy");
  2. nhiều trường trên cùng một dòng bị gộp theo chiều ngang, trong khi MC-OCR chỉ gán nhãn một phần dòng (ví dụ "Số GD: … Ngày: …");
  3. ảnh hóa đơn bị xoay 90°, trong khi augment chỉ xoay ±5°.
- Lần chạy thử 3k iteration cho recall 0,719, gần như bằng 100k. Nghĩa là train thêm không thu hẹp được khoảng cách synth → real.

**Theo spec, concern 6:** kết quả đã ghi kèm run id; owner quyết định có chấp nhận v1 hay không. Không hạ ngưỡng khi owner chưa đồng ý.

**File:**
- `results/text_det/v1_{synth_val,mcocr_val}.json` (thư mục `results/` nằm trong gitignore);
- `best.pt`, `last.pt` và `mlruns/` ở thư mục scratch, owner tự chuyển vào `artifacts/` và `mlruns/`.

**Quyết định của owner (2026-11-19):** chấp nhận v1.
- v1 đạt có điều kiện: AC-10(a) đạt; AC-10(b) = 0,720, dưới ngưỡng, ghi nhận và **không hạ ngưỡng**.
- Checkpoint v1 dùng làm model sinh pseudo-label cho intent 1.5.
- Mở intent mới cho v2 để xử lý ba nguyên nhân trên:
  - synth in dòng sát hơn và nhiều trường trên cùng một dòng;
  - augment xoay 90°;
  - fine-tune bằng pseudo-label MC-OCR.
- Đã trả máy thuê sau khi chép kết quả về (checksum khớp).
