# Spec 003 — DBNet text detection v1

Status: Accepted
Date: 2026-11-12
Intent: ./intent.md

## Acceptance criteria

Ký hiệu: r = `shrink_ratio` = 0,4. Với polygon diện tích A, chu vi L, khoảng shrink là D = A·(1 − r²)/L (DB paper, PaddleOCR `MakeShrinkMap`).

| ID | Tiêu chí | Proof |
|---|---|---|
| AC-1 | **Shape model.** Với input `[N,3,S,S]`, S ∈ {640, 960}, N ∈ {1, 2}: ở `train()` model trả dict `{"prob_logits","thresh","binary"}`, mỗi phần `[N,1,S,S]`. Ở `eval()` model trả **một** tensor `prob` `[N,1,S,S]`, mọi giá trị ∈ [0, 1]. | `tests/test_dbnet.py::test_forward_shapes_train_eval` |
| AC-2 | **Export khớp serving.** `export_onnx(model.eval(), …, input_name="image", output_name="prob", opset=17)` tạo được file ONNX. Tên, dtype và dims I/O khớp `serving/model_repository/text_det/config.pbtxt` (test đọc file pbtxt). Batch động: chạy được N = 1 và N = 2. Parity: ONNX Runtime CPU vs PyTorch, max abs diff < 1e-3 (plan.md §Verification). | `tests/test_dbnet.py::test_export_onnx_matches_pbtxt`, `test_onnx_parity` |
| AC-3 | **Shrink map.** Với hình chữ nhật 200×20 px, D = 4000·0,84/440 ≈ 7,64 px. `shrink_map` = 1 đúng trên hình chữ nhật thu vào D mỗi phía (sai số ±1 px ở biên), bằng 0 ở chỗ khác. Polygon có cạnh ngắn của bbox < `min_text_size` = 8 px, hoặc text rỗng, thì `shrink_mask` = 0 trên vùng polygon (bị bỏ qua khi tính loss). | `tests/test_det_labels.py::test_shrink_map_rect`, `test_small_polygon_ignored` |
| AC-4 | **Threshold map.** Mọi giá trị ∈ [0,3; 0,7]. Pixel nằm trên cạnh polygon có giá trị 0,7 (±0,02). Pixel cách cạnh ≥ D có giá trị 0,3. `thresh_mask` = 1 trên polygon nới ra D và = 0 ngoài vùng đó. Công thức: `0,3 + 0,4·(1 − clip(dist/D, 0, 1))`, theo PaddleOCR `MakeBorderMap`. | `tests/test_det_labels.py::test_threshold_map_values` |
| AC-5 | **Loss.** `L = 5·BCE_ohem(prob_logits) + 10·L1(thresh) + 1·Dice(binary)`. Ở đây `binary = sigmoid(k·(sigmoid(prob_logits) − thresh))`, k = 50. BCE_ohem lấy **mọi** pixel dương cộng với k_neg = min(#âm, 3·#dương) pixel âm có loss lớn nhất; số pixel được đếm trên cả batch; nếu cả batch không có pixel dương thì `loss_prob` = 0 (giống mmocr mặc định, `fallback_negative_num = 0`). L1 và Dice chỉ tính trong mask tương ứng. Kiểm tra: (a) dự đoán hoàn hảo (logit ±20, thresh = label) cho L < 1e-3; (b) với 1 pixel dương và 10 pixel âm, đúng 3 pixel âm có loss lớn nhất được tính; (c) loss hữu hạn với cả fp16 autocast. | `tests/test_det_loss.py` |
| AC-6 | **Metric detection.** `match_boxes(preds, gts, iou_thr=0.5)` ghép 1–1 tham lam theo IoU giảm dần, ghép khi IoU ≥ 0,5. IoU tính trên box thẳng `[x1,y1,x2,y2]`. Tổng hợp micro trên cả tập: P = Σmatched/Σpred, R = Σmatched/Σgt, H = 2PR/(P+R). Mẫu số bằng 0 thì metric tương ứng = 0. Ca kiểm tra: (a) GT `[100,50,300,80]`, pred `[110,52,310,84]` → IoU = 5320/7080 ≈ 0,7514; (b) 20 GT, 22 pred, 19 ghép → P = 0,8636, R = 0,95, H = 0,9048; (c) một pred phủ hai GT chỉ ghép được 1; (d) có thể lọc GT theo `labels` để chỉ tính recall các trường cần (MC-OCR). | `tests/test_detection_metric.py` |
| AC-7 | **Đánh giá end-to-end đúng đường serving.** `evaluate_det` đi qua `P.det_preprocess`, model, rồi `P.det_postprocess`, với tham số đọc từ `configs/pipeline.yaml`. Dùng một "model giả" xuất **đúng shrink map GT** (lấy từ AC-3, tính trên ảnh letterbox) cho 2 ảnh fixture có dòng chữ 20×200 và 30×600 px (tọa độ ở 960). Kết quả phải là recall = 1,0 và P = 1,0 ở IoU 0,5, và IoU trung bình ≥ 0,85. Ca này **chỉ đạt được sau khi giải quyết Flagged concern 1**. | `tests/test_eval_det.py::test_gt_kernel_roundtrip` |
| AC-8 | **Vòng train chạy được.** (a) Smoke test CPU: 2 iteration, ảnh 64×64, `pretrained=False`, loss hữu hạn, ghi được checkpoint và log MLflow vào thư mục tạm. (b) Overfit trên RTX 3060: 10 ảnh synth train, ≤ 1.000 iteration, H-mean trên chính 10 ảnh đó ≥ 0,90 (plan.md §Verification). Kết quả (b) ghi vào plan.md, kèm MLflow run id. | `tests/test_train_det.py::test_smoke_two_iters`; (b) chạy tay |
| AC-9 | **Dataset tất định.** Cùng seed và cùng index → cùng ảnh crop và cùng 4 map label (so từng phần tử). | `tests/test_det_labels.py::test_dataset_deterministic` |
| AC-10 | **Nghiệm thu v1** (intent, owner chốt 2026-11-12). Checkpoint tốt nhất của lần train 100k iteration trên RTX 3090 phải đạt: (a) H-mean ≥ 0,90 trên synth val (1.000 ảnh); (b) recall ≥ 0,80 trên 668 box GT 4 trường của MC-OCR **val** (SELLER 119, ADDRESS 202, TIMESTAMP 139, TOTAL_COST 208). Cả hai qua `evaluate_det`, IoU ≥ 0,5. Có MLflow run id. Report kèm recall theo từng trường và theo nhóm chiều cao box (xem Flagged concern 5). | `scripts/eval_det.py`, ghi vào plan.md |

## Interface / data contract

```python
# src/docguard/models/dbnet.py
class DBNet(nn.Module):
    def __init__(self, pretrained: bool = True, inner_channels: int = 256, k: float = 50.0): ...
    def forward(self, x: Tensor) -> dict[str, Tensor] | Tensor
        # train(): {"prob_logits": [N,1,H,W], "thresh": [N,1,H,W], "binary": [N,1,H,W]}
        # eval():  prob = sigmoid(prob_logits), [N,1,H,W]; nhánh thresh không chạy (đây là graph export)
        # H, W chia hết cho 32

# src/docguard/train/det_labels.py
def make_shrink_map(polys: list[np.ndarray], ignore: list[bool], h: int, w: int,
                    shrink_ratio: float = 0.4, min_text_size: int = 8) -> tuple[np.ndarray, np.ndarray]
    # -> (shrink_map, shrink_mask), float32 HxW
def make_border_map(polys: list[np.ndarray], ignore: list[bool], h: int, w: int,
                    shrink_ratio: float = 0.4, thresh_min: float = 0.3, thresh_max: float = 0.7
                    ) -> tuple[np.ndarray, np.ndarray]
    # -> (thresh_map, thresh_mask)

# src/docguard/train/det_dataset.py
class DetDataset(torch.utils.data.Dataset):
    def __init__(self, jsonl_paths: list[str], image_root: str, train: bool, cfg: dict, seed: int): ...
    # train=True: ảnh 3×640×640 + 4 map label; train=False: ảnh gốc RGB + GT boxes (cho evaluate_det)
    # đọc polygon nếu có, không thì lấy 4 góc của bbox (định dạng MC-OCR)

# src/docguard/train/det_loss.py
class DBLoss(nn.Module):
    def __init__(self, alpha: float = 5.0, beta: float = 10.0, ohem_ratio: float = 3.0, eps: float = 1e-6): ...
    def forward(self, out: dict[str, Tensor], batch: dict[str, Tensor]) -> dict[str, Tensor]
    # trả {"loss","loss_prob","loss_thresh","loss_binary"}

# src/docguard/metrics/detection.py
def box_iou(a: list[int], b: list[int]) -> float
def match_boxes(preds: list[list[int]], gts: list[list[int]], iou_thr: float = 0.5) -> list[tuple[int, int, float]]
def detection_prf(per_image: list[tuple[list[list[int]], list[list[int]]]], iou_thr: float = 0.5) -> dict[str, float]
    # -> {"precision","recall","hmean","n_pred","n_gt","n_match"}

# src/docguard/train/eval_det.py
def evaluate_det(predict_prob: Callable[[np.ndarray], np.ndarray], samples, pipeline_cfg: dict,
                 labels: set[str] | None = None) -> dict
    # predict_prob nhận (1,3,960,960) float32, trả (960,960); labels=None nghĩa là mọi box GT
```

**CLI**
- `scripts/train.py --config configs/train/text_det.yaml [--set train.iters=200 ...]`. Override dùng `docguard.config.parse_overrides` có sẵn.
- `scripts/eval_det.py --ckpt <best.pt> --data synth_val|mcocr_val --out results/text_det/<name>.json [--mlflow-run-id <id>]`. `mcocr_val` đọc `annotations.jsonl` lọc theo `splits.json["val"]`, chỉ tính recall với 4 nhãn.

**Config** — `configs/train/text_det.yaml` (thay đổi so với hiện tại):
```yaml
model: dbnet
pretrained: true                 # torchvision MobileNet_V3_Large_Weights.IMAGENET1K_V2
data:
  ...                            # giữ train/val/image_root/input_size/shrink_ratio/thresh_*
  min_text_size: 8
  crop_size: 640
  scale_range: [0.7, 1.3]        # cạnh dài = 960 · s trước khi crop
  rotate_deg: 5
train:
  iters: 100000                  # thay cho epochs: 300 (intent, 2026-11-12)
  batch_size: 8
  lr: 0.001                      # Adam, poly decay power 0.9, warmup 1000 iter
  weight_decay: 0.0001
  amp: true
  num_workers: 16
  eval_every: 5000
  early_stop_patience: 5         # số lần eval liên tiếp không tăng H-mean
  ohem_ratio: 3.0
  loss_weights: {prob: 5.0, binary: 1.0, thresh: 10.0}   # prob 1.0 → 5.0 theo DB paper code / mmocr / PaddleOCR
  init: null
```

**Checkpoint** — `{out_dir}/{run_name}/best.pt` và `last.pt`, mỗi file là dict `{"model": state_dict, "config": cfg, "iter": int, "hmean": float, "mlflow_run_id": str}`.

**Dependency** — `pyclipper>=1.3` (MIT) thêm vào extra `train`.

## Technical approach
- **Backbone:** torchvision `mobilenet_v3_large(weights=IMAGENET1K_V2)`, BSD-3. Lấy feature ở cuối `features[3]` (stride 4, 24 kênh), `[6]` (s8, 40), `[12]` (s16, 112) và `[16]` (s32, 960). Implementer kiểm lại index bằng test shape. Chuẩn hóa ảnh RGB theo ImageNet mean/std, giống `P.to_chw`. PaddleOCR đọc BGR rồi dùng mean RGB; ở đây không làm theo, mà giữ RGB ở mọi chỗ.
- **Neck và head:** port **PaddleOCR `DBFPN`** (`ppocr/modeling/necks/db_fpn.py`) và **`DBHead`** (`heads/det_db_head.py`), Apache-2.0.
  - Neck: conv 1×1 về 256 kênh, cộng top-down bằng upsample nearest ×2, conv 3×3 về 64 kênh, upsample về stride 4, rồi concat thành 256 kênh.
  - Head: conv 3×3 → BN → ReLU → ConvTranspose 2×2 s2 → BN → ReLU → ConvTranspose 2×2 s2, ra full resolution.
  - Head prob trả **logits**, sigmoid chỉ áp ở `eval()`. Thay đổi so với Paddle: Paddle áp sigmoid trong head; ở đây tách ra để BCE an toàn khi chạy AMP.
  - Mọi op (Conv, BN, Hardswish, ConvTranspose, Resize nearest với scale cố định, Concat) đều có trong ONNX opset 17 và TensorRT.
- **Label:** port **PaddleOCR `MakeShrinkMap`, `MakeBorderMap` và `EastRandomCropData`** (`ppocr/data/imaug/`), Apache-2.0, chỉ dùng numpy, cv2 và pyclipper. Giữ nguyên thuật toán, chỉ đổi interface.
- **Augment train:** resize cạnh dài về 960·s với s ~ U(0,7; 1,3), xoay U(−5°, 5°), rồi crop 640 kiểu EAST (tránh cắt ngang chữ, giữ tỉ lệ, pad). **Không lật ngang:** Paddle có lật, nhưng chữ hóa đơn bị lật không bao giờ xuất hiện ở serving. Không thêm hiệu ứng màu hay blur, vì dữ liệu synth đã có (intent 002).
- **Loss:** theo **mmocr `DBModuleLoss`** (Apache-2.0) và code của bài báo DB: BCE-with-logits cân bằng OHEM 3:1, α = 5, β = 10, Dice trên binary map. Không theo bản PaddleOCR mặc định: ở đó `BalanceLoss(main_loss_type="DiceLoss")` thực chất là Dice, không phải BCE như bài báo.
- **Train:** Adam lr 1e-3, poly 0,9, warmup 1k, AMP fp16 + GradScaler, batch 8, 100k iteration. Cứ 5k iteration chạy `evaluate_det` trên synth val (letterbox 960). Lưu `best.pt` theo H-mean, early stop sau 5 lần eval không tăng.
- **MLflow:** experiment `text_det`. Log params (toàn bộ config), metrics (các thành phần loss, H-mean val) và artifact `best.pt`. Tracking URI đọc từ biến môi trường, mặc định `./mlruns`. Owner chạy train trên máy thuê rồi chép `mlruns/` về.
- **Tham khảo (đã đọc, ghi attribution):**
  - PaddleOCR `configs/det/det_mv3_db.yml`: shrink 0,4, thresh 0,3–0,7, k = 50, Adam 1e-3, crop 640;
  - mmocr (loss);
  - WenmuZhou/DBNet.pytorch (đối chiếu vòng train);
  - MhLiao/DB không có license, chỉ đọc để đối chiếu thuật toán.
  - Tạo file `NOTICE` ở gốc repo, ghi PaddleOCR và mmocr (Apache-2.0); comment đầu mỗi file port ghi nguồn và đường dẫn file gốc.
- **Phương án đã loại:**
  - dùng thẳng mmocr: kéo theo mmcv/mmengine, nặng, khó cài trên Windows;
  - dùng weight PP-OCR: trái quyết định "tự train" (plan.md);
  - backbone MobileNetV3 ×0,5 như Paddle: torchvision không có weight ×0,5 pretrained.

## Non-functional
| Loại | Ngưỡng | Đo bằng |
|---|---|---|
| Tốc độ train trên 3090 | Chạy thử 15 phút trước; nếu ước tính 100k iteration > 6 giờ (tức < 37 ảnh/s) thì **dừng, báo owner** trước khi chạy tiếp | log ảnh/s trong MLflow |
| VRAM train | ≤ 20 GB ở batch 8, crop 640, AMP | `torch.cuda.max_memory_allocated` |
| Thời gian `pytest -q` thêm vào | ≤ 60 s trên CPU | CI |
| Kích thước model | ghi lại số tham số và MB của file `.pt` (thông tin, không phải ngưỡng) | MLflow |

## Flagged concerns
1. **[Đã chốt 2026-11-12: phương án A] `det_postprocess` unclip không bù đủ phần shrink, nên recall bị kéo xuống dù model đoán đúng hoàn hảo.**
   - Mô phỏng: đặt **đúng** shrink map GT vào `P.det_postprocess` hiện tại (unclip_ratio 1,5), rồi so box ra với GT.

     | Dòng chữ (h×w px, ở 960) | 12×80 | 20×100 | 20×200 | 20×400 | 30×300 | 30×600 |
     |---|---|---|---|---|---|---|
     | IoU, unclip 1,5 | 0,81 | 0,77 | **0,47** | **0,49** | 0,64 | 0,52 |
     | IoU, unclip 2,5 | 0,84 | 0,89 | 0,68 | 0,69 | 0,93 | 0,72 |
   - Nguyên nhân: với dòng dài, D ≈ 0,42·h nên kernel chỉ còn khoảng 16% chiều cao. Công thức unclip `d = A'·ratio/L'` chỉ trả lại khoảng 40% chiều cao. Dòng hóa đơn điển hình (20×200) rơi xuống IoU < 0,5, tức bị tính là trượt.
   - Box quá thấp còn **cắt mất dấu tiếng Việt** khi crop cho CRNN.
   - PaddleOCR có cùng công thức, nhưng nó unclip polygon và người dùng thường tăng `unclip_ratio` lên 2,0 khi thấy box chật.
   - **(A) Đề xuất:** sửa `P.det_postprocess` để unclip **chính xác** bằng nghịch đảo công thức shrink cho box thẳng. Với kernel w′×h′, giải (8 − 4k)·d² + 2(1 − k)(w′ + h′)·d − k·w′h′ = 0, trong đó k = 1 − r². Box ra là w′ + 2d, h′ + 2d.
     - Đã thử: 20×200 cho ra 200,9×20,9; 30×600 cho ra 600×30.
     - `configs/pipeline.yaml`: thay `unclip_ratio: 1.5` bằng `shrink_ratio: 0.4` (dùng chung giá trị với `text_det.yaml`), và sửa `serving/.../pipeline/1/model.py` cho khớp.
     - Cũng sửa ngưỡng lọc kernel từ `bw < 3 or bh < 3` thành `< 2`: dòng chữ cao 11–12 px có kernel chỉ 2–3 px, nên hiện đang bị lọc mất.
     - Việc này **trái với intent** (intent ghi `processing.py` chỉ đọc, không sửa), nên cần owner duyệt.
     - Không vi phạm luật "eval vs B0", vì B0 chưa đo; B0 sẽ đo **với** postprocess mới.
     - Test hiện có `test_det_postprocess_maps_back_to_image_coords` vẫn đúng.
   - **(B)** Giữ code, quét `unclip_ratio` ∈ {1,5; 2,0; 2,5; 3,0} trên synth val. Rẻ hơn, nhưng mô phỏng cho thấy dòng dài vẫn chỉ đạt IoU khoảng 0,7. AC-7 phải hạ yêu cầu IoU trung bình.
2. **[Đã chốt 2026-11-12: thêm torch CPU vào CI] CI chưa có torch.** CI chỉ cài `.[dev]`, nên các test AC-1/2/5/8 sẽ bị skip trên CI. Đề xuất thêm vào `ci.yml` bước cài `torch` và `torchvision` bản CPU (index `download.pytorch.org/whl/cpu`), `onnx`, `onnxruntime` (CPU) và `pyclipper`. CI chậm thêm khoảng 1–2 phút. Test model dùng `pytest.importorskip("torch")` để vẫn chạy được ở môi trường thiếu torch.
3. **`.venv` local chưa có torch.** Cần cài bản CUDA cho RTX 3060 (`pip install -e ".[train,export]"`, torch cu12x).
4. **`artifacts/` và `mlruns/` được bảo vệ** (CLAUDE.md, hook chặn). Claude không ghi vào đó ở local. Test dùng `tmp_path`. Train thật do owner chạy trên máy thuê theo lệnh trong plan.md, rồi owner chép `best.pt` và `mlruns/` về.
5. **Một phần box GT của MC-OCR bao hai dòng** (chiều cao p95 là 56–90 px ở cạnh dài 1024, trong khi trung vị khoảng 25 px). DBNet tách theo từng dòng nên sẽ trượt các box này, làm recall AC-10(b) thấp hơn thực tế. Report sẽ chia recall theo nhóm chiều cao (≤ 1,6 lần trung vị của trường đó và > 1,6 lần) để thấy rõ. Ngưỡng 0,80 vẫn tính trên toàn bộ 668 box như owner đã chốt.
6. **Khoảng cách synth → real.** v1 chỉ train trên dữ liệu tổng hợp. Nếu AC-10(b) không đạt 0,80, kết quả vẫn được ghi lại kèm run id, và owner quyết định: chấp nhận v1 làm mô hình sinh pseudo-label (1.5), hay sửa dữ liệu hoặc augment. Không hạ ngưỡng nếu owner chưa đồng ý.
7. **Train và benchmark khác GPU.** Train trên RTX 3090, benchmark B0 trên T4. Checkpoint không phụ thuộc GPU; latency và VRAM đo ở giai đoạn 3 trên T4.

## Out of scope
- Pseudo-label MC-OCR (1.5) và fine-tune DBNet v2.
- Build TensorRT, FP16/INT8, pruning (giai đoạn 4). Ở đây chỉ có test export ONNX và parity.
- Đo latency và VRAM inference (giai đoạn 3, trên T4).
- Postprocess dạng polygon, box xoay (`minAreaRect`).
- Sửa `serving/model_repository/text_det/config.pbtxt`.
