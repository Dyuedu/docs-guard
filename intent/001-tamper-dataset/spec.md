# Spec 001 — Sinh và kiểm định tập giả mạo

Status: Accepted
Date: 2026-11-11
Intent: ./intent.md

## Acceptance criteria

| ID | Tiêu chí | Proof |
|---|---|---|
| AC-1 | Không rò rỉ: tập `source` của train/val/test rời nhau từng đôi. Mọi `source` của split X đều nằm trong `splits.json[X]`. Donor của splice cũng thuộc cùng split. | `tests/test_tamper_stats.py::test_leak_check*` + report `checks.no_leak` |
| AC-2 | Mask hợp lệ: ảnh và mask cùng H×W, giá trị mask ⊆ {0,255}. Mọi ảnh `tampered=1` có mask khác rỗng. Mọi ảnh `authentic` có mask toàn 0. | test + report `checks.masks_valid` |
| AC-3 | Mỗi phép biến đổi (copy_move, splice, inpaint) chiếm ≥ 20% số ảnh giả trong mỗi split. Mục tiêu đều nhau là 33%. Biên 20% cho phép copy-move bị bỏ qua ở hóa đơn có ít hơn 2 box số. | report `checks.op_balance` |
| AC-4 | Diện tích mask: trung vị `mask_area_ratio` của ảnh giả nằm trong khoảng [0,0001; 0,01]. Ảnh có ratio > 0,05 bị gắn cờ `outliers` để xem tay. Cận dưới bằng 1/20 ngưỡng `decision.tampered_area_ratio = 0.002`, cận trên bằng 5 lần ngưỡng đó. | report `mask_area` + `outliers` |
| AC-5 | Tất định: chạy lại cùng seed, cùng input thì `index.jsonl` giống hệt. | `tests/test_tamper_stats.py::test_generation_deterministic` (fixture nhỏ) |
| AC-6 | Report chạy được bằng một lệnh, exit ≠ 0 nếu bất kỳ `checks.*` nào FAIL. | `scripts/tamper_report.py` trên dữ liệu thật |
| AC-7 | Owner xem bảng ảnh xem trước của 50 cặp (train/val, phân tầng theo phép biến đổi). Số cặp hỏng ≤ 5. "Hỏng" nghĩa là mask không trùng vùng bị sửa, hoặc vùng sửa nằm ngoài chữ. | ghi kết quả vào `plan.md` › Deviations |
| AC-8 | *(bổ sung 2026-11-11, owner duyệt fix bug)* Inpaint chỉ sửa pixel **bên trong** bbox của box được chọn. Box số nằm dọc (h > w) không được chọn để inpaint; nếu hóa đơn chỉ có box dọc thì ném `ValueError`. | `tests/test_synth_tamper.py::test_inpaint_*` |

## Interface / data contract

**`index.jsonl`** (mỗi split), giữ các trường cũ và thêm 2 trường:
```json
{"name": "..._splice_0", "source": "mcocr_public_x.jpg", "op": "splice", "tampered": 1, "jpeg_q": 83,
 "donor": "mcocr_public_y.jpg", "font": "BeVietnamPro-Regular.ttf"}
```
`donor` chỉ có ở splice, `font` chỉ có ở inpaint.

**CLI**
- `make_tamper_dataset.py --font <file|dir>`: nếu là thư mục thì mỗi lần inpaint chọn ngẫu nhiên một file `*.ttf`. Thêm `--tampered-per-image-train N` (mặc định = `--tampered-per-image`).
- `scripts/tamper_report.py --root data/tamper --splits data/processed/splits.json --out results/data/tamper [--preview 50] [--seed 0]`.

**`src/docguard/data/tamper_stats.py`**
```python
def load_index(split_dir: Path) -> list[dict]: ...
def leak_check(index_by_split: dict[str, list[dict]], splits: dict[str, set[str]]) -> list[str]: ...  # [] = OK
def mask_stats(split_dir: Path, rows: list[dict]) -> list[dict]: ...  # name, op, h, w, area_ratio, valid, error
def summarize(index_by_split, stats_by_split, cfg) -> dict: ...       # counts, op ratios, area percentiles, outliers, checks
def preview_grid(split_dir: Path, rows: list[dict], out_path: Path, crop_pad: int = 48) -> None: ...
```
Đầu ra report: `results/data/tamper/report.json`, `report.md`, và `preview/*.jpg`. Mỗi ô xem trước gồm 3 phần:
1. ảnh toàn cảnh có viền đỏ quanh bbox của mask;
2. ảnh cắt phóng to quanh mask;
3. mask phủ màu lên ảnh cắt.

## Technical approach
- Kiểm định theo kiểu "data validation", không cần thư viện mới. Tham khảo cách DocTamper và CASIA trình bày cặp ảnh/mask và thống kê `tampered area ratio` trong bài báo của họ.
- Xoay vòng font: lấy font đã kiểm tra glyph trong `data/fonts/`. Không dùng font của Windows, vì lý do giấy phép và để chạy được trên máy cloud.
- Chỉ dùng numpy, OpenCV và Pillow, đều đã có trong dependency.
- Đã loại: Great Expectations/pandera, vì nặng so với khoảng 10 bất biến.

## Non-functional
| Loại | Ngưỡng | Đo bằng |
|---|---|---|
| Thời gian sinh cả tập trên laptop | ≤ 30 phút | đo bằng `time` |
| Thời gian chạy report | ≤ 5 phút | đo bằng `time` |

## Flagged concerns
- Ảnh MC-OCR vốn đã là JPEG. Ảnh giả sẽ bị nén JPEG 2 lần, ảnh thật cũng bị nén lại 1 lần. Model vẫn có thể học dấu vết khác biệt giữa vùng sửa và nền, đúng như mục tiêu. Tuy vậy, dấu vết nén kép trên ảnh giả thật ngoài đời có thể khác.
- Mask được tính **trước** khi nén JPEG. Nhiễu JPEG lan ra ngoài mask vài pixel. Đây là quy ước chuẩn của các bộ dữ liệu như CASIA hay DocTamper, ghi lại ở đây để biết.

## Out of scope
- Pretrain bằng DocTamper.
- Thêm phép biến đổi mới, ví dụ sửa chữ bằng GAN hay diffusion.
- Train CAT-Net (Phase 2.3).
