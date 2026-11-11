# Plan 002 — Dữ liệu chữ tổng hợp cho OCR

Status: Approved
Date: 2026-11-11
Spec: ./spec.md

## Files to change
| File | Thay đổi |
|---|---|
| `tests/test_synth_text.py` (mới) | AC-1, AC-2, AC-5, AC-6 trên fixture nhỏ (corpus 50 dòng, 1 font) |
| `tests/test_synth_receipt.py` (mới) | AC-3, AC-6: 20 hóa đơn từ fixture, kiểm polygon |
| `tests/fonts/` (mới) | **vendor 1 font OFL** (BeVietnamPro-Regular + OFL.txt, ~130 KB) để test không phụ thuộc font hệ thống. Đồng thời giải quyết điểm minor #1 của intent 001 |
| `src/docguard/data/synth_text.py` (mới) | `ReceiptCorpus`, `LineRenderer`, lọc charset, sinh số tiền/ngày/SĐT |
| `src/docguard/data/synth_receipt.py` (mới) | layout hóa đơn + render + hiệu ứng + polygon |
| `scripts/make_synth_ocr.py` (mới) | 3 sub-command `corpus` / `receipts` / `lines`, multiprocessing |
| `scripts/synth_ocr_report.py` (mới) | checks + preview, exit ≠ 0 khi FAIL |
| `pyproject.toml` | extra `[synth]` (tùy nhánh dependency) |
| `data/fonts/` (qua script/curl, không commit) | thêm 2 font mono/in nhiệt OFL + license |
| `README.md` | mục Dữ liệu: thêm lệnh sinh synth OCR |

## Order of work
1. Chốt dependency (Flagged concern) dựa trên kết quả cài thử synthtiger. Ghi vào spec.
2. Vendor font vào `tests/fonts/`; viết `tests/test_synth_text.py` và `tests/test_synth_receipt.py` → FAIL.
3. `synth_text.py`: corpus + lọc charset + `LineRenderer`. Chạy test AC-1, AC-2, AC-5, AC-6 (lines).
4. `synth_receipt.py`: layout hóa đơn → render từng dòng → ghép lên giấy → hiệu ứng (perspective cập nhật polygon bằng cùng ma trận) → nền → hiệu ứng ảnh. Test AC-3, AC-6 (receipts).
5. `make_synth_ocr.py` + `synth_ocr_report.py`.
6. `pytest -q`.
7. Chạy thật: `corpus` → `receipts --n-train 20000 --n-val 1000` → `lines --n 5000` → report `--preview 30`. Ghi thời gian (AC-7) và kết quả checks vào plan.
8. Owner xem 30 preview (AC-4). Verifier. PR từ `feat/002-synth-ocr-data`.

## Proof
- Test: `pytest -q`, các test mới AC-1/2/3/5/6.
- Report dữ liệu thật: `checks.charset`, `no_leak`, `polygons_valid` PASS; phân bố chiều cao chữ (AC-4); thời gian (AC-7).
- Owner review preview (AC-4).
- Không đụng model → không có eval/MLflow. Ghi hash `splits.json` và SHA corpus vào report.

## Risks
- synthtiger không cài được trên 3.12 → nhánh (b) tốn thêm khoảng 1–2 ngày để tự làm hiệu ứng.
- Domain gap font hiện đại vs in nhiệt: chấp nhận, bù bằng fine-tune trên MC-OCR.
- Corpus tên món tự soạn có thể nghèo → CRNN học lệch về số. Giảm bằng cách trộn 30 % dòng từ `words.txt` ghép ngẫu nhiên.
- 20k ảnh JPEG ~5 GB: kiểm tra đĩa trước khi chạy.

## Deviations (điền khi làm xong)
