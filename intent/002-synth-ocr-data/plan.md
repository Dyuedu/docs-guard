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
- **Dependency:** không thêm extra `[synth]`. synthtiger bị loại vì imgaug vỡ với numpy 2, nên `pyproject.toml` không đổi.
- **Font:** Courier Prime và Share Tech Mono thiếu 92–102 glyph tiếng Việt nên bị loại. Thay bằng IBM Plex Mono (Regular/Bold), Noto Sans Mono, Source Code Pro (OFL), đều phủ đủ 221 ký tự trong vocab.
- **Tốc độ:** nhiễu nền ban đầu sinh bằng vòng lặp Python (0,38 s/hóa đơn, 4,9 ms/dòng). Đã chuyển sang `np.random.default_rng(seed từ rng)`, vẫn tất định, còn 0,18 s/hóa đơn và 1,6 ms/dòng.
- **AC-2 sửa định nghĩa (owner duyệt):** đổi từ "corpus không trùng val/test" sang "corpus chỉ lấy từ train". Lý do: 63,7% text val/test của MC-OCR trùng nguyên văn với train, vì cùng chuỗi cửa hàng nằm ở cả hai split. `check_leak` được thay bằng `check_provenance`, có thêm 3 test. Tỉ lệ trùng được ghi vào report dưới dạng thông tin.
- **Report:** nhãn được đếm bằng `Counter` thay cho `np.unique`, để in ra tên nhãn thay vì `np.str_`.

## Kết quả chạy thật (2026-11-11, splits.json sha256 `7d19106a…`, corpus sha256 `25fb1076…`)
- **Corpus:** 5.286 dòng, 2.606 từ, lấy từ 924 hóa đơn train.
- **Hóa đơn:** 20.000 train + 1.000 val, mất 24,6 phút với 8 worker (AC-7 ≤ 60 phút: PASS). Tổng 1,6 GB cho cả `data/synth_ocr`.
  - Số dòng/ảnh (min/trung vị/max): 9/20/30.
  - Chiều cao chữ quy về cạnh dài 1024px (p5/p50/p95): 11,9/20,6/32,9 px (AC-4 cần p50 ∈ [12, 40]: PASS).
  - Nhãn của train: OTHER 296k, ADDRESS 30k, TOTAL_COST 30k, SELLER 20k, TIMESTAMP 20k.
- **Dòng chữ val:** 5.000 dòng, 10 giây. Chiều rộng p50 là 281px, độ dài nhãn p50 là 16 ký tự, 100% nhãn khác rỗng. LineRenderer chạy 1,54 ms/dòng (AC-7 ≤ 5 ms: PASS).
- **Report:** 6/6 check PASS (`charset`, `no_leak`, `polygons_valid`, `text_height`, `lines_valid`, `line_render_speed`), exit 0.
- **AC-4 (owner xem 30 preview trong `results/data/synth_ocr/preview/`):** PASS, owner duyệt 2026-11-11 (≤ 3/30 ảnh hỏng).
