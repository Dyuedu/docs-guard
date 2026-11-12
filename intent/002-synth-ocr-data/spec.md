# Spec 002 — Dữ liệu chữ tổng hợp cho OCR

Status: Implemented
Date: 2026-11-11
Intent: ./intent.md

## Acceptance criteria

| ID | Tiêu chí | Proof |
|---|---|---|
| AC-1 | Charset: mọi ký tự trong nhãn của hóa đơn tổng hợp và dòng chữ tổng hợp đều ∈ `configs/vocab_vi.txt` ∪ {" "}. Ký tự ngoài vocab bị loại ngay lúc sinh corpus, không phải lúc train. | `tests/test_synth_text.py::test_corpus_charset_subset_of_vocab` + report `checks.charset` |
| AC-2 | *(sửa 2026-11-11, owner duyệt)* Corpus không rò rỉ **theo nguồn gốc**: mọi dòng trong `receipt_lines.txt` đều có trong text của box thuộc split **train** (sau chuẩn hóa khoảng trắng và chữ hoa/thường). Dòng sinh theo mẫu không được tái tạo text chỉ có trong val/test (dòng > 3 ký tự). Dòng xuất hiện ở **cả** train lẫn val/test là hợp lệ. *(bổ sung 2026-11-12, owner chọn phương án a)* Vế 2 được kiểm trên dữ liệu thật: `make_synth_ocr.py corpus` ghi `heldout_blocklist.txt` (text chỉ có ở val/test, > 3 ký tự), generator sinh lại dòng/hóa đơn trùng blocklist, report có check `no_heldout_generated` quét toàn bộ text hóa đơn + dòng chữ đã sinh. Report ghi thêm tỉ lệ text val/test trùng với train (thông tin, không phải check). *Lý do sửa:* bản cũ đòi corpus không trùng val/test, nhưng 63% text val/test của MC-OCR trùng nguyên văn với train (cùng chuỗi cửa hàng ở hai split), nên check FAIL với 256 chuỗi đều lấy hợp lệ từ train. | `tests/test_synth_text.py::test_corpus_no_val_test_leak`, `test_provenance_check_*`, `test_heldout_only_*`, `test_check_heldout_generated_*`, `test_corpus_with_blocklist_*` + report `checks.no_leak`, `checks.no_heldout_generated` |
| AC-3 | Hóa đơn tổng hợp: mỗi ảnh có ≥ 8 dòng chữ; mỗi dòng có `polygon` 4 điểm nằm trong ảnh, `text` khác rỗng, và polygon của hai dòng khác nhau không chồng nhau quá IoU 0,1. | test trên 20 ảnh sinh từ fixture + report `checks.polygons_valid` |
| AC-4 | Hóa đơn tổng hợp trông giống hóa đơn: chiều cao chữ trung vị trong [12, 40] px ở ảnh cạnh dài 1024, có ít nhất 3 kiểu nội dung (header, dòng món, tổng/ngày). Owner xem 30 ảnh preview, ≤ 3 ảnh hỏng. "Hỏng" = chữ không đọc được, polygon lệch chữ, hoặc layout không giống hóa đơn. | report `text_height` + owner review ghi vào plan.md |
| AC-5 | Dòng chữ tổng hợp: ảnh cao đúng 32 px, rộng ≤ 320 px: co về cao 32 giữ tỉ lệ; nếu khi đó rộng > 320 thì ép ngang về 320 (như PaddleOCR `resize_norm_img`); dòng ngắn giữ bề rộng tự nhiên, pad phải làm lúc ghép batch *(sửa câu chữ 2026-11-12 cho khớp code, theo verifier)*, nhãn ≤ 40 ký tự, ≥ 95 % nhãn có ít nhất 1 ký tự khác khoảng trắng. | `tests/test_synth_text.py::test_line_image_shape_and_label` |
| AC-6 | Tất định: cùng seed → cùng `index.jsonl` (hóa đơn) và cùng ảnh dòng chữ (so pixel). | `test_*_deterministic` |
| AC-7 | Hiệu năng: 20k hóa đơn sinh xong ≤ 60 phút trên laptop với 8 worker; 1 dòng chữ ≤ 5 ms trung bình (đủ cho on-the-fly). | đo bằng `time`, ghi vào plan.md |
| AC-8 | Report một lệnh, exit ≠ 0 nếu check FAIL (giống intent 001). | `scripts/synth_ocr_report.py` |

## Interface / data contract

**Corpus** — `data/synth_ocr/corpus/{receipt_lines.txt, words.txt}`: mỗi dòng một chuỗi, UTF-8, đã lọc charset. Nguồn: text của box MC-OCR **split train** + sinh theo mẫu (`synth_text.ReceiptCorpus`).

**Hóa đơn tổng hợp** — `data/synth_ocr/receipts/{train,val}/images/{id}.jpg` và `{train,val}/index.jsonl`, mỗi dòng:
```json
{"image": "images/000123.jpg", "height": 1024, "width": 768,
 "boxes": [{"bbox": [x1,y1,x2,y2], "polygon": [[x,y],[x,y],[x,y],[x,y]], "text": "Tổng cộng: 125.000", "label": "OTHER"}]}
```
Tương thích `docguard.data.annotations.Sample` (`polygon` là trường thêm; `bbox` là bao của polygon). `label` dùng tập nhãn MC-OCR khi dòng là seller/address/timestamp/total, còn lại `OTHER`. Nhờ vậy dữ liệu tổng hợp cũng dùng được để test rule KIE.

**Dòng chữ** — `data/synth_ocr/lines/val/{images/*.jpg, gt.txt}` (tab-separated `path\tlabel`). Train sinh on-the-fly.

**Python**
```python
# src/docguard/data/synth_text.py
class ReceiptCorpus:
    def __init__(self, train_texts: list[str], vocab: set[str], seed: int): ...
    def line(self, rng: random.Random) -> str          # một dòng kiểu hóa đơn, charset ⊆ vocab
    def receipt(self, rng: random.Random) -> list[tuple[str, str]]   # [(text, label)] theo thứ tự trên hóa đơn

class LineRenderer:
    def __init__(self, fonts: list[str], height: int = 32, max_width: int = 320, seed: int = 0): ...
    def render(self, text: str, rng: random.Random) -> np.ndarray   # uint8 HxWx3, H=32, W<=320

# src/docguard/data/synth_receipt.py
def render_receipt(lines: list[tuple[str, str]], fonts: list[str], rng, cfg: dict) -> tuple[np.ndarray, list[dict]]
    # trả ảnh RGB và list box {"polygon","bbox","text","label"} sau khi áp hiệu ứng hình học
```

**CLI**
- `scripts/make_synth_ocr.py corpus --ann ... --splits ... --vocab ... --out data/synth_ocr/corpus`
- `scripts/make_synth_ocr.py receipts --corpus ... --fonts data/fonts --out data/synth_ocr/receipts --n-train 20000 --n-val 1000 --workers 8 --seed 42`
- `scripts/make_synth_ocr.py lines --corpus ... --fonts data/fonts --out data/synth_ocr/lines/val --n 5000 --seed 42`
- `scripts/synth_ocr_report.py --root data/synth_ocr --splits data/processed/splits.json --ann data/processed/annotations.jsonl --out results/data/synth_ocr --preview 30`

## Technical approach
- **Tham khảo chính: SynthDoG** (clovaai/donut, MIT). Lấy: quy trình nền → giấy → nội dung → hiệu ứng document (elastic, nhiễu, perspective) → hiệu ứng ảnh (bóng, tương phản, blur, màu), và bộ tham số trong `config_en.yaml`. Sửa: thay `GridStack` bằng **layout hóa đơn** (header / bảng món / tổng / footer, căn trái-phải, cột giá căn phải), và **xuất polygon từng dòng** (SynthDoG chỉ xuất chuỗi).
- **Dòng chữ: synthtiger `examples/synthtiger`** (MIT). Lấy cấu hình màu/texture/transform/blur; giới hạn transform nhẹ (xoay ≤ 3°, skew nhỏ) vì crop hóa đơn đã được `crop_for_rec` nắn thẳng.
- **Dependency:** quyết định ở Flagged concerns. Hai nhánh:
  - (a) `synthtiger` trong extra `[synth]`, viết template kế thừa `synthtiger.templates.Template`, lấy `layer.quad` của từng text layer sau hiệu ứng document;
  - (b) tự viết bằng Pillow + OpenCV (`cv2.getPerspectiveTransform` cho polygon, `cv2.remap` cho elastic), chỉ dùng dependency đã có.
- **Corpus kiểu hóa đơn**: sinh theo mẫu (tên cửa hàng từ MC-OCR train; địa chỉ ghép từ tên đường/phường/quận lấy từ MC-OCR train; tên món từ danh sách khoảng 300 từ tự soạn + từ ghép ngẫu nhiên từ `words.txt`; số tiền theo định dạng `125.000`, `125,000`, `125000đ`, `1.250.000 VND`; ngày `dd/mm/yyyy`, `dd-mm-yyyy`, `dd.mm.yyyy hh:mm`; số lượng, mã HĐ, SĐT ngẫu nhiên).
- Đã loại: SynthText (cần depth map, dành cho scene text); dùng thẳng SynthDoG không sửa (không có polygon, layout không giống hóa đơn).

## Non-functional
| Loại | Ngưỡng | Đo bằng |
|---|---|---|
| Sinh 20k hóa đơn | ≤ 60 phút, 8 worker | `time` |
| Sinh 1 dòng chữ | ≤ 5 ms trung bình | microbenchmark trong report |
| Dung lượng | ≤ 6 GB cho 21k hóa đơn JPEG | `du` |

## Flagged concerns
- **Dependency synthtiger — đã thử, loại (2026-11-11).** Cài được trên Python 3.12 nhưng `imgaug` 0.4.0 vỡ với numpy 2 (`np.sctypes` bị bỏ). Hạ `numpy<2` thì vỡ chỗ khác (`np.long`) và xung đột với scipy ≥ 1.18. Gói không còn hợp với stack hiện tại (dự án đang dùng numpy 2.5). **Chốt nhánh (b)**: tự viết bằng Pillow + OpenCV, không thêm dependency. Thiết kế và bộ tham số hiệu ứng lấy từ `synthdog/config_en.yaml` và `synthtiger/examples/synthtiger` (MIT, ghi nguồn trong docstring).
- Hóa đơn tổng hợp dùng font OFL hiện đại, còn hóa đơn thật in nhiệt. Cần thêm ít nhất 1–2 font kiểu in nhiệt/mono (ví dụ `Courier Prime`, `Share Tech Mono`, OFL) để giảm lệch phân phối. Vẫn sẽ có domain gap; Phase 1.5 (pseudo-label) và fine-tune trên MC-OCR là để bù.
- Corpus dùng tên cửa hàng/địa chỉ từ MC-OCR train. Đây là dữ liệu doanh nghiệp in trên hóa đơn công khai, không phải dữ liệu cá nhân, nhưng giấy phép MC-OCR chỉ cho nghiên cứu, nên corpus cũng chỉ dùng nghiên cứu và không commit vào git (`data/` đã ignore).

## Out of scope
- Pseudo-label MC-OCR bằng DBNet v1 (Phase 1.5, intent sau).
- VinText.
- Sinh chữ viết tay.
