# Intent 002 — Dữ liệu chữ tổng hợp cho OCR (Phase 1.4)

Status: Accepted
Date: 2026-11-11
Owner: hoangduy20407
Proposal ref: docs/proposal.md §4, §5

## Problem
MC-OCR chỉ gán nhãn 4 trường (SELLER, ADDRESS, TIMESTAMP, TOTAL_COST), khoảng 6,6k box trên 1.155 ảnh. Các dòng còn lại trên hóa đơn (tên món, đơn giá, số lượng, mã hóa đơn…) không có nhãn. Hệ quả:
- **DBNet** train trên MC-OCR sẽ học "chỉ 4 dòng kia là chữ", còn lại là nền. Model sẽ bỏ sót phần lớn dòng chữ.
- **CRNN** chỉ có 6,6k crop, với 924 ảnh train thì khoảng 5,3k crop. Số này quá ít để học 221 ký tự tiếng Việt từ đầu (đã chốt: tự train, không dùng pretrained).

Cần một nguồn dữ liệu có nhãn đầy đủ: hóa đơn tổng hợp có polygon cho **mọi** dòng chữ, và ảnh dòng chữ tiếng Việt với số lượng lớn.

## Proposed outcome
- `data/synth_ocr/receipts/{train,val}/`: hóa đơn tổng hợp toàn trang, mỗi ảnh kèm polygon và text của mọi dòng. Định dạng jsonl giống `annotations.jsonl` (thêm `polygon`), để DBNet dùng chung một loader cho cả dữ liệu thật và tổng hợp.
- `data/synth_ocr/lines/`: ảnh dòng chữ 32×W kèm nhãn, sinh sẵn một tập val nhỏ. Tập train sinh on-the-fly trong DataLoader (Phase 2.2), cùng một hàm sinh.
- `data/synth_ocr/corpus/`: corpus tiếng Việt kiểu hóa đơn, chỉ lấy từ **split train** của MC-OCR cộng sinh theo mẫu.
- Report tự động: charset ⊆ vocab, phân bố độ dài, số dòng/ảnh, preview để owner xem.

## Affected components
- Mới: `src/docguard/data/synth_text.py` (corpus + sinh dòng chữ), `src/docguard/data/synth_receipt.py` (hóa đơn toàn trang), `scripts/make_synth_ocr.py`, `scripts/synth_ocr_report.py`.
- `pyproject.toml`: thêm dependency sinh dữ liệu (xem Open questions).
- `configs/train/text_det.yaml`, `text_rec.yaml`: đường dẫn dữ liệu (đã khai sẵn ở Phase 0).
- `data/fonts/`: cần thêm font kiểu máy in nhiệt / mono (OFL).

## Constraints
- Charset phải ⊆ `configs/vocab_vi.txt` (221 ký tự). Đã kiểm tra: toàn bộ 181 ký tự xuất hiện trong MC-OCR đều nằm trong vocab.
- Corpus **không được** lấy text từ split val/test của MC-OCR (rò rỉ vào CRNN).
- Không dùng dữ liệu cá nhân thật. Số điện thoại, mã số thuế, địa chỉ trong corpus phải sinh ngẫu nhiên hoặc lấy từ MC-OCR train (là dữ liệu cửa hàng, không phải cá nhân).
- Font chỉ dùng loại giấy phép OFL/Apache, kèm file license.
- Sinh được trên CPU laptop trong thời gian hợp lý (mục tiêu dưới 1 giờ cho 20k hóa đơn, nhiều worker).
- Theo CLAUDE.md: dựa trên mã nguồn mở, không tự viết từ đầu.

## Research (đã làm)
| Repo | License | Dùng được gì |
|---|---|---|
| [clovaai/donut → synthdog](https://github.com/clovaai/donut) | MIT | Hóa đơn/văn bản toàn trang: nền, giấy, layout lưới, hiệu ứng (elastic, perspective, bóng, blur). **Thiếu:** không xuất polygon từng dòng chữ, chỉ xuất chuỗi text. Phải sửa `template.py` để dump `quad` của từng text layer (quad được cập nhật sau hiệu ứng perspective ở mức document). |
| [clovaai/synthtiger](https://github.com/clovaai/synthtiger) (1.2.1) | MIT | Nền của SynthDoG. Template `examples/synthtiger` sinh dòng chữ kèm `gt.txt`, bbox từng ký tự, nhiều hiệu ứng. Dependency nặng: `imgaug`, `pygame`, `opencv-python` (xung đột với `opencv-python-headless` của repo). Hỗ trợ Python ≥ 3.6, chưa rõ 3.12. |
| [Belval/TextRecognitionDataGenerator](https://github.com/Belval/TextRecognitionDataGenerator) | MIT | Dòng chữ, đơn giản, Pillow thuần. Ít hiệu ứng hơn synthtiger. Phương án dự phòng nếu synthtiger không cài được. |
| [oh-my-ocr/text_renderer](https://github.com/oh-my-ocr/text_renderer) | MIT | Dòng chữ, còn bảo trì (05/2026). Cấu hình bằng Python. |
| [ankush-me/SynthText](https://github.com/ankush-me/SynthText) | Apache-2.0 | Scene text, cần depth map. Không hợp với hóa đơn. |

## Open questions
- [x] **Dependency:** dùng `synthtiger` (một gói cho cả 2 việc, nhiều hiệu ứng, nhưng nặng và chưa chắc chạy trên 3.12, đang thử cài) hay tự viết lớp mỏng bằng Pillow/OpenCV theo thiết kế của SynthDoG (ít dependency, nhưng phải tự làm hiệu ứng)? **Chốt (2026-11-11): tự viết bằng Pillow + OpenCV.** synthtiger cài được nhưng imgaug vỡ với numpy 2; hạ numpy<2 vẫn vỡ (`np.long`) và xung đột scipy.
- [x] Số lượng hóa đơn tổng hợp: **chốt 20k train + 1k val.**
- [x] Layout hóa đơn: SynthDoG dùng layout lưới chung chung (như trang wiki). Hóa đơn thật có cấu trúc: header (tên, địa chỉ, SĐT) → bảng món (tên | SL | giá) → tổng → ngày/cảm ơn. Đề xuất viết **layout hóa đơn riêng** (một file Python nhỏ) thay cho `GridStack` của SynthDoG, còn nền/giấy/hiệu ứng giữ nguyên. **Chốt: layout hóa đơn riêng.**
- [x] Có thêm VinText (scene text tiếng Việt, giấy phép nghiên cứu) không? **Chốt: không.**
