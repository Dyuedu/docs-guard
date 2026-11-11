# Intent 001 — Sinh và kiểm định tập giả mạo (Phase 1.3)

Status: Accepted
Date: 2026-11-11
Owner: hoangduy20407
Proposal ref: docs/proposal.md §4, §7

## Problem
CAT-Net-lite (Phase 2.3) cần ảnh giả mạo kèm mask pixel. `scripts/make_tamper_dataset.py` đã có nhưng chưa từng chạy trên MC-OCR thật. Hiện chưa có cách nào kiểm tra bằng số liệu xem kết quả sinh ra có đúng không. Có thể sai ở các chỗ sau:
- rò rỉ dữ liệu giữa các split;
- mask rỗng, hoặc mask lớn bất thường do Otsu thất bại khi inpaint;
- lệch tỉ lệ giữa các phép biến đổi, vì copy-move cần ít nhất 2 box số;
- kích thước ảnh và mask không khớp nhau.

Nếu dữ liệu train sai thì mọi metric pixel-F1 về sau đều vô nghĩa.

## Proposed outcome
- `data/tamper/{train,val,test}` được sinh hoàn toàn bằng script, theo đúng `data/processed/splits.json`.
- Có báo cáo thống kê tự động. Mọi bất biến (không rò rỉ, mask hợp lệ, tỉ lệ phép biến đổi) đều PASS.
- Owner xem bảng ảnh xem trước của 50 cặp ảnh/mask và xác nhận chất lượng đạt.

## Affected components
- `scripts/make_tamper_dataset.py`: thêm xoay vòng font và ghi metadata vào index.
- Mới: `src/docguard/data/tamper_stats.py`, `scripts/tamper_report.py`.
- Đầu ra do script sinh ra: `data/tamper/`, `results/data/tamper/`.

## Constraints
- MC-OCR chỉ cho phép dùng vào nghiên cứu. Owner cần gửi thỏa thuận cho VNDAG.
- Không sửa tay `data/` hay `results/`, chỉ sinh bằng script. Hook đang chặn việc sửa tay.
- Test set chỉ dùng để báo cáo cuối. Ảnh xem trước chỉ lấy từ train và val.
- Font phải phủ đủ `configs/vocab_vi.txt`. Đã kiểm tra: 3 font OFL trong `data/fonts/` đều đạt.

## Open questions
- [x] `--tampered-per-image`: giữ 2 (train ≈ 924 ảnh thật cộng ≈ 1.8k ảnh giả) hay tăng lên 4 cho train để có nhiều dữ liệu hơn? **Chốt (owner, 2026-11-11): train = 4, val/test = 2.**
- [x] Inpaint có xoay vòng 3 font không, hay chỉ dùng 1 font? **Chốt: xoay vòng 3 font.**
- [x] Ngưỡng ảnh hỏng chấp nhận được khi owner xem tay: **Chốt: ≤ 5/50 (AC-7).**
