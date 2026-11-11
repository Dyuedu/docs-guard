# Plan 001 — Sinh và kiểm định tập giả mạo

Status: Approved
Date: 2026-11-11
Spec: ./spec.md

## Files to change
| File | Thay đổi |
|---|---|
| `tests/test_tamper_stats.py` (mới) | Test cho AC-1, AC-2, AC-5 trên fixture tổng hợp nhỏ: 3 ảnh trắng có box số, tạo trong `tmp_path` |
| `src/docguard/data/tamper_stats.py` (mới) | `load_index`, `leak_check`, `mask_stats`, `summarize`, `preview_grid` |
| `scripts/tamper_report.py` (mới) | CLI: chạy kiểm định, ghi report.json/md, ghi preview; exit 1 nếu có check FAIL |
| `scripts/make_tamper_dataset.py` | `--font` nhận file hoặc thư mục; thêm `--tampered-per-image-train`; ghi `donor`/`font` vào index; tách lõi thành hàm `generate(...)` để test gọi được |
| `README.md` | Cập nhật lệnh ở mục "1. Dữ liệu" (thêm `--splits`, report) |

## Order of work
1. Viết `tests/test_tamper_stats.py` trước. Test phải FAIL vì module chưa tồn tại.
2. Tách `generate()` trong `make_tamper_dataset.py`, thêm xoay vòng font và metadata (AC-5).
3. Implement `tamper_stats.py` để test PASS (AC-1, AC-2).
4. Viết `tamper_report.py`, gồm `summarize` và `preview_grid` (AC-3, AC-4, AC-6).
5. Chạy `pytest -q`.
6. Sinh dữ liệu thật:
   `make_tamper_dataset.py --ann data/processed/annotations.jsonl --splits data/processed/splits.json --images data/raw/mcocr/train_images --out data/tamper --font data/fonts --tampered-per-image 2 --tampered-per-image-train 4`
   Lệnh này chỉ được chạy khi owner đã chốt Open questions.
7. Chạy `tamper_report.py --preview 50`. Đính kèm report.md vào PR. Owner xem preview (AC-7).
8. Chạy subagent `verifier`, mở PR từ branch `feat/001-tamper-dataset`.

## Proof
- Test: `pytest -q`, gồm các test mới của AC-1, AC-2, AC-5.
- Report trên dữ liệu thật: `checks.*` đều PASS, cùng số liệu về tỉ lệ phép biến đổi, diện tích mask và outliers (AC-3, AC-4, AC-6).
- Owner xem tay preview (AC-7).
- Không đụng model nên chưa có eval hay MLflow. Hash `splits.json` được ghi vào report.json để truy vết.

## Risks
- Otsu thất bại trên box có độ tương phản thấp, làm mask inpaint phình to. AC-4 bắt được qua `outliers`. Cách sửa (nếu cần) là giới hạn mask trong bbox, và việc này sẽ thành một intent fix riêng.
- Font có kích thước `h*0.8` có thể tràn ra ngoài bbox. Mask vẫn đúng vì tính từ pixel thực sự thay đổi, nhưng trông thiếu tự nhiên. AC-7 sẽ phát hiện.
- Dung lượng đĩa ước tính: (924×5 + 231×3) ảnh JPEG q100 cộng mask PNG, khoảng 3–4 GB.

## Deviations (điền khi làm xong)
- `summarize(index, stats, leak_errors, cfg=None)`: leak errors được truyền vào thay vì tính bên trong, để report và test dùng chung một `leak_check`. `cfg` là tham số tùy chọn, mặc định lấy ngưỡng của spec.
- Splice không chọn chính hóa đơn đang xét làm donor nếu split có hơn 1 hóa đơn. Code cũ có thể tự splice từ chính nó, khi đó thực chất là copy-move.
- Không có thông tin font thì generate chỉ dùng copy_move và splice.
- Preview được chia file theo split (`train_XX.jpg`, `val_XX.jpg`), 10 dòng mỗi file.

## Kết quả chạy thật (2026-11-11, splits.json sha256 `7d19106a…`)
- Sinh dữ liệu: 2 phút 19 giây, 1,7 GB. Ảnh train/val/test: 4.483/339/338. Ảnh giả: 3.559/223/223.
- Report: 40 giây. `no_leak`, `masks_valid`, `op_balance`, `mask_area` đều PASS. Mỗi phép biến đổi chiếm 30–36%. Trung vị diện tích mask: 0,00225/0,00215/0,00258.
- **Phát hiện lỗi (chờ owner quyết):** cả 14 outlier (ratio > 0,05) đều là inpaint trên box số **dọc** (h > w, ảnh hóa đơn bị xoay; 122/3.338 box, 52 hóa đơn). `inpaint_and_render` dùng cỡ font `h*0.8` và luôn vẽ ngang, nên chữ khổng lồ tràn ra nửa hóa đơn. Số ảnh inpaint có ratio > 1%: train 71/1.213, val 1/75, test 4/70.
- **Fix (owner duyệt, theo quy trình bug):** commit `4c3c96d` thêm test FAIL cho AC-8, sau đó sửa `inpaint_and_render`:
  - bỏ qua box dọc;
  - `_fit_font` giảm cỡ chữ đến khi vừa box;
  - vẽ trên vùng cắt của box.
  - Chỉ sửa code, không sửa test.
- **Sinh lại** (đã xóa `data/tamper` cũ, 2 phút 22 giây):
  - Số ảnh train/val/test: 4.441/337/336; trong đó ảnh giả: 3.517/221/221.
  - 4 check đều PASS. Mỗi phép biến đổi chiếm 29–37%. Trung vị diện tích mask: 0,00215/0,00220/0,00222.
  - Diện tích mask inpaint lớn nhất giảm từ 0,126 xuống 0,027.
- Outlier còn lại: 1 ảnh (`train/mcocr_public_145014wvxfq_copy_move_3`, 0,128). Nguyên nhân là **lỗi nhãn MC-OCR**: box ADDRESS rộng 587×577px, phủ nửa hóa đơn. Mẫu này được giữ lại và report đã gắn cờ. Lọc box có nhãn bất thường là việc của một intent sau, nếu cần.
- AC-7 (owner xem 50 ảnh trong `results/data/tamper/preview/`): **chờ owner**.
