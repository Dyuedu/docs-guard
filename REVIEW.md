# REVIEW.md — chính sách review PR

Áp dụng cho review của người và của AI (Claude Code, `claude-code-action`). Kết quả review của AI chỉ là góp ý: không tự approve, không tự block. Merge cần owner approve.

## Ba lượt review

### 1. Bug và tính đúng
- Logic sai, off-by-one, sai shape hoặc dtype, sai thứ tự kênh (BGR/RGB), sai chuẩn hóa ảnh.
- Tiền xử lý hoặc hậu xử lý lệch nhau giữa offline eval và Triton. Cả hai phải dùng chung `docguard.pipeline.processing`.
- Rò rỉ dữ liệu: ảnh gốc của cùng một hóa đơn nằm ở cả train và test, hoặc chọn hyperparameter trên test set.
- Test không assert gì, hoặc assert lại chính output của code (tautology). Giá trị kỳ vọng phải đến từ spec hoặc proposal.

### 2. Bảo mật và dữ liệu
- Secret hoặc token trong diff. Đọc `.env`.
- Upload ở gateway: kiểm tra kích thước, định dạng, cạnh dài ≤ 2048 px (proposal §2). Không tin metadata do client gửi.
- Không log nội dung ảnh hay text trích xuất ở mức INFO. Không dùng dữ liệu cá nhân thật (proposal §4).
- Giấy phép: code port từ repo khác phải có attribution. DocTamper chỉ được dùng cho nghiên cứu.

### 3. Khớp spec và plan
- PR link tới `intent/NNN-*/spec.md` và `plan.md`.
- Mỗi acceptance criterion có proof (test hoặc eval). Thiếu proof thì ghi rõ trong PR.
- Diff khớp `plan.md`. Phần lệch đã được ghi ở mục "Deviations".
- Thay đổi model, precision hoặc ngưỡng phải có kết quả eval so với baseline và MLflow run id.

## Mức độ

| Mức | Ý nghĩa |
|---|---|
| **blocker** | Sai kết quả, rò rỉ dữ liệu hoặc secret, vi phạm giấy phép, không khớp spec |
| **major** | Thiếu proof, lệch plan mà không giải thích, hồi quy hiệu năng |
| **minor** | Thiếu thông tin hoặc khó bảo trì nhưng không sai |
| **nit** | Style. Tối đa 3 nit mỗi review. |

## Bỏ qua khi review
`data/**`, `artifacts/**`, `mlruns/**`, `*.onnx`, `*.plan`, `*.engine`, `src/*.egg-info/**`, file lock và file sinh tự động.
