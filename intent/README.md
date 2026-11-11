# intent/

Mỗi đơn vị công việc có một thư mục `intent/NNN-<slug>/` (ví dụ `intent/003-crnn-ctc-training/`), gồm 3 file theo thứ tự:

| File | Giai đoạn | Ai viết | Ai duyệt |
|---|---|---|---|
| `intent.md` | Plan | Agent soạn từ trao đổi | Owner chấp nhận hoặc đóng |
| `spec.md` | Design | Agent | Owner |
| `plan.md` | Build | Agent (plan mode) | Owner, trước khi sửa code |

Template nằm trong `intent/_template/`. Quy trình đầy đủ: `docs/process/ai-native-sdlc.md`.

Trạng thái ghi ở dòng `Status:` đầu mỗi file. Không xóa intent cũ. Intent bị bỏ thì đặt `Status: Closed` và ghi lý do.
