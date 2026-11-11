# Quy trình AI-native SDLC — doc-guard

> Dựa trên Anthropic, *The AI-Native SDLC Playbook* (08/2026), có điều chỉnh cho một repo ML nhỏ.
> Ý chính của playbook: vẫn giữ 6 giai đoạn, nhưng agent là người thực thi chính. Con người duyệt tại các gate
> và chịu trách nhiệm cho mọi quyết định cần phán đoán. Agent được làm tới sát gate production nhưng không được tự vượt qua.
>
> Phần điều chỉnh cho doc-guard:
> - Thêm **eval model** (pixel-F1, CER, Field EM, latency, VRAM) làm phần "proof" của Test. Unit test không đủ làm oracle cho DBNet, CRNN hay CAT-Net.
> - Chưa có production thật, nên Deploy chỉ gồm build TensorRT engine, cập nhật Triton model repository và chạy benchmark.
> - Phần Maintain (theo dõi drift theo ngưỡng σ) để sau, khi gateway có traffic.

## 1. Nguyên tắc

1. **Intent và spec có trước, code có sau.** Spec là nguồn sự thật. Code lệch spec thì sửa code hoặc sửa spec qua review, không im lặng bỏ qua.
2. **Plan được duyệt trước khi sửa code.**
3. **Proof là bắt buộc.** Mỗi thay đổi phải có test hoặc eval chứng minh. Fix bug thì viết test FAIL trước.
4. **Người giữ các gate.** Agent không tự merge, không push lên `main`, không tự đổi ngưỡng mục tiêu trong `docs/proposal.md`.
5. **Guardrail chặn bằng code, không chỉ bằng lời dặn.** Hook trong `.claude/settings.json` chặn sửa dữ liệu, kết quả và mục tiêu.

## 2. Các giai đoạn

| Giai đoạn | Agent làm | Artifact | Gate (người) |
|---|---|---|---|
| **Plan** | Soạn intent từ trao đổi với owner: vấn đề, kết quả mong muốn, thành phần bị ảnh hưởng, ràng buộc, câu hỏi mở. | `intent/NNN-slug/intent.md` | Owner chấp nhận hoặc đóng intent |
| **Design** | Viết spec: acceptance criteria đo được, interface, ngưỡng eval, lựa chọn kỹ thuật kèm repo tham khảo (CLAUDE.md: research trước khi tự viết). Gắn cờ những chỗ xung đột với ràng buộc. | `intent/NNN-slug/spec.md` | Owner chấp nhận spec. Quyết định rủi ro cao (đổi model, precision, ngưỡng quyết định) phải được owner xác nhận riêng. |
| **Build** | Ở plan mode: liệt kê file sẽ đổi, thứ tự làm, rủi ro, proof. Sau khi plan được duyệt mới code. Task song song thì mỗi task một worktree. | `intent/NNN-slug/plan.md`, code | Owner duyệt plan **trước khi** sửa code |
| **Test** | Chạy `pytest -q` (lệnh một dòng, exit ≠ 0 khi fail). Đổi model hoặc tiền xử lý thì chạy eval và ghi MLflow run id. Subagent `verifier` kiểm tra lại trong context mới, chỉ báo cáo, không sửa. | Kết quả test/eval trong PR | CI xanh |
| **Deploy** | Soạn PR. Review theo `REVIEW.md` qua 3 lượt: bug, bảo mật, khớp spec/plan. Build engine và cập nhật Triton config khi spec yêu cầu. | PR, Triton model version | Owner approve PR (branch protection). AI review chỉ là góp ý, không tự approve hay block. |
| **Maintain** *(sau)* | Phát hiện bất thường bằng rule tất định (baseline + ngưỡng σ). Agent chẩn đoán rồi viết thành `intent.md` mới. | `intent/` | Owner quyết định sửa ngay, lên lịch hay bỏ qua |

## 3. Áp dụng theo loại thay đổi

| Loại thay đổi | Bắt buộc |
|---|---|
| Typo, docs, format | PR + review |
| Fix bug không đổi hành vi công khai | Test tái hiện FAIL được commit trước → fix (bật test-lock) → PR |
| Tính năng hoặc module mới | intent → spec → plan → code → test → PR |
| Đổi model, precision, tiền xử lý, ngưỡng `configs/pipeline.yaml` | Như trên, **thêm** eval so với baseline. Không được vượt mức suy giảm trong `docs/proposal.md` §2. |

## 4. Proof cho model

- Ngưỡng lấy từ `docs/proposal.md` §2 và §7. So với B0 (PyTorch FP32) trên cùng GPU T4: pixel-F1 và Field EM giảm không quá 1,5 điểm %.
- Mỗi lần train, export hoặc benchmark ghi vào MLflow: git SHA, file config, hash của `data/processed/splits.json`, biến thể (B0–B5), phiên bản driver/TensorRT/Triton.
- Số liệu trong README hoặc báo cáo phải có MLflow run id đi kèm. Không chép số từ trí nhớ hay từ log rời.
- Test set chỉ dùng để báo cáo cuối, không dùng để chọn hyperparameter.

## 5. Guardrail tự động (`.claude/settings.json` → `.claude/hooks/guard_edits.py`)

| Chặn | Lý do |
|---|---|
| Sửa trực tiếp `data/**`, `artifacts/**`, `mlruns/**`, `results/**` | Dữ liệu và kết quả chỉ được tạo ra bằng script, không sửa tay |
| Sửa `docs/proposal.md` | Mục tiêu và ngưỡng do owner quyết định |
| Sửa `.claude/settings.json`, `.claude/hooks/**` | Agent không được tự nới guardrail |
| Sửa `tests/**` khi có file `.claude/test-lock` | Khi fix bug: test FAIL đã commit thì chỉ được sửa code |
| Nội dung trông giống secret (private key, token) | Giữ credential ngoài diff |
| Đọc `.env*` (permission deny) | Như trên |

Bật test-lock bằng `touch .claude/test-lock`, tắt bằng `rm .claude/test-lock`. File này không được commit.

## 6. Đo lường (xem lại mỗi phase)

- Tỉ lệ PR merge ngay lần đầu, không phải làm lại.
- Diff đã merge có khớp `plan.md` không.
- Số lần spec phải sửa sau khi đã có plan (đo mức rework).
- Lỗi lặp lại mà lẽ ra CLAUDE.md đã phải ngăn. Gặp lỗi lặp thì thêm một dòng vào mục "Recurring mistakes" của CLAUDE.md.
