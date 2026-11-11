# Spec NNN — [tên ngắn]

Status: Draft            <!-- Draft | Accepted | Implemented | Superseded by NNN -->
Date: YYYY-MM-DD
Intent: ./intent.md

## Acceptance criteria
Mỗi dòng phải đo được. Test hoặc eval tham chiếu ID ở cột đầu.

| ID | Tiêu chí | Proof |
|---|---|---|
| AC-1 | Given … when … then … | `tests/test_x.py::test_y` |
| AC-2 | [metric] so với B0 giảm ≤ … | eval, MLflow run |

## Interface / data contract
Hàm, schema, config hoặc định dạng file thay đổi. Agent không được dùng contract nằm ngoài mục này.

```python
def fn(x: InputType) -> OutputType: ...
```

## Technical approach
- Lựa chọn: …
- Tham khảo: [repo/paper + license]. Lấy gì, sửa gì.
- Phương án đã loại và lý do: …

## Non-functional
| Loại | Ngưỡng | Đo bằng |
|---|---|---|
| Latency p95 | … | `docguard.bench.latency` |
| Peak VRAM | … | … |

## Flagged concerns
Những chỗ xung đột với ràng buộc, rủi ro cao, hoặc cần owner quyết định.
- …

## Out of scope
- …
