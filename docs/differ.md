
**MC-OCR 2021 là gì**

MC-OCR (Mobile-Captured Receipt Recognition) là một cuộc thi do nhóm VNDAG tổ chức tại hội nghị RIVF 2021. Các đội thi trên CodaLab từ tháng 12/2020 đến tháng 1/2021, với hai đề bài:

1. **Đánh giá chất lượng ảnh:** dự đoán một điểm số cho biết ảnh hóa đơn chụp bằng điện thoại có đủ tốt để dùng không. Đây là cột `anno_image_quality` trong CSV.
2. **Trích xuất thông tin (KIE):** lấy ra 4 trường `SELLER`, `ADDRESS`, `TIMESTAMP`, `TOTAL_COST`. Kết quả chấm bằng tỉ lệ lỗi ký tự (CER) giữa chuỗi dự đoán và đáp án.

Bộ dữ liệu gồm khoảng 1,1 nghìn ảnh train có nhãn và 391 ảnh test public không có nhãn. Đây là bộ dữ liệu bạn vừa tải. Các đội thường làm theo chuỗi: chuẩn hóa ảnh (xoay, cắt nền) → phát hiện chữ → nhận dạng chữ → gán nhãn trường. Phần lớn dùng mô hình pretrained có sẵn như PaddleOCR hay VietOCR. Mình chưa lấy được bảng xếp hạng và điểm CER cụ thể, nên không nêu số ở đây.

**Dự án này khác gì so với cuộc thi**

|                                | MC-OCR 2021                                          | doc-guard                                                                                                                                                                                                    |
| ------------------------------ | ---------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Bài toán chính              | Đọc đúng 4 trường và chấm chất lượng ảnh | **Phát hiện ảnh bị chỉnh sửa ở mức pixel** cộng KIE. Mục tiêu là chống gian lận khi giải ngân                                                                                          |
| Chỉnh sửa ảnh               | Không có                                           | Là trọng tâm: copy-move, splicing, inpaint, có mask, và trả về `overlaps_fields`, tức vùng bị sửa có đè lên tổng tiền hay ngày không                                                    |
| Thước đo thành công       | Độ chính xác (CER) trên leaderboard             | Độ chính xác **cộng chi phí chạy**: giảm ≥ 50% latency, giảm ≥ 40% VRAM, throughput ≥ 2× khi có 32 người dùng đồng thời, và độ chính xác không được giảm quá 1,5 điểm |
| Kết quả nộp                 | File `results.csv` nộp offline                    | Một microservice chạy thật: Triton + gRPC, FastAPI, Docker, đo tải bằng Locust                                                                                                                         |
| Tối ưu mô hình             | Thường không quan tâm                            | Là phần nghiên cứu chính: ONNX, TensorRT FP16/INT8, pruning, ma trận thí nghiệm B0–B5                                                                                                               |
| Câu hỏi nghiên cứu         | Mô hình nào đọc chuẩn nhất                    | INT8 làm hỏng tín hiệu phát hiện giả mạo nhiều hơn hay làm hỏng OCR nhiều hơn (so sánh B3 với B5)                                                                                            |
| Cách dùng dữ liệu MC-OCR   | Là toàn bộ bài toán                             | Chỉ là **nguyên liệu**: dùng làm ảnh gốc để sinh dữ liệu giả mạo, và làm tập fine-tune cho OCR/KIE                                                                                    |
| Đánh giá chất lượng ảnh | Đề bài 1                                          | Bỏ qua                                                                                                                                                                                                      |

Tóm lại, cuộc thi hỏi "đọc hóa đơn chuẩn đến đâu". Dự án này hỏi "hóa đơn có bị sửa không, sửa ở đâu, và hệ thống chạy được rẻ và nhanh đến mức nào khi có tải thật".

**Hệ quả với dự án**

- Kết quả KIE không thể so trực tiếp với leaderboard năm 2021. Cuộc thi chấm bằng CER trên tập test ẩn, còn mình dùng Field Exact Match trên một phần tách ra từ tập train.
- Dữ liệu giả mạo là tự sinh. Chưa ai, kể cả cuộc thi, kiểm chứng phương pháp này trên gian lận thật. README đã ghi rõ giới hạn này.

Nguồn:

- [MC-OCR preprint](https://people.cs.umu.se/sonvx/files/MCOCR_Preprint.pdf)
- [CodaLab: MC-OCR competition](https://competitions.codalab.org/competitions/27798)
- [dblp: RIVF 2021 MC-OCR papers](https://dblp.org/rec/conf/rivf/TranHHABAPNNL21.html)
- [Proceedings RIVF 2021 TOC](https://proceedings.com/content/061/061718webtoc.pdf)
