# Bảng Chú Thích và Giải Nghĩa Thuật Ngữ — Dự Án doc-guard

> **Tài liệu tham chiếu chuẩn hóa về khái niệm, công nghệ, kiến trúc và từ viết tắt trong toàn bộ hệ thống `doc-guard`.**

---

## Mục lục

1. [Tổng quan dự án & Kiến trúc hệ thống (System & Architecture)](#1-tổng-quan-dự-án--kiến-trúc-hệ-thống)
2. [Mô hình Trí tuệ Nhân tạo & Thị giác Máy tính (AI & Computer Vision Models)](#2-mô-hình-trí-tuệ-nhân-tạo--thị-giác-máy-tính)
3. [Xử lý Ảnh & Kỹ thuật Sinh Dữ liệu Giả mạo (Image Processing & Forensics)](#3-xử-lý-ảnh--kỹ-thuật-sinh-dữ-liệu-giả-mạo)
4. [Tối ưu hóa Mô hình, Lượng tử hóa & Runtime (Optimization & Runtime)](#4-tối-ưu-hóa-mô-hình-lượng-tử-hóa--runtime)
5. [Ma trận Thí nghiệm (Experiment Matrix B0 – B5)](#5-ma-trận-thí-nghiệm-b0--b5)
6. [Tập Dữ liệu, Phân chia & Định dạng (Datasets & Formats)](#6-tập-dữ-liệu-phân-chia--định-dạng)
7. [Chỉ số Đánh giá & Thước đo Hiệu năng (Evaluation Metrics & Benchmarks)](#7-chỉ-số-đánh-giá--thước-đo-hiệu-năng)
8. [Nghiệp vụ Hóa đơn & Bóc tách Thông tin (Business Logic & KIE Fields)](#8-nghiệp-vụ-hóa-đơn--bóc-tách-thông-tin)
9. [MLOps, Quy trình AI-Native SDLC & Tuân thủ Pháp lý (Governance & MLOps)](#9-mlops-quy-trình-ai-native-sdlc--tuân-thủ-pháp-lý)
10. [Bảng Tra Cứu Nhanh Từ Viết Tắt (Alphabetical Acronym Index)](#10-bảng-tra-cứu-nhanh-từ-viết-tắt)

---

## 1. Tổng quan dự án & Kiến trúc hệ thống

```
client ──REST──► FastAPI gateway ──gRPC──► Triton Inference Server
                                            └─ pipeline (Python BLS, CPU)
                                                 ├─ tamper_seg  (patch 512², FP16)   ┐ chạy
                                                 └─ text_det → crop → text_rec (INT8) ┘ song song
                                                 → KIE rules → decision → JSON
```

### doc-guard
- **Định nghĩa:** Tên của dự án/microservice.
- **Ý nghĩa trong dự án:** Hệ thống dịch vụ tự động tiếp nhận ảnh chụp hóa đơn tiếng Việt, thực hiện đồng thời hai nhiệm vụ: (1) phát hiện và khoanh vùng chỉnh sửa giả mạo (tampering localization) và (2) bóc tách các trường thông tin quan trọng (OCR + KIE), được tối ưu hóa bằng TensorRT/ONNX và phục vụ trên Triton Inference Server.

### Microservice
- **Định nghĩa:** Kiến trúc dịch vụ phân tán vi mô, phân chia hệ thống thành các khối độc lập có trách nhiệm chuyên biệt.
- **Ý nghĩa trong dự án:** Tách biệt giữa tầng đón nhận request (API Gateway viết bằng FastAPI) và tầng tính toán suy luận AI tải nặng (Triton Inference Server).

### Gateway / FastAPI Gateway
- **Định nghĩa:** Cổng giao tiếp API biên (Edge API Gateway) xây dựng trên nền web framework FastAPI (Python).
- **Ý nghĩa trong dự án:** Đặt tại `serving/gateway/app.py`. Có nhiệm vụ: kiểm tra content-type và kích thước file upload (tối đa 10 MB, đổi được qua `MAX_UPLOAD_BYTES`), kiểm tra magic bytes của ảnh (`JPEG/PNG`), gọi sang Triton qua giao thức gRPC và định dạng đầu ra chuẩn theo schema Pydantic `VerifyResponse`.

### Triton Inference Server
- **Định nghĩa:** Máy chủ phục vụ suy luận mô hình học sâu chuyên dụng nguồn mở hiệu năng cao của NVIDIA.
- **Ý nghĩa trong dự án:** Chạy trong container Docker, quản lý vòng đời, phân phối tài nguyên tính toán (CPU/GPU) và tối ưu hóa hàng đợi cho 4 mô hình: `pipeline`, `tamper_seg`, `text_det`, `text_rec`.

### BLS (Business Logic Scripting)
- **Định nghĩa:** Tính năng của Triton Python Backend cho phép nhúng mã nguồn Python tùy biến để điều phối luồng gọi giữa các mô hình bên trong Triton server.
- **Ý nghĩa trong dự án:** Nằm tại `serving/model_repository/pipeline/1/model.py`. BLS cho phép giải quyết bài toán "quạt ra động" (*variable fan-out*): số lượng crop chữ của mỗi hóa đơn là khác nhau, đồng thời kích hoạt chạy song song bất đồng bộ (`asyncio.gather`) giữa nhánh phát hiện giả mạo (`tamper_seg`) và nhánh OCR (`text_det` → `text_rec`).

### Static Triton Ensemble
- **Định nghĩa:** Mô hình điều phối luồng tính toán dạng đồ thị tĩnh định nghĩa thuần qua file cấu hình `config.pbtxt` của Triton.
- **Ý nghĩa trong dự án:** Không sử dụng Ensemble tĩnh vì kích thước tensor đầu ra của Text Detection (số lượng hộp chữ nhật crop) thay đổi theo từng ảnh đầu vào, vượt ngoài khả năng biểu diễn của đồ thị tĩnh. BLS được chọn để thay thế.

### Model Repository
- **Định nghĩa:** Cấu trúc thư mục chuẩn theo quy định của Triton dùng để chứa cấu hình (`config.pbtxt`), mã thực thi (`model.py`) hoặc trọng số đã biên dịch (`model.plan`, `model.onnx`).
- **Ý nghĩa trong dự án:** Thư mục `serving/model_repository/` gồm 4 thư mục con: `pipeline`, `tamper_seg`, `text_det`, `text_rec`.

### gRPC (Google Remote Procedure Call)
- **Định nghĩa:** Giao thức gọi thủ tục từ xa hiệu năng cao chạy trên nền HTTP/2 và mã hóa nhị phân Protocol Buffers (Protobuf).
- **Ý nghĩa trong dự án:** Giao thức kết nối giữa FastAPI Gateway và Triton (cổng 8001), giúp truyền tải mảng byte ảnh và tensor nhị phân với độ trễ tối thiểu và không bị phân mảnh như REST/JSON.

### Dynamic Batching
- **Định nghĩa:** Tính năng tự động gom các yêu cầu suy luận đến riêng lẻ trong một cửa sổ thời gian trễ cho phép (`max_queue_delay_microseconds`) thành một lô (batch) lớn hơn trước khi đẩy vào GPU.
- **Ý nghĩa trong dự án:** Cấu hình trong `config.pbtxt` của các mô hình TRT (ví dụ: `preferred_batch_size: [8, 16]`, trễ tối đa 2000 µs), giúp tối đa hóa hiệu suất phần cứng khi chịu tải cao mà không tăng độ trễ đáng kể.

### Instance Group
- **Định nghĩa:** Cấu hình của Triton xác định số lượng bản sao thực thi đồng thời (worker instances) và phần cứng gán cho từng mô hình (`KIND_CPU` hoặc `KIND_GPU`).
- **Ý nghĩa trong dự án:** Mô hình `pipeline` chạy trên `4 CPU instances` để tiền/hậu xử lý ảnh không chiếm GPU; các mô hình suy luận `tamper_seg`, `text_det`, `text_rec` chạy trên `1 GPU instance`.

---

## 2. Mô hình Trí tuệ Nhân tạo & Thị giác Máy tính

```
                   ┌─ tamper_seg (CAT-Net-lite: RGB + DCT) ──► Mask & Score
                   │
Ảnh hóa đơn đầu vào ┤
                   └─ text_det (DBNet) ──► Bounding Boxes ──► Crop
                                                                │
                                                                ▼
                     KIE Rules ◄── ctc_greedy_decode ◄── text_rec (CRNN + CTC)
```

### Tampering Localization
- **Định nghĩa:** Bài toán khoanh vùng và định vị vị trí các điểm ảnh bị chỉnh sửa hoặc cắt dán trên tài liệu ở cấp độ pixel (pixel-level binary segmentation).
- **Ý nghĩa trong dự án:** Được thực hiện bởi mô hình `tamper_seg`. Đầu ra là bản đồ xác suất (probability map) biểu thị khả năng từng pixel bị can thiệp.

### OCR (Optical Character Recognition)
- **Định nghĩa:** Nhận dạng ký tự quang học — quá trình chuyển đổi ảnh chụp văn bản thành văn bản dạng số (text/string).
- **Ý nghĩa trong dự án:** Gồm hai giai đoạn tuần tự:
  1. **Text Detection (`text_det`):** Xác định vị trí tọa độ các dòng/từ.
  2. **Text Recognition (`text_rec`):** Đọc chuỗi ký tự từ từng vùng ảnh cắt ra.

### KIE (Key Information Extraction)
- **Định nghĩa:** Bóc tách thông tin trọng tâm — trích xuất các trường dữ liệu ngữ nghĩa có cấu trúc từ tập hợp các dòng văn bản không cấu trúc của tài liệu.
- **Ý nghĩa trong dự án:** Trích xuất 3 trường: đơn vị phát hành (`seller`), ngày lập (`timestamp`), và số tiền thanh toán (`total_amount`).

### DBNet (Real-time Scene Text Detection with Differentiable Binarization)
- **Định nghĩa:** Kiến trúc mạng nơ-ron phát hiện chữ trong ảnh thời gian thực, tích hợp hàm xấp xỉ nhị phân hóa vi phân được (Differentiable Binarization) trực tiếp vào quá trình lan truyền ngược.
- **Ý nghĩa trong dự án:** Mô hình Text Detection chính (`src/docguard/models/dbnet.py`, intent 003).
  - Backbone MobileNetV3-large của torchvision; neck DBFPN và head DB port từ PaddleOCR.
  - Head xác suất trả logits khi train; sigmoid chỉ áp khi `eval()`. Binary map = $\sigma(k\cdot(\sigma(\text{logits}) - \text{thresh}))$ với $k = 50$.
  - Đầu vào `960×960` (letterbox). Bản export ONNX chỉ xuất bản đồ xác suất `prob [N, 1, 960, 960]`; nhánh threshold không chạy khi suy luận.

### CRNN (Convolutional Recurrent Neural Network)
- **Định nghĩa:** Mạng học sâu kết hợp giữa mạng tích chập (CNN - trích xuất đặc trưng không gian), mạng hồi quy (RNN/BiLSTM - học tương quan tuần tự của ngôn ngữ) và tầng giải mã CTC.
- **Ý nghĩa trong dự án:** Mô hình Text Recognition đọc dòng chữ kích thước chuẩn `32×320`. Không sử dụng cơ chế tự hồi quy (non-autoregressive), giúp tốc độ suy luận nhanh và dễ dàng lượng tử hóa INT8.

### BiLSTM (Bidirectional Long Short-Term Memory)
- **Định nghĩa:** Mạng bộ nhớ ngắn-dài hai chiều, quét chuỗi đặc trưng theo cả chiều tiến và lùi để nắm bắt ngữ cảnh văn bản.
- **Ý nghĩa trong dự án:** Tầng Recurrent trong CRNN gồm 2 lớp BiLSTM với kích thước ẩn (hidden size) 256.

### CTC (Connectionist Temporal Classification)
- **Định nghĩa:** Hàm mất mát và kỹ thuật suy luận dành cho bài toán xử lý chuỗi không cần căn chỉnh nhãn trước (alignment-free), sử dụng thêm token đặc biệt `blank` (chỉ số 0).
- **Ý nghĩa trong dự án:** Tầng phân loại đầu ra của CRNN tạo ra ma trận logits `[N, T, C]` (với $C = \text{len}(\text{vocab}) + 1 = 222$).

### CTC Greedy Decode
- **Định nghĩa:** Thuật toán giải mã tham lam cho kết quả CTC: chọn ra chỉ số có xác suất lớn nhất tại mỗi bước thời gian $t$, loại bỏ các ký tự lặp liên tiếp nếu giữa chúng không có token `blank`, sau đó xóa bỏ toàn bộ `blank`.
- **Ý nghĩa trong dự án:** Nằm tại hàm `ctc_greedy_decode` trong `src/docguard/pipeline/processing.py`, chạy trực tiếp trên CPU bằng NumPy.

### CAT-Net (Compression Artifact Tracing Network)
- **Định nghĩa:** Kiến trúc mạng nơ-ron chuyên sâu cho giám định ảnh số (digital image forensics), trích xuất và kết hợp dấu vết nén JPEG không đồng nhất ở miền không gian (RGB) và miền tần số (DCT).
- **Ý nghĩa trong dự án:** Cảm hứng cho mô hình `tamper_seg`. Bản đầy đủ quá nặng và không tương thích ONNX, nên dự án thiết kế phiên bản tinh giản `CAT-Net-lite`.

### CAT-Net-lite
- **Định nghĩa:** Biến thể tinh giản của CAT-Net được tác giả dự án thiết kế để phù hợp môi trường serving công nghiệp.
- **Ý nghĩa trong dự án:** Nhánh RGB dùng backbone MobileNetV3-large pretrained ImageNet; nhánh DCT chuyển ảnh sang kênh độ sáng Y và dùng các bộ lọc tích chập cố định (fixed conv filters) kích thước $8\times8$, stride 8 để tính hệ số DCT mà không cần phụ thuộc thư viện giải mã JPEG bên ngoài.

### DCT (Discrete Cosine Transform)
- **Định nghĩa:** Biến đổi Cosine rời rạc — phép biến đổi toán học chuyển tín hiệu ảnh từ miền không gian sang miền tần số, là nền tảng cốt lõi của chuẩn nén ảnh JPEG trên các khối $8\times8$ pixel.
- **Ý nghĩa trong dự án:** Khi kẻ gian cắt dán (copy-move/splicing), lưới $8\times8$ hoặc phân phối hệ số tần số DCT tại vùng can thiệp sẽ bị lệch pha với phần còn lại của hóa đơn, tạo tín hiệu phát hiện giả mạo.

### SRM (Spatial Rich Model)
- **Định nghĩa:** Tập hợp các bộ lọc tuyến tính bậc cao trích xuất nhiễu và vi sai bề mặt ảnh, thường dùng trong phân tích giấu tin (steganography) và phát hiện can thiệp ảnh.
- **Ý nghĩa trong dự án:** Một nhánh trích xuất đặc trưng tùy chọn (ablation option) trong `tamper_seg` dùng 3 bộ lọc SRM cơ bản.

### SegFormer / SegFormer-B0
- **Định nghĩa:** Kiến trúc mạng biến đổi phân đoạn ngữ nghĩa gọn nhẹ dựa trên Transformer không dùng positional encoding phức tạp.
- **Ý nghĩa trong dự án:** Phương án kiến trúc dự phòng / ablation được cân nhắc so sánh với CAT-Net-lite trong đề xuất đề tài.

### MobileNetV3 (Large / Small)
- **Định nghĩa:** Mạng nơ-ron tích chập nhẹ tối ưu bằng phương pháp tìm kiếm kiến trúc mạng (NAS), sử dụng inverted residual blocks và kích hoạt Hard-Swish.
- **Ý nghĩa trong dự án:** Dùng bản `MobileNetV3-large` làm backbone cho DBNet và nhánh RGB của CAT-Net-lite; dùng `MobileNetV3-small` hoặc `VGG-lite` làm backbone cho CRNN.

### FPN (Feature Pyramid Network)
- **Định nghĩa:** Mạng kim tự tháp đặc trưng kết hợp thông tin ngữ nghĩa cấp cao ở các tầng sâu với thông tin tọa độ không gian sắc nét ở các tầng nông.
- **Ý nghĩa trong dự án:** Bộ phận cổ chai (neck) trong mô hình DBNet và decoder của CAT-Net-lite.

### LR-ASPP (Lite Reduced Atrous Spatial Pyramid Pooling)
- **Định nghĩa:** Khối gộp không gian đa tỉ lệ dạng rút gọn dùng phép tích chập lỗ rỗng (atrous/dilated conv) giúp mở rộng vùng tiếp nhận (receptive field) với chi phí tính toán thấp.
- **Ý nghĩa trong dự án:** Thành phần decoder tiềm năng trong nhánh phân đoạn giả mạo.

### OHEM (Online Hard Example Mining)
- **Định nghĩa:** Kỹ thuật lọc và khai phá mẫu khó trực tuyến trong quá trình huấn luyện: chỉ tính gradient trên các pixel hoặc mẫu có loss cao nhất.
- **Ý nghĩa trong dự án:** Áp dụng trong hàm loss của DBNet (`DBLoss`, theo MMOCR): BCE lấy mọi pixel dương (vùng chữ đã co) cộng với số pixel âm có loss lớn nhất, tối đa bằng $3\times$ số pixel dương, để cân bằng diện tích nền áp đảo so với vùng chữ.

---

## 3. Xử lý Ảnh & Kỹ thuật Sinh Dữ liệu Giả mạo

### Tiling & Stitching (Cắt mẩu & Ghép xác suất)
- **Định nghĩa:** Kỹ thuật chia ảnh lớn thành nhiều mẩu vuông nhỏ gối chồng lên nhau, sau đó ghép các bản đồ xác suất đầu ra lại bằng phép trung bình cộng.
- **Ý nghĩa trong dự án:** Nằm tại `P.tile` và `P.stitch`. Nhánh `tamper_seg` bắt buộc chạy trên patch ở **độ phân giải gốc** (`patch_size: 512, stride: 384, overlap: 128 px`). Không resize toàn bộ ảnh lớn về $512\times512$ vì resize sẽ nội suy pixel, phá hủy toàn bộ dấu vết nén JPEG và artifact biên chỉnh sửa.

### Native Resolution (Độ phân giải gốc)
- **Định nghĩa:** Kích thước điểm ảnh nguyên bản của ảnh chụp từ thiết bị phần cứng trước khi qua bất kỳ phép co dãn tỉ lệ nào.
- **Ý nghĩa trong dự án:** Yêu cầu tiên quyết của bài toán giám định tài liệu số. Mọi thao tác giám định giả mạo phải thực thi trên độ phân giải gốc để giữ nguyên đặc tính vi mô.

### `max_long_side` (Giới hạn cạnh dài)
- **Định nghĩa:** Ngưỡng kích thước điểm ảnh tối đa cho phép của cạnh dài nhất trong ảnh (trong dự án đặt là 2048 px).
- **Ý nghĩa trong dự án:** Ảnh vượt quá 2048 px sẽ buộc phải hạ tỉ lệ (downscale) để tránh tràn VRAM. Hành động này làm tổn hại tín hiệu forensic nên hệ thống ghi cảnh báo log lại.

### Letterbox
- **Định nghĩa:** Phép biến đổi co dãn ảnh giữ nguyên tỉ lệ khung hình (aspect ratio) gốc, sau đó đặt vào một khung vuông chuẩn (canvas) và đệm viền đen/xám vào phần khuyết.
- **Ý nghĩa trong dự án:** Dùng trong tiền xử lý cho Text Detection (`det_preprocess`), đưa ảnh hóa đơn về khung vuông $960\times960$ canh góc trên-trái.

### Shrink / Unclip
- **Định nghĩa:** Khi huấn luyện DBNet, mỗi vùng chữ ground truth được **co** (shrink) vào trong một khoảng $D = \frac{A(1 - r^2)}{L}$, với $A$ là diện tích, $L$ là chu vi và $r$ = `shrink_ratio`. **Unclip** là bước ngược lại ở hậu xử lý: nới vùng chữ dự đoán ra để lấy lại kích thước thật.
- **Ý nghĩa trong dự án:** `shrink_ratio = 0.4`, dùng chung cho `configs/train/text_det.yaml` và `configs/pipeline.yaml`. Từ intent 003, `det_postprocess` unclip **chính xác** bằng nghịch đảo công thức shrink, thay cho công thức xấp xỉ cũ $d = A \cdot \text{unclip\_ratio} / L$ (công thức cũ làm box của dòng chữ dài chỉ đạt IoU khoảng 0,5):
  - kích thước kernel $w' \times h'$ lấy từ `cv2.minAreaRect` của contour, để dòng chữ nghiêng vài độ không làm box bị cao quá;
  - $d$ là nghiệm dương của $(8 - 4k)d^2 + 2(1 - k)(w' + h')d - k\,w'h' = 0$, với $k = 1 - r^2$;
  - hình chữ nhật xoay được nới thêm $d$ mỗi phía; đầu ra là bbox thẳng `[x1, y1, x2, y2]` của nó.

### Inpainting
- **Định nghĩa:** Kỹ thuật toán học / học máy phục chế hoặc xóa một vùng nội dung trên ảnh bằng cách suy diễn màu sắc và cấu trúc từ các pixel biên lân cận.
- **Ý nghĩa trong dự án:** Thuật toán sinh giả mạo số tiền và ngày: dùng thuật toán Telea / Navier-Stokes của OpenCV để xóa sạch con số cũ trên hóa đơn, sau đó vẽ đè chuỗi số mới lên bằng font chữ tương đồng.

### Splicing
- **Định nghĩa:** Hành vi giả mạo cắt ghép nội dung: lấy một đoạn ký tự/con số từ một tài liệu/hóa đơn khác và dán đè vào hóa đơn mục tiêu.
- **Ý nghĩa trong dự án:** Hàm `splicing` trong `synth_tamper.py` lấy box số từ ảnh nhà tài trợ (donor image) ghép sang ảnh đích và ghi lại mask pixel chính xác.

### Copy-Move
- **Định nghĩa:** Hành vi sao chép một cụm ký tự tại một vị trí trong hóa đơn và dán vào một vị trí khác trên chính hóa đơn đó (ví dụ: copy số '9' ở mã số thuế dán vào hàng tổng tiền).
- **Ý nghĩa trong dự án:** Hàm `copy_move` trong `synth_tamper.py`. Cần tối thiểu 2 hộp chứa chữ số trên cùng một ảnh.

### JPEG Re-compression
- **Định nghĩa:** Quá trình nén lại file ảnh JPEG với hệ số chất lượng (Quality Factor - Q) khác hoặc bị dịch chuyển lưới khối $8\times8$.
- **Ý nghĩa trong dự án:** Sau khi cấy ghép pixel giả mạo, ảnh được lưu lại với chất lượng ngẫu nhiên $Q \in [70, 95]$ để mô phỏng chính xác hiện tượng ảnh bị nén khi gửi qua ứng dụng chat/web.

### Otsu Thresholding & Ink Mask
- **Định nghĩa:** Thuật toán phân ngưỡng tự động Otsu để tách biệt mực đen của chữ (foreground) ra khỏi nền giấy trắng (background).
- **Ý nghĩa trong dự án:** Hàm `_ink_mask` trong `synth_tamper.py` dùng Otsu và phép giãn ảnh (dilation) để chỉ tính mask giả mạo trên các pixel có mực in thật sự, tránh đánh dấu nhầm nền giấy.

### Bounding Box (BBox)
- **Định nghĩa:** Hộp chữ nhật giới hạn đối tượng song song trục tọa độ, quy ước trong dự án là danh sách 4 số nguyên `[x1, y1, x2, y2]`.
- **Ý nghĩa trong dự án:** Tọa độ tuyệt đối trên không gian pixel ảnh gốc gửi từ client.

---

## 4. Tối ưu hóa Mô hình, Lượng tử hóa & Runtime

```
PyTorch Model (.pt)
       │
       ▼ (torch.onnx.export - opset 17)
ONNX Model (.onnx, FP32)
       │
       ├─────────────────────────────────► Variant B1 (ONNX Runtime FP32)
       │
       ├─ to_fp16() ─────────────────────► Variant B2 (TensorRT FP16)
       │
       ├─ quantize_int8_static() (QDQ) ──► Variant B3 / B5 (TensorRT INT8)
       │
       └─ torch-pruning (30%) ───────────► Variant B4 (Pruned + TensorRT INT8)
```

### ONNX (Open Neural Network Exchange)
- **Định nghĩa:** Chuẩn định dạng mở đại diện cho đồ thị tính toán của các mô hình học máy, cho phép chuyển đổi liền mạch giữa các framework (PyTorch, TensorFlow, TensorRT).
- **Ý nghĩa trong dự án:** Cầu nối trung gian để chuyển đổi toàn bộ mô hình PyTorch (`.pt`) sang TensorRT.

### ONNX Runtime (ORT)
- **Định nghĩa:** Engine thực thi suy luận mô hình ONNX đa nền tảng tối ưu của Microsoft.
- **Ý nghĩa trong dự án:** Dùng cho biến thể B1 (baseline ORT FP32), kiểm tra đối chiếu sai số (parity check) và thực hiện lượng tử hóa tĩnh QDQ trên CPU.

### TensorRT (TRT)
- **Định nghĩa:** Bộ công cụ phát triển phần mềm (SDK) suy luận học sâu hiệu năng cao của NVIDIA, thực hiện nung ghép lớp (layer fusion), tối ưu bộ nhớ kernel và tận dụng tối đa phần cứng GPU NVIDIA.
- **Ý nghĩa trong dự án:** Engine suy luận cốt lõi trên môi trường production (các biến thể B2–B5).

### `trtexec`
- **Định nghĩa:** Tiện ích dòng lệnh đi kèm TensorRT để biên dịch file ONNX thành file engine thực thi độc quyền (`.plan` hoặc `.engine`) và đo kiểm tốc độ.
- **Ý nghĩa trong dự án:** Lệnh trong `scripts/build_trt.sh` dùng để build engine cho `tamper_seg`, `text_det`, `text_rec`.

### Precision (Độ chính xác biểu diễn số học)
- **FP32 (Single-precision float, 32-bit):** Chuẩn dấu phẩy động 32-bit, dùng làm mốc so sánh gốc (B0, B1).
- **FP16 (Half-precision float, 16-bit):** Chuẩn dấu phẩy động 16-bit bán chính xác. Tận dụng Tensor Cores trên GPU Turing T4, giảm $50\%$ dung lượng VRAM và tăng tốc độ tính toán gấp 2–3 lần với độ suy giảm chính xác gần như bằng 0.
- **INT8 (8-bit Integer):** Biểu diễn số nguyên có dấu 8-bit. Giảm $75\%$ dung lượng tensor so với FP32, mang lại thông lượng suy luận cực đại.

### Quantization (Lượng tử hóa)
- **Định nghĩa:** Quá trình ánh xạ dải giá trị liên tục số thực dấu phẩy động (FP32) sang tập rời rạc số nguyên có dải biểu diễn hẹp hơn (INT8).
- **Ý nghĩa trong dự án:** Giảm mạnh footprint bộ nhớ và độ trễ suy luận của mô hình OCR.

### PTQ (Post-Training Quantization - Lượng tử hóa sau huấn luyện)
- **Định nghĩa:** Kỹ thuật lượng tử hóa mô hình trực tiếp từ checkpoint đã train mà không cần huấn luyện lại từ đầu (khác với QAT - Quantization-Aware Training).
- **Ý nghĩa trong dự án:** Áp dụng cho các mô hình bằng cách chạy qua 500 ảnh hiệu chuẩn (calibration).

### Calibration / Calibration Dataset (Tập dữ liệu hiệu chuẩn)
- **Định nghĩa:** Tập con dữ liệu thực tế (500 ảnh lấy từ tập train) được đưa qua mạng để ghi nhận phân phối histogram giá trị kích hoạt (activations), từ đó tính toán hệ số tỉ lệ (scale factor) và điểm lệch (zero point) cho INT8.
- **Ý nghĩa trong dự án:** Sử dụng phương pháp đo entropy (`CalibrationMethod.Entropy`).

### QDQ (Quantize and DeQuantize)
- **Định nghĩa:** Chuẩn biểu diễn đồ thị lượng tử hóa tường minh trong ONNX, trong đó các node tính toán được kẹp giữa cặp toán tử `QuantizeLinear` (Q) và `DequantizeLinear` (DQ).
- **Ý nghĩa trong dự án:** TensorRT nhận đồ thị QDQ từ ONNX để biên dịch chính xác các khối tính toán INT8 mà không bị sai lệch thuật toán.

### Pruning / Structured Pruning (Cắt tỉa cấu trúc)
- **Định nghĩa:** Kỹ thuật loại bỏ bớt các kênh (channels) hoặc khối tích chập dư thừa trong mạng nơ-ron để giảm kích thước và số phép tính (FLOPs). Khác với un-structured pruning (chỉ đặt trọng số về 0 mà không tăng tốc phần cứng), structured pruning thực sự thu nhỏ ma trận trọng số.
- **Ý nghĩa trong dự án:** Áp dụng cắt tỉa $30\%$ tham số backbone qua thư viện `torch-pruning` trong biến thể B4.

### Parity Check / `max_abs_diff`
- **Định nghĩa:** Phép kiểm thử đối chiếu độ lệch tuyệt đối tối đa giữa tensor đầu ra của mô hình PyTorch gốc và mô hình sau khi xuất sang ONNX/TensorRT:
  $$\Delta_{\max} = \max |y_{\text{pytorch}} - y_{\text{onnx}}|$$
- **Ý nghĩa trong dự án:** Hàm `max_abs_diff` trong `onnx_export.py` đảm bảo việc chuyển đổi định dạng không làm sai lệch hàm toán học của mô hình.

### Opset (ONNX Operator Set)
- **Định nghĩa:** Phiên bản đặc tả tập toán tử chuẩn của ONNX. Dự án chốt sử dụng **opset 17** để hỗ trợ đầy đủ các toán tử BiLSTM và kích hoạt hiện đại.

### Dynamic Axes
- **Định nghĩa:** Khai báo các chiều không gian (thường là batch size hoặc kích thước ảnh) có thể thay đổi linh hoạt tại runtime trong đồ thị ONNX.
- **Ý nghĩa trong dự án:** Cấu hình cho phép các mô hình nhận batch size linh động từ 1 đến batch tối đa trong Triton.

---

## 5. Ma trận Thí nghiệm (B0 – B5)

Ma trận thí nghiệm được thiết kế tại `configs/experiments.yaml` và `docs/proposal.md §6` để trả lời các câu hỏi cốt lõi về sự đánh đổi giữa hiệu năng (latency/VRAM) và độ chính xác (accuracy/forensic quality):

| Mã | Tên Biến thể | Runtime | Độ chính xác (Precision) | Mục đích thí nghiệm & Ý nghĩa |
|---|---|---|---|---|
| **B0** | **Baseline PyTorch** | PyTorch | FP32 | Mốc chuẩn gốc (ground baseline) để đối chứng. FastAPI + PyTorch thuần trên GPU T4. |
| **B1** | **ONNX Runtime** | ONNX Runtime | FP32 | Đo lường mức cải thiện hiệu năng khi chỉ chuyển đổi runtime đồ thị, chưa đổi precision. |
| **B2** | **TensorRT FP16** | TensorRT | FP16 | Đánh giá năng lực tăng tốc của Tensor Cores với số thực bán chính xác. |
| **B3** | **TensorRT INT8** | TensorRT | INT8 (Toàn bộ) | Đo lường mức độ suy giảm của bài toán phát hiện giả mạo khi bị lượng tử hóa sâu. |
| **B4** | **Pruned + INT8** | TensorRT | INT8 + Pruning 30% | Thử nghiệm giới hạn nén tối đa (Ablation study) kết hợp cắt tỉa backbone và INT8. |
| **B5** | **Mixed Precision** | TensorRT | **Tamper: FP16<br>OCR: INT8** | **Cấu hình tối ưu được đề xuất:** Giữ nguyên FP16 cho bài toán giả mạo pixel vi mô; dùng INT8 cho OCR/KIE để đạt throughput tối đa. |

> [!IMPORTANT]
> **Câu hỏi nghiên cứu trọng tâm:** Lượng tử hóa INT8 và phép co dãn ảnh (resize) ảnh hưởng tiêu cực đến Pixel-F1 của bài toán tampering nhiều hơn bao nhiêu so với CER của OCR? Cấu hình so sánh đối đầu giữa **B3** và **B5** sẽ trả lời câu hỏi này.

---

## 6. Tập Dữ liệu, Phân chia & Định dạng

### MC-OCR 2021 (Mobile-Captured Receipt Recognition)
- **Định nghĩa:** Bộ dữ liệu cuộc thi nhận dạng hóa đơn chụp bằng thiết bị di động do nhóm VNDAG tổ chức tại hội nghị quốc tế RIVF 2021.
- **Ý nghĩa trong dự án:** Tập dữ liệu ảnh hóa đơn thực tế làm nguyên liệu đầu vào. Gồm 1.155 ảnh train có gán nhãn 4 trường và 391 ảnh test public không có nhãn.

### VNDAG (Vietnamese Natural Document Analysis Group)
- **Định nghĩa:** Nhóm nghiên cứu phân tích tài liệu tự nhiên tiếng Việt, tác giả phát hành tập dữ liệu MC-OCR 2021.

### RIVF
- **Định nghĩa:** *Research, Innovation and Vision for the Future* — Hội nghị quốc tế uy tín về khoa học máy tính và truyền thông tại Việt Nam.

### DocTamper
- **Định nghĩa:** Bộ dữ liệu quy mô lớn chuẩn quốc tế dành cho bài toán phát hiện giả mạo tài liệu văn bản.
- **Ý nghĩa trong dự án:** Được cân nhắc để pretrain mô hình `tamper_seg` trước khi fine-tune trên hóa đơn tiếng Việt (chỉ dùng cho nghiên cứu).

### Split (`splits.json`)
- **Định nghĩa:** File chỉ mục phân chia tập dữ liệu cố định theo tỷ lệ: **Train (80%) / Validation (10%) / Test (10%)** dựa trên mã `img_id` của ảnh gốc.
- **Ý nghĩa trong dự án:** Tạo ra bởi `scripts/make_splits.py`. Cả hai bài toán Tampering và OCR/KIE đều dùng chung một file split này.

### Data Leakage (Rò rỉ dữ liệu)
- **Định nghĩa:** Sai lầm nghiêm trọng trong học máy khi thông tin từ tập kiểm thử (validation/test) bị rò rỉ vào tập huấn luyện (train).
- **Ý nghĩa trong dự án:** Được ngăn chặn nghiêm ngặt: (1) Mọi biến thể giả mạo sinh từ một ảnh gốc chỉ nằm duy nhất trong split của ảnh đó; (2) Corpus từ ngữ dùng để sinh dữ liệu tổng hợp chỉ được trích xuất từ tập train của MC-OCR.

### Pseudo-label (Gán nhãn giả)
- **Định nghĩa:** Kỹ thuật bán giám sát (semi-supervised): dùng một mô hình đã học (DBNet v1) để suy luận ra nhãn trên tập dữ liệu chưa gán nhãn đầy đủ, sau đó dùng các nhãn tự sinh này để fine-tune mô hình thế hệ tiếp theo (DBNet v2).
- **Ý nghĩa trong dự án:** Vì MC-OCR chỉ gán nhãn 4 trường thông tin chính chứ không gán nhãn tất cả các dòng chữ li ti khác trên hóa đơn, DBNet v1 sẽ sinh pseudo-label cho các dòng còn lại trên tập train.

### Ground Truth (GT)
- **Định nghĩa:** Nhãn chuẩn chân lý đo đạc thực tế dùng làm thước đo đánh giá hoặc huấn luyện.

### JSONL (`.jsonl` - JSON Lines)
- **Định nghĩa:** Định dạng tệp tin văn bản trong đó mỗi dòng là một chuỗi JSON hợp lệ riêng biệt.
- **Ý nghĩa trong dự án:** Định dạng chuẩn hóa lưu trữ annotation (`annotations.jsonl`) chứa kích thước ảnh và danh sách các box chữ, label.

### OFL (SIL Open Font License)
- **Định nghĩa:** Giấy phép bản quyền phần mềm tự do nguồn mở chuyên biệt dành cho các font chữ.
- **Ý nghĩa trong dự án:** Mọi font chữ trong thư mục `data/fonts/` bắt buộc có giấy phép OFL, và file giấy phép được để kèm (`OFL-*.txt`). Hiện gồm Be Vietnam Pro, IBM Plex Mono, Noto Sans Mono, Roboto Mono và Source Code Pro.

### Charset / `vocab_vi.txt`
- **Định nghĩa:** Tập ký tự tiếng Việt gồm đúng **221 ký tự** (chữ cái có dấu, chữ số, ký tự đặc biệt và `<space>`).
- **Ý nghĩa trong dự án:** Từ điển ký tự chuẩn dùng cho tầng giải mã CTC của mô hình `text_rec`.

---

## 7. Chỉ số Đánh giá & Thước đo Hiệu năng

### Pixel-F1
- **Định nghĩa:** Điểm F1-score tính toán trên từng điểm ảnh (pixel) giữa mặt nạ nhị phân dự đoán và mặt nạ ground truth:
  $$\text{Pixel-F1} = \frac{2 \times TP}{2 \times TP + FP + FN}$$
- **Ý nghĩa trong dự án:** Thước đo độ chính xác cốt lõi của nhánh phát hiện giả mạo (`tamper_seg`) với ngưỡng phân loại xác suất $0.5$.

### IoU (Intersection over Union / Jaccard Index)
- **Định nghĩa:** Tỷ số giữa diện tích vùng giao nhau và diện tích vùng hợp nhất giữa dự đoán và ground truth:
  $$\text{IoU} = \frac{|P \cap G|}{|P \cup G|}$$
- **Ý nghĩa trong dự án:** Đánh giá độ phủ khít của mặt nạ giả mạo và các hộp phát hiện chữ.

### ROC-AUC (Receiver Operating Characteristic – Area Under Curve)
- **Định nghĩa:** Diện tích dưới đường cong ROC đo lường năng lực phân loại của mô hình qua mọi ngưỡng quyết định.
- **Ý nghĩa trong dự án:** Đo ở cấp độ ảnh (image-level) để phân loại hóa đơn là "Thật" hay "Giả mạo" dựa trên điểm số `tampered_score`, tính bằng thống kê Mann-Whitney U.

### CER (Character Error Rate)
- **Định nghĩa:** Tỉ lệ lỗi ký tự giữa chuỗi văn bản nhận dạng và chuỗi nhãn chuẩn:
  $$\text{CER} = \frac{\text{Khoảng cách Levenshtein (Số phép Thay thế + Xóa + Chèn)}}{\text{Tổng số ký tự ground truth}}$$
- **Ý nghĩa trong dự án:** Thước đo độ chính xác của mô hình Text Recognition (`text_rec`).

### Levenshtein Distance (Edit Distance)
- **Định nghĩa:** Khoảng cách chỉnh sửa tối thiểu — số lượng thao tác sửa đổi đơn ký tự (chèn, xóa, thay thế) ít nhất để biến chuỗi này thành chuỗi kia.
- **Ý nghĩa trong dự án:** Thuật toán tính CER tại `src/docguard/metrics/ocr.py`.

### Field Exact Match (Field EM)
- **Định nghĩa:** Tỉ lệ trích xuất đúng hoàn toàn ($100\%$) từng trường thông tin sau khi đã qua bước chuẩn hóa văn bản (loại bỏ toàn bộ khoảng trắng, viết hoa chuỗi và chuẩn hóa chuỗi số):
  $$\text{Field EM} = \frac{\text{Số trường khớp chính xác}}{\text{Tổng số trường}}$$
- **Ý nghĩa trong dự án:** Thước đo nghiệp vụ thực tế cho mô hình KIE; khớp hoàn toàn mới tính điểm 1, lệch 1 ký tự tính 0 điểm.

### H-mean (Harmonic Mean / F-measure)
- **Định nghĩa:** Trung bình điều hòa giữa độ chính xác (Precision) và độ thu hồi (Recall):
  $$H\text{-mean} = 2 \times \frac{P \times R}{P + R}$$
- **Ý nghĩa trong dự án:** Dùng để đánh giá chất lượng các hộp chữ nhật phát hiện được của mô hình `text_det` theo ngưỡng IoU $0.5$.

### `tampered_score`
- **Định nghĩa:** Điểm số đại diện cho độ tin cậy giả mạo của bức ảnh, được tính bằng phân vị $99.9$ (`np.percentile(prob, 99.9)`) trên bản đồ xác suất.
- **Ý nghĩa trong dự án:** Sử dụng phân vị $99.9$ thay vì giá trị max tuyệt đối để tránh hiện tượng báo động giả do các điểm ảnh nhiễu cô lập (isolated hot pixels). Ngưỡng kích hoạt gian lận: $> 0.80$.

### `tampered_area_ratio`
- **Định nghĩa:** Tỉ lệ diện tích vùng bị nghi ngờ giả mạo so với toàn bộ diện tích ảnh:
  $$\text{area\_ratio} = \frac{\sum (\text{prob} \ge 0.5)}{\text{Chiều cao} \times \text{Chiều rộng}}$$
- **Ý nghĩa trong dự án:** Ngưỡng kích hoạt gian lận: $> 0.002$ ($0.2\%$ diện tích hóa đơn).

### `overlaps_fields`
- **Định nghĩa:** Danh sách tên các trường thông tin nghiệp vụ quan trọng (`total_amount`, `timestamp`, `seller`) bị hộp giới hạn giả mạo đè lên.
- **Ý nghĩa trong dự án:** Tín hiệu nghiệp vụ cảnh báo quan trọng nhất trong hệ thống giải ngân tài chính: nếu vùng giả mạo rơi đúng vào con số tổng tiền, hóa đơn bị phân loại rủi ro cực kỳ nghiêm trọng.

### Latency Percentiles (Phân vị độ trễ)
- **p50 (Median):** $50\%$ số yêu cầu được phản hồi nhanh hơn thời gian này.
- **p95:** $95\%$ số yêu cầu phản hồi nhanh hơn mức này (chỉ số cam kết SLA).
- **p99:** $99\%$ số yêu cầu phản hồi nhanh hơn mức này (đo lường độ trễ đuôi - tail latency do hàng đợi hoặc nghẽn GC).

### Peak VRAM
- **Định nghĩa:** Dung lượng bộ nhớ RAM đồ họa tối đa bị chiếm dụng trên GPU tại thời điểm tải nặng nhất.
- **Ý nghĩa trong dự án:** Mục tiêu đề ra là TensorRT phải giảm $\ge 40\%$ peak VRAM so với baseline PyTorch FP32.

### Throughput / QPS (Queries Per Second)
- **Định nghĩa:** Số lượng yêu cầu (requests) mà hệ thống xử lý hoàn tất thành công trong một giây dưới một mức đồng thời nhất định.
- **Ý nghĩa trong dự án:** Mục tiêu đạt $\ge 2\times$ so với baseline khi tải 32 người dùng đồng thời.

---

## 8. Nghiệp vụ Hóa đơn & Bóc tách Thông tin

### `seller` (Đơn vị bán hàng)
- **Ý nghĩa:** Tên của siêu thị, cửa hàng hoặc nhà cung cấp dịch vụ in trên hóa đơn.
- **Quy tắc trích xuất (`src/docguard/kie/rules.py`):** Lọc các dòng văn bản nằm ở góc trên cùng ($25\%$ chiều cao ảnh) có chứa ít nhất 3 ký tự chữ cái; chọn dòng có chiều cao hộp chữ lớn nhất và nằm sát trên đỉnh nhất.

### `timestamp` (Ngày giờ giao dịch)
- **Ý nghĩa:** Thời điểm diễn ra giao dịch in trên hóa đơn.
- **Quy tắc trích xuất:** Sử dụng biểu thức chính quy (Regex) quét các định dạng ngày `DD/MM/YYYY`, `DD-MM-YYYY`, `DD.MM.YYYY`, kiểm tra tính hợp lệ logic (ngày 1–31, tháng 1–12, năm 1990–2100).

### `total_amount` / `TOTAL_COST` (Tổng tiền thanh toán)
- **Ý nghĩa:** Số tiền thanh toán cuối cùng của hóa đơn.
- **Quy tắc trích xuất:** Tìm kiếm các từ khóa nhận diện tổng tiền (`tong cong`, `tong tien`, `thanh toan`, `total`), loại trừ các dòng gây nhiễu (`tien khach dua`, `tien thua`, `giam gia`). Hỗ trợ bóc tách giá trị số in cùng dòng hoặc in ở cột số tiền bên phải ngang hàng; chọn ứng viên xuất hiện ở vị trí thấp nhất trên trang hóa đơn.

### `is_authentic` & `decision_reason`
- **Ý nghĩa:** Quyết định nhị phân cuối cùng của hệ thống (`true`: hóa đơn thật; `false`: phát hiện can thiệp/gian lận) kèm chuỗi lý do minh bạch (ví dụ: `"tampered_area_ratio > 0.002"`).

### Luồng giải ngân online (Lending flow)
- **Ý nghĩa nghiệp vụ:** Quy trình phê duyệt khoản vay trực tuyến trong các công ty Fintech/Ngân hàng số dựa trên chứng từ thu nhập hoặc hóa đơn chi tiêu. Dự án đóng vai trò phòng vệ lớp đầu chống hành vi làm giả số tiền và ngày tháng nhằm trục lợi tài chính.

---

## 9. MLOps, Quy trình AI-Native SDLC & Tuân thủ Pháp lý

### AI-native SDLC
- **Định nghĩa:** Quy trình vòng đời phát triển phần mềm định hướng AI (dựa trên *The AI-Native SDLC Playbook* của Anthropic, 08/2026), trong đó trợ lý AI (agent) là thực thể thực thi chính các bước, con người giữ vai trò kiểm soát tại các "cổng" (gates).
- **Ý nghĩa trong dự án:** Tài liệu hóa tại `docs/process/ai-native-sdlc.md`.

### Intent – Spec – Plan
- **Intent (`intent.md`):** Tài liệu mô tả ý định ban đầu: bối cảnh vấn đề, kết quả mong đợi, thành phần bị ảnh hưởng và câu hỏi mở cần giải quyết.
- **Spec (`spec.md`):** Đặc tả kỹ thuật chi tiết: các tiêu chí chấp nhận đo lường được (Acceptance Criteria - AC), thiết kế giao diện I/O, ràng buộc kỹ thuật.
- **Plan (`plan.md`):** Kế hoạch triển khai từng bước: danh sách file cần sửa, thứ tự thực thi, phân tích rủi ro và phương pháp kiểm chứng (proof).

### Human Gate (Cổng phê duyệt của con người)
- **Định nghĩa:** Điểm dừng kiểm soát bắt buộc trong quy trình AI-native SDLC: Agent không được tự ý nhảy cóc sang viết code khi chưa có phê duyệt `plan.md` từ Owner, không được tự merge PR và không được tự đẩy code lên nhánh `main`.

### Proof / Oracle
- **Định nghĩa:** Bằng chứng thực nghiệm khách quan chứng minh tính đúng đắn của thay đổi phần mềm hoặc mô hình (kết quả chạy test, mã MLflow run id, bảng số liệu so sánh so với baseline).

### Test-lock (`.claude/test-lock`)
- **Định nghĩa:** Cơ chế bảo vệ kiểm thử khi sửa lỗi (bug-fixing flow): khi sửa một con bug, test-case tái hiện lỗi phải được viết và commit trước, sau đó file `.claude/test-lock` được kích hoạt để chặn tuyệt đối AI sửa đổi file test nhằm mục đích làm cho test pass một cách giả tạo.

### Guardrail (`.claude/hooks/guard_edits.py`)
- **Định nghĩa:** Hệ thống hook kiểm soát tiền thực thi công cụ (PreToolUse Hook) ngăn chặn tự động:
  - Chặn sửa đổi trực tiếp dữ liệu thô và kết quả: `data/`, `artifacts/`, `mlruns/`, `results/`.
  - Chặn sửa đổi tài liệu mục tiêu chiến lược cốt lõi: `docs/proposal.md`.
  - Chặn sửa cấu hình hook và bảo vệ secret/token/private key.

### Verifier Subagent
- **Định nghĩa:** Trợ lý ảo phụ trách kiểm định độc lập, được kích hoạt trong một ngữ cảnh sạch (fresh context) trước khi mở PR nhằm kiểm tra đối chiếu toàn bộ mã nguồn, tài liệu và test-case mà không bị thiên kiến của agent lập trình.

### MLflow
- **Định nghĩa:** Nền tảng MLOps nguồn mở quản lý vòng đời học máy.
- **Ý nghĩa trong dự án:** Theo dõi thí nghiệm (experiment tracking), lưu trữ siêu tham số, ghi nhận Run ID, log các metric đánh giá B0–B5 và lưu giữ các trọng số mô hình.

### Locust
- **Định nghĩa:** Công cụ kiểm thử tải (load testing) phân tán viết bằng Python mô phỏng hành vi người dùng.
- **Ý nghĩa trong dự án:** File kịch bản `loadtest/locustfile.py` thực hiện bắn tải lên Gateway với các mức concurrency 1, 8, 32, 64 users để đo QPS, p95, p99.

### Docker & Docker Compose
- **Định nghĩa:** Nền tảng ảo hóa đóng gói ứng dụng thành các container độc lập và công cụ điều phối đa dịch vụ.
- **Ý nghĩa trong dự án:** Định nghĩa trong `docker-compose.yml` gồm 3 services liên kết mạng: `triton` (máy chủ suy luận GPU), `gateway` (cổng API) và `mlflow` (máy chủ quản lý thí nghiệm).

### NVIDIA Container Toolkit
- **Định nghĩa:** Gói tiện ích phần mềm cho phép container Docker truy cập trực tiếp vào driver và phần cứng GPU NVIDIA của máy chủ vật lý.

### Prometheus Metrics
- **Định nghĩa:** Hệ thống giám sát và lưu trữ số liệu chuỗi thời gian (time-series metrics).
- **Ý nghĩa trong dự án:** Triton mở cổng 8002 để trích xuất các thông số GPU utilization, inference queue time, latency phục vụ giám sát hệ thống.

### NVIDIA Tesla T4
- **Định nghĩa:** Card đồ họa tính toán chuyên dụng kiến trúc Turing của NVIDIA dành cho trung tâm dữ liệu đám mây (16GB VRAM, hỗ trợ FP16 và INT8 Tensor Cores).
- **Ý nghĩa trong dự án:** Phần cứng GPU tiêu chuẩn được lựa chọn để tiến hành đo đạc toàn bộ ma trận thí nghiệm B0–B5 và benchmark hiệu năng.

### Nghị định 13/2023/NĐ-CP
- **Định nghĩa:** Nghị định của Chính phủ Việt Nam về bảo vệ dữ liệu cá nhân (Personal Data Protection Decree - PDPD).
- **Ý nghĩa trong dự án:** Ràng buộc pháp lý tối cao: nghiêm cấm sử dụng thông tin cá nhân thực tế (số điện thoại thật, căn cước công dân thật, tài khoản ngân hàng thật) trong quá trình huấn luyện và kiểm thử hệ thống.

---

## 10. Bảng Tra Cứu Nhanh Từ Viết Tắt

| Từ viết tắt | Dạng đầy đủ (Tiếng Anh) | Giải nghĩa tiếng Việt tóm tắt |
|---|---|---|
| **AC** | Acceptance Criteria | Tiêu chí chấp nhận tính năng trong đặc tả kỹ thuật |
| **AMP** | Automatic Mixed Precision | Tự động sử dụng độ chính xác hỗn hợp trong huấn luyện |
| **ASPP** | Atrous Spatial Pyramid Pooling | Gộp không gian kim tự tháp lỗ rỗng đa tỉ lệ |
| **AUC** | Area Under the Curve | Diện tích dưới đường cong (thường là ROC-AUC) |
| **BBox** | Bounding Box | Hộp chữ nhật bao quanh đối tượng `[x1, y1, x2, y2]` |
| **BiLSTM** | Bidirectional Long Short-Term Memory | Mạng bộ nhớ ngắn-dài hai chiều tuần tự |
| **BLS** | Business Logic Scripting | Kịch bản điều phối luồng mô hình động trong Triton |
| **CAT-Net** | Compression Artifact Tracing Network | Mạng học sâu truy vết artifact nén JPEG |
| **CCCD** | Căn cước công dân | Giấy tờ tùy thân (nằm trong kế hoạch mở rộng tương lai) |
| **CER** | Character Error Rate | Tỉ lệ lỗi ký tự trong nhận dạng văn bản OCR |
| **CNN** | Convolutional Neural Network | Mạng nơ-ron tích chập |
| **CRNN** | Convolutional Recurrent Neural Network | Mạng kết hợp tích chập và hồi quy nhận dạng chuỗi chữ |
| **CTC** | Connectionist Temporal Classification | Thuật toán phân loại chuỗi thời gian không cần căn nhãn |
| **DBNet** | Real-time Scene Text Detection with Differentiable Binarization | Mạng phát hiện chữ sử dụng nhị phân hóa vi phân được |
| **DCT** | Discrete Cosine Transform | Biến đổi Cosine rời rạc trong chuẩn nén JPEG |
| **EM** | Exact Match | Khớp chính xác tuyệt đối ($100\%$) |
| **FPN** | Feature Pyramid Network | Mạng kim tự tháp đặc trưng đa tỉ lệ |
| **FP16** | 16-bit Floating Point (Half Precision) | Số thực bán chính xác 16-bit |
| **FP32** | 32-bit Floating Point (Single Precision) | Số thực đơn chính xác 32-bit |
| **gRPC** | Google Remote Procedure Call | Giao thức gọi thủ tục từ xa hiệu năng cao trên HTTP/2 |
| **GT** | Ground Truth | Dữ liệu nhãn chân lý thực tế chuẩn |
| **H-mean** | Harmonic Mean | Trung bình điều hòa (F-measure) |
| **INT8** | 8-bit Integer | Số nguyên 8-bit trong lượng tử hóa mô hình |
| **IoU** | Intersection over Union | Tỉ lệ diện tích phần giao trên phần hợp |
| **JSONL** | JSON Lines | Định dạng tệp tin lưu mỗi bản ghi JSON trên một dòng |
| **KIE** | Key Information Extraction | Bóc tách thông tin trọng tâm có cấu trúc từ văn bản |
| **LR-ASPP** | Lite Reduced Atrous Spatial Pyramid Pooling | Khối gộp không gian đa tỉ lệ dạng rút gọn |
| **MC-OCR** | Mobile-Captured Receipt Recognition | Cuộc thi và bộ dữ liệu nhận dạng hóa đơn chụp di động |
| **MLOps** | Machine Learning Operations | Quy trình và công cụ vận hành triển khai học máy |
| **NAS** | Neural Architecture Search | Tự động tìm kiếm cấu trúc mạng nơ-ron |
| **NFC / NFD** | Normalization Form C / D | Chuẩn chuẩn hóa ký tự Unicode tiếng Việt |
| **OFL** | Open Font License | Giấy phép bản quyền mở dành cho font chữ tự do |
| **OHEM** | Online Hard Example Mining | Khai phá mẫu khó trực tuyến khi huấn luyện |
| **ONNX** | Open Neural Network Exchange | Chuẩn mở định dạng đồ thị mạng nơ-ron |
| **ORT** | ONNX Runtime | Trình thực thi mô hình ONNX của Microsoft |
| **p50 / p95 / p99** | 50th / 95th / 99th Percentile | Phân vị độ trễ thứ 50, 95 và 99 |
| **PR** | Pull Request | Yêu cầu nhập nhánh mã nguồn trong Git |
| **PTQ** | Post-Training Quantization | Lượng tử hóa mô hình sau huấn luyện |
| **QDQ** | Quantize and DeQuantize | Mô hình lượng tử hóa tường minh với cặp toán tử Q và DQ |
| **QPS** | Queries Per Second | Số lượng truy vấn hệ thống phục vụ được trong một giây |
| **REST** | Representational State Transfer | Chuẩn kiến trúc giao tiếp web qua các phương thức HTTP |
| **RIVF** | Research, Innovation and Vision for the Future | Hội nghị quốc tế RIVF tại Việt Nam |
| **RNN** | Recurrent Neural Network | Mạng nơ-ron hồi quy |
| **ROC** | Receiver Operating Characteristic | Đường cong đặc trưng hoạt động của bộ thu phân loại |
| **SDLC** | Software Development Life Cycle | Vòng đời phát triển phần mềm |
| **SLA** | Service Level Agreement | Cam kết chất lượng dịch vụ kỹ thuật |
| **SRM** | Spatial Rich Model | Tập bộ lọc trích xuất đặc trưng nhiễu vi mô của ảnh |
| **TRT** | TensorRT | Bộ tăng tốc suy luận GPU hiệu năng cao của NVIDIA |
| **VNDAG** | Vietnamese Natural Document Analysis Group | Nhóm nghiên cứu phân tích tài liệu tự nhiên tiếng Việt |
| **VRAM** | Video Random Access Memory | Bộ nhớ RAM đồ họa chuyên dụng trên GPU |

---

*Tài liệu này được biên soạn và bảo trì để sử dụng xuyên suốt toàn bộ vòng đời phát triển dự án `doc-guard`.*
