# Giai đoạn 1: Làm quen với ONNX (Cơ bản)
## Extension để đọc file onnx trên vscode

```
ONNX Graph Viewer
```
## Mục tiêu
1. Hiểu ONNX là gì, dùng để làm gì
    - ONNX (Open Neural Network Exchange) là một chuẩn định dạng mở giúp mô hình học máy có thể trao đổi giữa các framework như PyTorch, TensorFlow, scikit-learn, v.v.
    - Giúp tách biệt quá trình huấn luyện và triển khai, từ đó dễ dàng deploy trên nhiều môi trường như máy chủ, thiết bị edge, mobile.
    - Hỗ trợ các kỹ thuật tối ưu hóa mô hình, quantization, và tăng tốc inference thông qua ONNX Runtime.
2. Biết cách chuyển mô hình từ các framework phổ biến sang ONNX
    - PyTorch → ONNX: dùng torch.onnx.export
    - TensorFlow → ONNX: dùng tf2onnx.convert
    - Kiểm tra cấu trúc mạng và tính tương thích đầu vào/ra bằng Netron hoặc onnx.checker
## Tổng quan ONNX
1. Lịch sử ra đời của ONNX
    - Trước đây, các framework như PyTorch, TensorFlow, Caffe2, MXNet đều có định dạng mô hình riêng, không tương thích với nhau. Điều này gây khó khăn khi:
        - Huấn luyện mô hình trong một framework nhưng muốn triển khai ở môi trường khác.
        - Tối ưu hóa mô hình cho các nền tảng phần cứng khác nhau.
    - ONNX (Open Neural Network Exchange) được phát triển lần đầu tiên vào cuối năm 2017 bởi Microsoft và Facebook AI Research.
    - Mục tiêu: Chuẩn hóa định dạng trao đổi mô hình giữa các framework machine learning và deep learning.
2. Ứng dụng của ONNX trong thực tế
    - Trao đổi mô hình giữa các framework
        - Ví dụ: Huấn luyện mô hình bằng PyTorch, sau đó convert sang ONNX để:
            - Triển khai trên TensorFlow Serving
            - Hoặc dùng trong ứng dụng mobile qua ONNX.js
    - Tối ưu hóa tốc độ và hiệu suất
        - ONNX Runtime giúp tăng tốc độ inference:
            - Tích hợp với TensorRT, OpenVINO, DirectML...
            - Tốc độ cao hơn so với việc chạy mô hình gốc
    - Triển khai mô hình trên nhiều nền tảng:
        - Server (Cloud), Edge devices (Jetson Nano, Raspberry Pi), Mobile (Android/iOS), Web (via ONNX.js)
        - Đặc biệt hiệu quả khi triển khai trong các hệ thống yêu cầu real-time inference.
    - Chuẩn hóa quy trình MLOps:
        - ONNX tạo nên định dạng chung, giúp quy trình CI/CD cho AI model trở nên nhất quán và tự động hóa dễ dàng.
3.  Định dạng file .onnx
    - Cấu trúc của file .onnx:
        - Là một file nhị phân (binary) hoặc dạng văn bản protobuf biểu diễn kiến trúc và trọng số của mô hình.
        - Mô hình ONNX bao gồm:
            - graph: biểu diễn toàn bộ kiến trúc mạng (các node là toán tử)
            - initializer: các trọng số (weight, bias...)
            - input, output: định nghĩa tensor đầu vào/ra
            - metadata_props: thông tin mô tả thêm
    -  Ưu điểm định dạng .onnx:
        - Có thể import/export từ nhiều framework
        - Phân tích bằng Netron để kiểm tra kiến trúc mạng
        - Gọn nhẹ, dễ tích hợp vào workflow MLOps
        - Hỗ trợ quantization, graph optimization, fusion
    - Ví dụ: 
        - Khi dùng torch.onnx.export(...), thu được file resnet18.onnx:
            - Được lưu dưới định dạng protobuf
            - Có thể mở bằng Netron hoặc onnx.load()
## ONNX Runtime: Tăng tốc suy luận mô hình AI
1. ONNX Runtime là gì?
    - ONNX Runtime là một thư viện thực thi mô hình học máy (inference engine) được phát triển bởi Microsoft. 
    - Mục tiêu : cung cấp một engine hiệu suất cao, linh hoạt, tương thích với các mô hình được export ở định dạng ONNX.
2. Vai trò chính
    - Tăng tốc inference : giúp mô hình chạy nhanh hơn trên CPU/GPU
    - Tương thích đa nền tảng : 	Chạy trên server, edge, mobile, và cloud
    - Tích hợp backend tối ưu hóa : Hỗ trợ nhiều backend như TensorRT, OpenVINO, DirectML
    - API đa ngôn ngữ
    - Hỗ trợ training : ONNX Runtime Training hỗ trợ forward + backward (cho một số use case)
3. Kiến trúc ONNX Runtime
    - ONNX Runtime hoạt động như một interpreter cho mô hình ONNX:
    ```bash
        ONNX Model (.onnx)
                ↓
        Inference Session (ONNX Runtime)
                ↓
        Execution Provider (CPU / CUDA / TensorRT / OpenVINO / etc.)
                ↓
        Kết quả output
Các Execution Provider chính:

        | Backend                       | Mô tả                                         |
        | ----------------------------- | --------------------------------------------- |
        | **CPUExecutionProvider**      | Mặc định, không cần cài đặt thêm              |
        | **CUDAExecutionProvider**     | Dùng NVIDIA GPU (yêu cầu cài CUDA, cuDNN)     |
        | **TensorRTExecutionProvider** | Tăng tốc sâu hơn trên NVIDIA GPU qua TensorRT |
        | **OpenVINOExecutionProvider** | Tối ưu trên CPU/GPU/FPGA Intel                |
        | **DirectMLExecutionProvider** | Cho GPU trên Windows (DirectX)                |
        | **CoreMLExecutionProvider**   | Cho iOS/macOS (Apple Silicon)                 |
        | **ACL / NNAPI**               | Hỗ trợ thiết bị ARM (Android)                 |
        
4. Tốc độ và hiệu năng
    - ONNX Runtime thường nhanh hơn 2–10 lần so với PyTorch hoặc TensorFlow trong môi trường inference.
    - Đặc biệt hiệu quả khi kết hợp:
        - TensorRT (tối ưu kernel GPU)
        - Quantization (giảm bit-width)
        - Graph optimization (fuse conv + BN, loại node không cần thiết)
5. Các tính năng nâng cao

        | Tính năng             | Mô tả                                              |
        | --------------------- | -------------------------------------------------- |
        | **Dynamic shapes**    | Hỗ trợ input có kích thước thay đổi                |
        | **Quantization**      | Giảm kích thước và tăng tốc độ                     |
        | **Custom operator**   | Cho phép định nghĩa toán tử đặc biệt nếu cần       |
        | **Profiling**         | Theo dõi hiệu suất từng node inference             |
        | **TensorRT Subgraph** | Tự động tối ưu các phần mạng khả thi bằng TensorRT |

## ONNX Opset	
1. Định nghĩa
    - Opset (Operator Set Version) là phiên bản của tập hợp các operator  mà ONNX hỗ trợ
    - Operator là các khối xây dựng cơ bản của mô hình như Conv, Relu, Gemm, Add, BatchNorm,...
    - Mỗi khi ONNX phát triển thêm hoặc sửa đổi cách hoạt động của toán tử, họ phát hành một phiên bản opset mới.
    - Giữa mô hình (ONNX model) và inference engine (ONNX Runtime) phải cùng hiểu chung một phiên bản opset.
2. Vì sao cần?
    - Cần đảm bảo tương thích giữa runtime và opset
    - Opset mới hỗ trợ nhiều operator mới
    - Runtime hỗ trợ nhiều opset cũ
    - Nếu không chỉ định opset_version, Pytorch mặc định dùng latest

3. Ghi nhớ

```
PyTorch Model  -->  Export to ONNX (opset_version=X)  -->  ONNX Runtime (phải hỗ trợ opset X)
```
