# tổng quan về tensorrt

# TensorRT là gì

- TensorRT là một SDK tăng tốc inference deep learning của NVIDIA
- Nó giúp chạy mô hình **nhanh hơn nhiều lần** bằng cách:
    - Tối ưu graph.
    - Giảm độ chính xác hợp lý (quantization).
    - Sử dụng hiệu quả phần cứng GPU (Tensor Core, CUDA).
- Hỗ trợ đa dạng backend: CNN, RNN, Transformer, LLM,…
- TensorRT hoạt động chủ yếu sau bước huấn luyện (**post-training optimization**).

# Các khái niệm

## Inference Engine

- Là bộ máy chính thực hiện việc chạy mô hình sau khi đã được tối ưu.
- TensorRT Engine = mô hình đã optimize + serialize thành binary.
- Engine là output cuối cùng của quá trình build, chạy cực nhanh, không cần load model nặng như ONNX hay PyTorch.

## Optimization Profiles

- Dùng trong trường hợp input có kích thước linh hoạt (dynamic shape).
- Cho phép TensorRT tối ưu cho các profile input cụ thể
- Mỗi profile xác định:
    - Min shape.
    - Opt shape (ưu tiên tối ưu).
    - Max shape.

## **Precision Modes**

| Precision | Ý nghĩa | Nhanh cỡ nào | Ghi chú |
| --- | --- | --- | --- |
| FP32 | Float 32-bit | Cơ bản | Chính xác, chậm nhất |
| FP16 | Float 16-bit | 2x nhanh | Giảm nhẹ chính xác, tối ưu cho Tensor Core |
| INT8 | Integer 8-bit | 4-8x nhanh | Cần calibration dữ liệu chuẩn |
- TensorRT hỗ trợ mixed-precision: một số layer FP16, một số FP32 để cân bằng accuracy/speed.

## Calibrator (INT8 Calibration)

- Khi dùng INT8 cần dữ liệu mẫu để:
    - Tính toán thống kê min/max activation.
    - Sinh ra scale, zero-point cho từng layer.
- TensorRT cung cấp các calibrator sẵn (Entropy, MinMax, Percentile).
- Dùng vài trăm/thousand sample đại diện là đủ.

# Cách TensorRT hoạt động

- TensorRT có thể chia thành 4 bước lớn:
    1. Parsing
    2. Optimization
    3. Engine Building
    4. Execution
- hiểu rõ về cách tensorrt hoạt động: parsing, optimization, engine building, execution.