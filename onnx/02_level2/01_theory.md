hiểu cách onnx hỗ trợ tối ưu mô hình phục vụ triển khai thực tế.

biết cách giảm độ trễ (latency), giảm kích thước mô hình và tăng hiệu quả inference.

áp dụng onnx để phục vụ các sản phẩm ai chạy thực tế, cả trên server lẫn thiết bị edge.

## lý thuyết chung

* mô hình sau huấn luyện thường được biểu diễn dưới dạng đồ thị (graph):

  * node: đại diện cho 1 operator như: matmul, conv2d, relu, ...
  * edge: đại diện cho tensor truyền giữa các node.
* graph optimization là quá trình biến đổi graph để:

  * tăng tốc độ tính toán.
  * giảm chi phí bộ nhớ.
  * giảm chi phí tính toán.
  * vẫn đảm bảo kết quả đầu ra.
* 2 kỹ thuật chính:

  * node elimination: cắt bỏ các node không cần thiết:

    * constant folding: ví dụ: `y = (3 + 5) * x → y = 8 * x`.
    * identity removal: loại bỏ các phép toán vô nghĩa như:

      * nhân với 1: `y = x * 1 → y = x`.
      * cộng với 0: `y = x + 0 → y = x`.
      * reshape giữ nguyên shape cũ: `reshape(x, original_shape_of_x) → x`.
    * dead node elimination: loại bỏ các node mà output của nó không được sử dụng ở downstream.
  * operator fusion:

    * gộp nhiều operator liên tiếp thành 1 operator duy nhất để:

      * giảm số lượng kernel launch.
      * giảm overhead đọc ghi bộ nhớ trung gian giữa các bước.
    * ví dụ điển hình:

      * convolution + bias\_add + activation fusion.
      * elementwise fusion:

        * các phép toán như: `+`, `-`, `*`, `/`, `relu`, `gelu`, `sigmoid`, `tanh`, `log`, `exp`, `clip`, `sqrt`, `abs`, ... và add bias.
        * ví dụ: `y = relu(x * 1.5 + 0.2)`.

          * `x * 1.5` là elementwise.
          * cộng `+ 0.2` là elementwise.
          * `relu` cũng là elementwise.
          * nếu không fuse, framework thực hiện 3 phép toán liên tiếp, mỗi bước cần tạo buffer trung gian, đọc ghi nhiều lần giữa cpu/gpu memory → gọi nhiều kernel.
    * cách hoạt động:

      * compiler hoặc inference engine sẽ phân tích chuỗi phép toán elementwise liên tiếp rồi gộp lại thành một kernel duy nhất:

        * `y[i] = relu(x[i] * 1.5 + 0.2)`.
    * hạn chế:

      * dependency graph:

        * fusion chỉ xảy ra với các toán tử độc lập trên từng phần tử.
        * nếu có rẽ nhánh (branch), điều kiện (if), loop nội bộ → fusion engine sẽ bỏ qua.
      * hardware limitation:

        * phần cứng cũ hoặc driver thiếu tối ưu hóa kernel fusion.
        * khi số lượng toán tử quá dài, compiler có thể tự cắt bớt fusion.
      * datatype:

        * các phép toán cần có cùng kiểu dữ liệu.
        * nếu có type-casting liên tục → fusion có thể bị giới hạn.
        * khi mixed precision (fp16 + fp32) hoạt động → đôi khi phải tách kernel.

## công cụ chính

1. onnxruntime: chỉ thực hiện inference:

   * load model vào bộ nhớ.
   * thực hiện các phép toán theo graph đã mô tả trong onnx.
   * trả về output.
2. onnxruntime\_tools:

   * hoạt động trước khi inference: làm sạch và tinh chỉnh model onnx trước, sau đó mới nạp vào onnxruntime.
   * hỗ trợ công cụ tối ưu hóa: tinh chỉnh graph để chạy nhanh hơn.
   * chuyển đổi định dạng, ví dụ từ float32 về float16.
   * benchmark: đo hiệu năng model sau khi tối ưu để so sánh.

## các tối ưu phổ biến

* constant folding.
* operator fusion (ví dụ: conv + bn fusion).
* eliminate identity node.

## quantization (giảm kích thước mô hình)

* quantization là quá trình chuyển đổi các số thực (float32 hoặc float16) thành các số nguyên (int8, int16, uint8, ...).
* mục tiêu:

  * tăng tốc tính toán.
  * giảm kích thước mô hình.
  * tiết kiệm tài nguyên hệ thống.
* sau khi huấn luyện mô hình, các weight, bias thường được lưu ở định dạng 32-bit → chính xác nhưng tốn nhiều bộ nhớ, chậm khi inference.
* static quantization:

  * chuyển trọng số về int8, fix sẵn khi chuyển đổi.
  * cần dữ liệu calibration.
* dynamic quantization:

  * lượng tử hóa động ở bước inference.
  * không cần dữ liệu calibration.
* công cụ sử dụng: `onnxruntime.quantization`.
* lợi ích:

  * giảm kích thước file (từ MB → KB).
  * giảm latency.
  * tiết kiệm RAM.
  * phù hợp cho edge device.

## dynamic & static input shapes

* xử lý input có kích thước động (dynamic shapes) giúp linh hoạt hơn khi inference với nhiều kích thước input khác nhau.
* khi cần tối ưu tốc độ, có thể fix static shape (rất phù hợp cho production).

## chạy onnx trên cpu/gpu
* execution provider 
    - là backend thực hiện các toán tử trong graph. Mỗi EP hỗ trợ các phần cứng khác nhau
    * cách chọn execution providers: 
        - `CPUExecutionProvider`
            - phần cứng chạy: CPU (x86, ARM…)
            - đặc điểm:
                - mặc định
                - hỗ trợ full opset.
                - hoạt động ổn định, độ chính xác cao.
                - không yêu cầu driver, cài đặt thêm.
            - khi nào nên dùng:
                - khi inference trên server không có GPU.
                - khi mô hình nhỏ, latency thấp (dưới vài ms).
        - `CUDAExecutionProvider`
            - phần cứng chạy: NVIDIA GPU (compute capability >= 5.0 thường được hỗ trợ tốt).
            - Đặc điểm:
                - hỗ trợ phần lớn opset ONNX.
                - tốc độ inference nhanh hơn CPU với mô hình vừa đến lớn.
                - yêu cầu: cài CUDA toolkit và cuDNN phù hợp với ONNX Runtime bản đang dùng.
            - khi nào nên dùng:
                - khi có GPU.
                - khi mô hình lớn: CNN, Transformer, BERT, ViT...
                - khi latency và throughput là quan trọng.
        - `TensorRTExecutionProvider`
            - phần cứng chạy: NVIDIA GPU, nhưng tận dụng TensorRT SDK để tối ưu inference.
            - đặc điểm:
                - tối ưu cực mạnh với các model CNN, Transformer.
                - hỗ trợ operator fusion, precision calibration (FP16, INT8).
                - cần export ONNX tương thích (vì TensorRT không hỗ trợ 100% opset ONNX).
            - khi nào nên dùng:
                - yêu cầu latency cực thấp (real-time inference).
                - production system trên NVIDIA server.
                - mô hình hỗ trợ TensorRT conversion tốt.
            - lưu ý:
                - việc convert sang TensorRT có thể mất nhiều thời gian compile engine.
                - cần kiểm tra kỹ compatibility giữa ONNX model và TensorRT version.
                - TensorRT EP có thể dùng kết hợp fallback sang CUDA hoặc CPU khi cần.

* tối ưu cấu hình session options:
    - khi khởi tạo onnx runtime session, có thể điều chỉnh nhiều thông số để tối ưu hiệu suất.
    
        | option                       | giải thích                                                                     | khi nào nên điều chỉnh                         |
        | ---------------------------- | ------------------------------------------------------------------------------ | ---------------------------------------------- |
        | `intra_op_num_threads`       | số thread dùng trong 1 toán tử (parallelism nội bộ)                            | tăng nếu CPU nhiều core                        |
        | `inter_op_num_threads`       | số toán tử có thể chạy song song                                               | dùng với các pipeline graph phức tạp           |
        | `execution_mode`             | `ORT_SEQUENTIAL` hoặc `ORT_PARALLEL`                                           | dùng `PARALLEL` khi graph có thể phân nhánh    |
        | `graph_optimization_level`   | `ORT_DISABLE_ALL`, `ORT_ENABLE_BASIC`, `ORT_ENABLE_EXTENDED`, `ORT_ENABLE_ALL` | luôn nên để `ORT_ENABLE_ALL` để bật mọi tối ưu |
        | `enable_mem_pattern`         | bật pattern reuse bộ nhớ                                                       | nên bật (mặc định là True)                     |
        | `log_severity_level`         | điều chỉnh mức log                                                             | dùng để debug                                  |
        | `enable_profiling`           | bật profiling inference                                                        | hữu ích để đo performance                      |
        | `add_session_config_entry()` | thêm các custom config (cho EP đặc biệt như TensorRT)                          | tùy trường hợp                                 |

    - lưu ý nâng cao:
        - với TensorRT EP: nên bật engine caching (trt_engine_cache_enable) để tránh compile lại engine.
        - với CPU EP: điều chỉnh số thread phù hợp với số physical cores.
        - với CUDA EP: nên kiểm tra memory allocation size (bằng các env variable của CUDA).

## trích xuất output từ intermediate layer
- Mục đích:
    - debug mô hình (xem internal activations)
    - kiểm tra giá trị feature maps
    - thực hiện multi-stage inference (cắt model làm nhiều phần)
    - phục vụ các bài toán explainability
- với ONNX Runtime, có 2 phương pháp phổ biến:
    1. sửa đổi ONNX graph (thêm output vào graph)
        - nguyên lý:
            - các node trung gian không được khai báo output mặc định.
            - ONNX Runtime chỉ trả về output nodes (là output ban đầu của model).
            - ta có thể sửa ONNX model để thêm các node trung gian vào phần outputs.
    2. dùng ONNX Runtime session.run với output_names tùy chỉnh 
    3. dùng debug mode ONNX Runtime (ít phổ biến)

* cách lấy giá trị từ các layer ẩn khi debug hoặc xây pipeline multi-stage.

## thực hành

* chuyển mô hình segmentation (deeplabv3, unet, hoặc bisenet) sang onnx.
* viết script benchmark so sánh tốc độ inference giữa:

  * pytorch (native).
  * onnx (default runtime).
  * onnx optimized + quantization.
* áp dụng quantization với công cụ:

  * `onnxruntime.quantization.quantize_dynamic()`.
  * `onnxruntime.quantization.quantize_static()`.

📁 dự án vừa:

web api nhận diện ảnh dùng onnx runtime:

* flask/fastapi backend.
* tích hợp mô hình onnx đã chuyển đổi.
* cho phép upload ảnh và trả về kết quả phân loại hoặc nhận diện.
