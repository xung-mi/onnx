import onnxruntime as ort
import numpy as np

# 1️⃣ Load mô hình ONNX
session = ort.InferenceSession("resnet18.onnx")

# 2️⃣ Lấy tên input tensor đầu vào
input_name = session.get_inputs()[0].name

# 3️⃣ Chuẩn bị dữ liệu input (dummy data kích thước chuẩn)
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)

# 4️⃣ Chạy inference
outputs = session.run(None, {input_name: input_data})

# 5️⃣ In ra shape của output
print("Output shape:", outputs[0].shape)
