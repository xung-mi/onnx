import onnxruntime as ort
import numpy as np

# Chuẩn bị input giả lập (bạn thay bằng input thực nếu cần)
input_tensor = np.random.rand(1, 3, 224, 224).astype(np.float32)

# Đọc và chạy model gốc
session_orig = ort.InferenceSession("resnet18.onnx")
input_name = session_orig.get_inputs()[0].name
outputs_orig = session_orig.run(None, {input_name: input_tensor})

print("Model gốc outputs:")
for i, out in enumerate(outputs_orig):
    print(f"  Output {i}: shape {out.shape}")

# Đọc và chạy model đã thêm intermediate output
session_mod = ort.InferenceSession("model_with_intermediate.onnx")
outputs_mod = session_mod.run(None, {input_name: input_tensor})

print("\nModel đã thêm intermediate outputs:")
for i, out in enumerate(outputs_mod):
    print(f"  Output {i}: shape {out.shape}")
