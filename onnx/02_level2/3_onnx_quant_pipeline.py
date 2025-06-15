import torch
import torch.nn as nn
import torch.nn.functional as F
import subprocess
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
import onnxruntime as ort
import numpy as np
import os

# ==== 1. Định nghĩa mô hình có node dư thừa ====
class RedundantModel(nn.Module):
    def __init__(self):
        super(RedundantModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(32 * 8 * 8, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        
        # Identity (redundant node)
        x = torch.clone(x)
        
        # Transpose (redundant reshape)
        x = x.permute(0, 1, 3, 2)
        x = x.permute(0, 1, 3, 2)
        
        x = self.conv2(x)
        
        # Cast (simulated redundant op)
        x = x.float()
        
        # Reshape (redundant)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# ==== 2. Export ONNX ====
def export_onnx():
    model = RedundantModel()
    model.eval()
    dummy_input = torch.randn(1, 3, 8, 8)
    torch.onnx.export(model, dummy_input, "redundant_model.onnx",
                      input_names=["input"], output_names=["output"],
                      opset_version=13, dynamic_axes={"input": {0: "batch_size"}})
    print("✅ Export ONNX xong!")

# ==== 3. Simplify ONNX ====
def simplify_onnx():
    cmd = ["python3", "-m", "onnxsim", "redundant_model.onnx", "redundant_model_simplified.onnx"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ Simplify thành công!")
    else:
        print("❌ Simplify lỗi:", result.stderr)

# ==== 4. Quantize ONNX ====
def quantize_onnx():
    quantize_dynamic("redundant_model_simplified.onnx",
                     "redundant_model_int8.onnx",
                     weight_type=QuantType.QInt8)
    print("✅ Quantize INT8 xong!")

# ==== 5. Benchmark Inference ====
def benchmark_inference():
    ort_sess = ort.InferenceSession("redundant_model_int8.onnx")
    dummy_input = np.random.randn(1, 3, 8, 8).astype(np.float32)

    import time
    N = 100
    start = time.time()
    for _ in range(N):
        _ = ort_sess.run(None, {"input": dummy_input})
    end = time.time()

    print(f"✅ Benchmark: {N} lượt inference mất {end-start:.3f} giây.")
    print(f"✅ Tốc độ trung bình: {(end-start)/N*1000:.3f} ms/lượt")

# ==== 6. Toàn bộ pipeline ====
def main():
    export_onnx()
    simplify_onnx()
    quantize_onnx()
    benchmark_inference()

if __name__ == "__main__":
    main()
