import torch
import torchvision
import onnx

# chuẩn bị model : tải pretrained deeplabv3
model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
model.eval()

# chuẩn bị input dummy
dummy_input = torch.randn(1, 3, 512, 512)  # batch size 1, RGB image 512x512

# export sang onnx
torch.onnx.export(
    model,                           # mô hình cần export
    dummy_input,                     # input mẫu
    "deeplabv3.onnx",                # file onnx đầu ra
    opset_version=12,                # opset version (tương thích inference engine)
    input_names=["input"],           # tên input
    output_names=["output"],         # tên output
    dynamic_axes={"input": {0: "batch_size", 2: "height", 3: "width"},
                  "output": {0: "batch_size", 2: "height", 3: "width"}}  # support dynamic shape
)

# kiểm tra sau export
model = onnx.load("deeplabv3.onnx")
onnx.checker.check_model(model)
print("ONNX model is valid!")

