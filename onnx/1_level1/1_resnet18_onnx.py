import torch
import torchvision.models as models

model = models.resnet18(pretrained=True)
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model, dummy_input, "resnet18.onnx",
    input_names=['input'],
    output_names=['output'],
    opset_version=11
)
