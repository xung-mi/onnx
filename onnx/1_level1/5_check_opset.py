import onnx

model = onnx.load("resnet18.onnx")
print("Opset version:", model.opset_import[0].version)