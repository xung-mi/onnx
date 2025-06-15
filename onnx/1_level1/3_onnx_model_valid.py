import onnx

model = onnx.load("resnet18.onnx")
onnx.checker.check_model(model)
print("Model hợp lệ!")
