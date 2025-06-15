from onnxruntime_tools import optimizer

optimized_model = optimizer.optimize_model("resnet18.onnx", model_type='bert', num_heads=12, hidden_size=768)
optimized_model.save_model_to_file("model_optimized.onnx")