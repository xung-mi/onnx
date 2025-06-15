import onnx
from onnx import helper

# load mô hình
model_path = "resnet18.onnx"
model = onnx.load(model_path)

# in thử danh sách các node name
print("List node:")
for node in model.graph.node:
    print(f"Node: {node.name} - outputs: {node.output}")

# giả sử bạn xác định tensor name là 'Conv_3_Output_0'
target_tensor_name = '/layer4/layer4.0/relu_1/Relu'

# tạo ValueInfo mới cho tensor output trung gian
intermediate_layer_value_info = helper.ValueInfoProto()
intermediate_layer_value_info.name = target_tensor_name

# thêm vào phần output của graph
model.graph.output.append(intermediate_layer_value_info)

# lưu lại model mới
modified_model_path = "model_with_intermediate.onnx"
onnx.save(model, modified_model_path)

print(f"Saved modified model to {modified_model_path}")

