import onnx

def check_onnx_input_shapes(model_path):
    """
    Kiểm tra input shapes của ONNX model, xác định static hay dynamic shape.

    Args:
        model_path (str): Đường dẫn tới file ONNX

    Returns:
        dict: Dictionary chứa input name và thông tin shape + dynamic/static
    """
    model = onnx.load(model_path)
    results = {}

    print(f"Model inputs của {model_path}:")
    for input_tensor in model.graph.input:
        name = input_tensor.name
        shape_proto = input_tensor.type.tensor_type.shape
        dims = []
        is_dynamic = False

        for d in shape_proto.dim:
            if d.HasField("dim_param") and d.dim_param != "":
                dims.append(d.dim_param)
                is_dynamic = True
            elif d.HasField("dim_value"):
                dims.append(d.dim_value)
            else:
                dims.append("?")
                is_dynamic = True

        print(f"- {name}: {dims} --> {'DYNAMIC' if is_dynamic else 'STATIC'}")
        results[name] = {"shape": dims, "dynamic": is_dynamic}

    return results

# Ví dụ sử dụng:
check_onnx_input_shapes("yolov5s.onnx")
