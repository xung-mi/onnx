import onnx
from collections import Counter, defaultdict
import numpy as np

def analyze_redundant_nodes(model_path):
    """
    Phân tích và phát hiện các node dư thừa trong mô hình ONNX
    """
    model = onnx.load(model_path)
    
    print("=== PHÂN TÍCH NODE DƯ THỪA TRONG ONNX ===\n")
    
    # 1. Thống kê tổng quan các loại node
    print("1. THỐNG KÊ TỔNG QUAN:")
    node_types = [node.op_type for node in model.graph.node]
    node_count = Counter(node_types)
    
    for node_type, count in sorted(node_count.items()):
        print(f"   {node_type}: {count} node(s)")
    
    print(f"\nTổng số node: {len(model.graph.node)}")
    print(f"Số loại node khác nhau: {len(node_count)}")
    
    # 2. Phát hiện các node có thể dư thừa
    print("\n2. PHÁT HIỆN NODE DƯ THỪA:")
    
    # 2.1 Tìm các node Identity dư thừa
    identity_nodes = [node for node in model.graph.node if node.op_type == 'Identity']
    if identity_nodes:
        print(f"\n   Identity nodes (có thể dư thừa): {len(identity_nodes)}")
        for i, node in enumerate(identity_nodes):
            print(f"     - {node.name or f'identity_{i}'}: {node.input[0]} -> {node.output[0]}")
    
    # 2.2 Tìm các node Reshape không cần thiết
    reshape_nodes = [node for node in model.graph.node if node.op_type == 'Reshape']
    redundant_reshapes = []
    for node in reshape_nodes:
        # Kiểm tra nếu input và output có cùng shape (cần thêm logic phức tạp hơn)
        if len(node.input) >= 2:
            redundant_reshapes.append(node)
    
    if redundant_reshapes:
        print(f"\n   Reshape nodes (có thể dư thừa): {len(redundant_reshapes)}")
        for node in redundant_reshapes:
            print(f"     - {node.name or 'unnamed'}: {node.input[0]} -> {node.output[0]}")
    
    # 2.3 Tìm các node Cast không cần thiết
    cast_nodes = [node for node in model.graph.node if node.op_type == 'Cast']
    if cast_nodes:
        print(f"\n   Cast nodes (kiểm tra xem có cần thiết): {len(cast_nodes)}")
        for node in cast_nodes:
            # Lấy thông tin về kiểu dữ liệu từ attributes
            to_type = None
            for attr in node.attribute:
                if attr.name == 'to':
                    to_type = attr.i
            print(f"     - {node.name or 'unnamed'}: {node.input[0]} -> type {to_type}")
    
    # 2.4 Tìm các node Transpose liên tiếp
    transpose_nodes = [node for node in model.graph.node if node.op_type == 'Transpose']
    if len(transpose_nodes) > 1:
        print(f"\n   Transpose nodes (kiểm tra chuỗi liên tiếp): {len(transpose_nodes)}")
        for node in transpose_nodes:
            perm = None
            for attr in node.attribute:
                if attr.name == 'perm':
                    perm = list(attr.ints)
            print(f"     - {node.name or 'unnamed'}: perm={perm}")
    
    # 3. Phân tích cấu trúc đồ thị
    print("\n3. PHÂN TÍCH CẤU TRÚC ĐỒ THỊ:")
    
    # Tạo mapping input -> output
    input_to_nodes = defaultdict(list)
    output_to_node = {}
    
    for node in model.graph.node:
        for inp in node.input:
            input_to_nodes[inp].append(node)
        for out in node.output:
            output_to_node[out] = node
    
    # Tìm các output không được sử dụng
    unused_outputs = []
    model_outputs = [output.name for output in model.graph.output]
    
    for node in model.graph.node:
        for output in node.output:
            if output not in model_outputs and output not in input_to_nodes:
                unused_outputs.append((node.name or 'unnamed', output))
    
    if unused_outputs:
        print(f"\n   Outputs không được sử dụng: {len(unused_outputs)}")
        for node_name, output in unused_outputs:
            print(f"     - Node '{node_name}' output '{output}'")
    
    # 4. Đề xuất tối ưu hóa
    print("\n4. ĐỀ XUẤT TỐI ƯU HÓA:")
    
    total_redundant = len(identity_nodes) + len(redundant_reshapes) + len(unused_outputs)
    
    if total_redundant > 0:
        print(f"   - Có thể loại bỏ {total_redundant} node(s) dư thừa")
        print(f"   - Tiết kiệm được khoảng {total_redundant/len(model.graph.node)*100:.1f}% số node")
    else:
        print("   - Không phát hiện node dư thừa rõ ràng")
    
    # Đề xuất các công cụ tối ưu hóa
    print("\n   Công cụ tối ưu hóa được đề xuất:")
    print("   - onnx.optimizer: loại bỏ node dư thừa cơ bản")
    print("   - onnxsim: tối ưu hóa toàn diện")
    print("   - TensorRT/OpenVINO: tối ưu hóa cho deployment")

def remove_identity_nodes(model_path, output_path):
    """
    Ví dụ loại bỏ các node Identity dư thừa
    """
    model = onnx.load(model_path)
    
    # Tìm tất cả node Identity
    identity_nodes = [node for node in model.graph.node if node.op_type == 'Identity']
    
    if not identity_nodes:
        print("Không có node Identity nào để loại bỏ")
        return
    
    print(f"Đang loại bỏ {len(identity_nodes)} node Identity...")
    
    # Tạo mapping để thay thế
    replacements = {}
    for node in identity_nodes:
        if len(node.input) == 1 and len(node.output) == 1:
            replacements[node.output[0]] = node.input[0]
    
    # Loại bỏ các node Identity
    model.graph.node[:] = [node for node in model.graph.node if node.op_type != 'Identity']
    
    # Cập nhật các tham chiếu
    for node in model.graph.node:
        for i, inp in enumerate(node.input):
            if inp in replacements:
                node.input[i] = replacements[inp]
    
    # Cập nhật model outputs nếu cần
    for output in model.graph.output:
        if output.name in replacements:
            output.name = replacements[output.name]
    
    # Lưu model đã tối ưu
    onnx.save(model, output_path)
    print(f"Đã lưu model tối ưu vào: {output_path}")

# Ví dụ sử dụng
if __name__ == "__main__":
    # Thay đổi đường dẫn file của bạn
    model_path = "redundant_model.onnx"
    
    try:
        # Phân tích node dư thừa
        analyze_redundant_nodes(model_path)
        
        print("\n" + "="*50)
        print("VÍ DỤ TỐI ƯU HÓA:")
        
        # Ví dụ loại bỏ Identity nodes
        output_path = "model_optimized.onnx"
        remove_identity_nodes(model_path, output_path)
        
        # So sánh trước và sau tối ưu hóa  
        print(f"\nSo sánh kích thước file:")
        import os
        if os.path.exists(model_path):
            original_size = os.path.getsize(model_path)
            print(f"Original: {original_size/1024/1024:.2f} MB")
            
        if os.path.exists(output_path):
            optimized_size = os.path.getsize(output_path)
            print(f"Optimized: {optimized_size/1024/1024:.2f} MB")
            print(f"Tiết kiệm: {(original_size-optimized_size)/original_size*100:.1f}%")
            
    except FileNotFoundError:
        print(f"Không tìm thấy file: {model_path}")
        print("Hãy thay đổi đường dẫn file trong biến model_path")
    except Exception as e:
        print(f"Lỗi: {e}")

# Hàm bổ sung: Sử dụng onnx-simplifier
def optimize_with_onnxsim(model_path, output_path):
    """
    Sử dụng onnx-simplifier để tối ưu hóa
    Cần cài: pip install onnx-simplifier
    """
    try:
        import onnxsim
        
        model = onnx.load(model_path)
        model_simplified, check = onnxsim.simplify(model)
        
        if check:
            onnx.save(model_simplified, output_path)
            print(f"Đã tối ưu hóa với onnx-simplifier: {output_path}")
        else:
            print("Không thể tối ưu hóa với onnx-simplifier")
            
    except ImportError:
        print("Cần cài đặt: pip install onnx-simplifier")
    except Exception as e:
        print(f"Lỗi khi sử dụng onnx-simplifier: {e}")