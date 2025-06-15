import onnxruntime as ort
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# Đường dẫn tới file mô hình ONNX
model_path = "resnet18.onnx"

# Khởi tạo ONNX Runtime session
session = ort.InferenceSession(model_path)

# Lấy tên input của mô hình
input_name = session.get_inputs()[0].name

# Đọc ảnh cần nhận diện
image_path = "images/pexels-ozgomz-868097.jpg"  # Thay bằng đường dẫn ảnh thực tế của bạn
img = Image.open(image_path).convert('RGB')

# Tiền xử lý ảnh theo chuẩn của ResNet18 (giống khi huấn luyện)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# Biến đổi ảnh thành tensor phù hợp với mô hình
img_tensor = preprocess(img).unsqueeze(0).numpy()

# Thực hiện suy luận (inference)
outputs = session.run(None, {input_name: img_tensor})

# Lấy output layer (vector softmax 1000 chiều)
predictions = outputs[0]

# Tìm class có xác suất cao nhất
predicted_class = np.argmax(predictions)

# In kết quả
print(f"Predicted class index: {predicted_class}")
