# Chuẩn bị dataset 
import os
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda

# Chuẩn bị dataset 
class YOLOv5CalibDataset:
    def __init__(self, image_dir, input_shape=(3, 640, 640)):
        self.image_dir = image_dir
        self.input_shape = input_shape  # (C, H, W)
        self.image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
        self.index = 0
        self.total = len(self.image_files)
        print(f"Found {self.total} images for calibration.")

    def reset(self):
        self.index = 0

    def read_next_batch(self, batch_size):
        if self.index >= self.total:
            return None
        
        batch_data = []
        for _ in range(batch_size):
            if self.index >= self.total:
                break

            img_path = self.image_files[self.index]
            img = self.preprocess(img_path)
            batch_data.append(img)
            self.index += 1

        if len(batch_data) == 0:
            return None
        
        batch_data = np.stack(batch_data, axis=0)
        return batch_data

    def preprocess(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (self.input_shape[2], self.input_shape[1]))  # resize to (W, H)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        img = img.astype(np.float32) / 255.0
        return img

# plug dataset vào TensorRT IInt8EntropyCalibrator2
class YOLOv5EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, dataset, batch_size):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.input_shape = (batch_size, 3, 640, 640)
        # self.device_input = cuda.mem_alloc(np.prod(self.input_shape) * np.float32().nbytes)
        '''
        NumPy và PyCUDA dùng 2 backend khác nhau. PyCUDA không tự động convert numpy.int64 về C unsigned long được => cần ép về native Python int
        '''
        size_in_bytes = int(np.prod(self.input_shape) * np.float32().nbytes)
        self.device_input = cuda.mem_alloc(size_in_bytes)

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        batch = self.dataset.read_next_batch(self.batch_size)
        if batch is None:
            return None
        cuda.memcpy_htod(self.device_input, batch)
        return [int(self.device_input)]

    def read_calibration_cache(self):
        if os.path.exists("calib.cache"):
            return open("calib.cache", "rb").read()
        return None

    def write_calibration_cache(self, cache):
        with open("calib.cache", "wb") as f:
            f.write(cache)



# Load dataset
dataset = YOLOv5CalibDataset("calib_images")
batch_size = 16
# Khởi tạo calibrator
calibrator = YOLOv5EntropyCalibrator(dataset, batch_size)

