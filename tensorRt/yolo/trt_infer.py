import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import cv2
import os
from glob import glob
import argparse

# Thông số model
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
CONF_THRESHOLD = 0.3
IOU_THRESHOLD = 0.4

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_engine(engine_path):
    with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def nms(boxes, scores, iou_threshold):
    indices = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESHOLD, iou_threshold)
    return indices.flatten() if len(indices) > 0 else []

def decode_output(output, img_shape):
    boxes = []
    confidences = []
    class_ids = []

    for pred in output[0]:
        confidence = pred[4]
        if confidence < CONF_THRESHOLD:
            continue

        class_scores = pred[5:]
        class_id = np.argmax(class_scores)
        score = class_scores[class_id] * confidence

        if score < CONF_THRESHOLD:
            continue

        cx, cy, w, h = pred[:4]
        x = (cx - w / 2) * img_shape[1] / INPUT_WIDTH
        y = (cy - h / 2) * img_shape[0] / INPUT_HEIGHT
        w = w * img_shape[1] / INPUT_WIDTH
        h = h * img_shape[0] / INPUT_HEIGHT

        boxes.append([int(x), int(y), int(w), int(h)])
        confidences.append(float(score))
        class_ids.append(class_id)

    indices = nms(boxes, confidences, IOU_THRESHOLD)
    results = [(boxes[i], confidences[i], class_ids[i]) for i in indices]
    return results

def infer_single_image(context, image_path, output_dir):
    original_img = cv2.imread(image_path)
    img_resized = cv2.resize(original_img, (INPUT_WIDTH, INPUT_HEIGHT))
    img = img_resized[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
    input_batch = np.expand_dims(img, axis=0)

    d_input = cuda.mem_alloc(input_batch.nbytes)
    h_output = np.empty((1, 25200, 85), dtype=np.float32)
    d_output = cuda.mem_alloc(h_output.nbytes)
    bindings = [int(d_input), int(d_output)]

    cuda.memcpy_htod(d_input, input_batch)
    context.execute_v2(bindings)
    cuda.memcpy_dtoh(h_output, d_output)

    detections = decode_output(h_output, original_img.shape)

    for box, score, class_id in detections:
        x, y, w, h = box
        cv2.rectangle(original_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"{class_id}: {score:.2f}"
        cv2.putText(original_img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    input_filename = os.path.basename(image_path)
    name, ext = os.path.splitext(input_filename)
    output_path = os.path.join(output_dir, f"{name}_output{ext}")

    cv2.imwrite(output_path, original_img)
    print(f"Processed: {image_path} -> {output_path}")

def infer_batch_folder(engine_path, input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    engine = load_engine(engine_path)
    context = engine.create_execution_context()

    image_files = glob(os.path.join(input_dir, '*.jpg')) + glob(os.path.join(input_dir, '*.png'))

    print(f"Found {len(image_files)} images in {input_dir}")

    for image_path in image_files:
        infer_single_image(context, image_path, output_dir)

def main():
    parser = argparse.ArgumentParser(description="YOLOv5 TensorRT Batch Inference")
    parser.add_argument("--engine", type=str, required=True, help="Path to TensorRT engine file")
    parser.add_argument("--input", type=str, required=True, help="Input folder containing images")
    parser.add_argument("--output", type=str, required=True, help="Output folder to save results")

    args = parser.parse_args()

    infer_batch_folder(args.engine, args.input, args.output)

if __name__ == "__main__":
    main()
