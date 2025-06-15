import onnxruntime
import numpy as np
import cv2
import os
from glob import glob
import argparse

# Config giống bản TensorRT cho dễ so sánh
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
CONF_THRESHOLD = 0.3
IOU_THRESHOLD = 0.4

def nms(boxes, scores, iou_threshold):
    indices = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESHOLD, iou_threshold)
    return indices.flatten() if len(indices) > 0 else []

def preprocess_image(image_path):
    original_img = cv2.imread(image_path)
    img_resized = cv2.resize(original_img, (INPUT_WIDTH, INPUT_HEIGHT))
    img = img_resized[:, :, ::-1].transpose(2, 0, 1)  # BGR -> RGB -> CHW
    img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
    input_tensor = np.expand_dims(img, axis=0)  # Add batch dimension
    return input_tensor, original_img

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

def onnx_inference(onnx_path, img_folder_path, output_folder_path):
    os.makedirs(output_folder_path, exist_ok=True)

    # Load ONNX model
    session = onnxruntime.InferenceSession(onnx_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    image_files = glob(os.path.join(img_folder_path, '*.jpg')) + glob(os.path.join(img_folder_path, '*.png'))
    print(f"Found {len(image_files)} images")

    for image_path in image_files:
        input_tensor, original_img = preprocess_image(image_path)

        # Run inference
        outputs = session.run(None, {input_name: input_tensor.astype(np.float32)})
        predictions = outputs[0]  # Shape: (1, 25200, 85)

        detections = decode_output(predictions, original_img.shape)

        for box, score, class_id in detections:
            x, y, w, h = box
            cv2.rectangle(original_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            label = f"{class_id}: {score:.2f}"
            cv2.putText(original_img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Save output
        filename = os.path.basename(image_path)
        name, ext = os.path.splitext(filename)
        output_path = os.path.join(output_folder_path, f"{name}_onnx{ext}")
        cv2.imwrite(output_path, original_img)
        print(f"Saved ONNX output to {output_path}")

# CLI Entry
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv5 ONNX Inference CLI")
    parser.add_argument("--onnx", type=str, required=True, help="Path to yolov5s.onnx model file")
    parser.add_argument("--input", type=str, required=True, help="Path to input image folder")
    parser.add_argument("--output", type=str, required=True, help="Path to output result folder")
    args = parser.parse_args()

    onnx_inference(args.onnx, args.input, args.output)
