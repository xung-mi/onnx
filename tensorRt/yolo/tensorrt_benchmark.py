import onnxruntime as ort
import tensorrt as trt
import numpy as np
import time
import pycuda.driver as cuda
import pycuda.autoinit
import os

# Global config
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
BATCH_SIZE = 1
HEIGHT = 640
WIDTH = 640
CHANNEL = 3
NUM_ITER = 100

# Chuẩn bị dữ liệu input test giống nhau cho cả 2 benchmark
input_data = np.random.rand(BATCH_SIZE, CHANNEL, HEIGHT, WIDTH).astype(np.float32)

########################################
# ONNX Inference Benchmark
########################################

def benchmark_onnx(onnx_path):
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"Không tìm thấy file {onnx_path}")

    ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = ort_session.get_inputs()[0].name

    # Warm-up
    for _ in range(10):
        _ = ort_session.run(None, {input_name: input_data})

    # Benchmark
    start = time.perf_counter()
    for _ in range(NUM_ITER):
        _ = ort_session.run(None, {input_name: input_data})
    end = time.perf_counter()

    avg_time = (end - start) / NUM_ITER
    print(f"ONNX avg time: {avg_time * 1000:.3f} ms")
    return avg_time, ort_session, input_name

########################################
# TensorRT Inference Benchmark
########################################

def load_engine(engine_path):
    with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def benchmark_tensorrt(engine_path):
    if not os.path.exists(engine_path):
        raise FileNotFoundError(f"Không tìm thấy file {engine_path}")

    engine = load_engine(engine_path)
    context = engine.create_execution_context()

    input_shape = engine.get_binding_shape(0)
    output_shape = engine.get_binding_shape(1)

    h_input = input_data
    h_output = np.empty(output_shape, dtype=np.float32)

    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    bindings = [int(d_input), int(d_output)]

    # Warm-up
    for _ in range(10):
        cuda.memcpy_htod(d_input, h_input)
        context.execute_v2(bindings)
        cuda.memcpy_dtoh(h_output, d_output)

    # Benchmark
    start = time.perf_counter()
    for _ in range(NUM_ITER):
        cuda.memcpy_htod(d_input, h_input)
        context.execute_v2(bindings)
        cuda.memcpy_dtoh(h_output, d_output)
    end = time.perf_counter()

    avg_time = (end - start) / NUM_ITER
    print(f"TensorRT avg time: {avg_time * 1000:.3f} ms")
    return avg_time, engine, context, d_input, d_output, bindings, h_output.shape

########################################
# So sánh output giữa ONNX & TensorRT
########################################

def compare_outputs(ort_session, input_name, engine, context, d_input, d_output, bindings, output_shape):
    # ONNX inference
    onnx_output = ort_session.run(None, {input_name: input_data})[0]

    # TensorRT inference
    h_input = input_data
    h_output = np.empty(output_shape, dtype=np.float32)
    cuda.memcpy_htod(d_input, h_input)
    context.execute_v2(bindings)
    cuda.memcpy_dtoh(h_output, d_output)

    # So sánh
    diff = np.abs(onnx_output.flatten() - h_output.flatten())
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    print(f"Max difference: {max_diff}")
    print(f"Mean difference: {mean_diff}")

########################################
# Run full benchmark
########################################

if __name__ == "__main__":
    onnx_path = "yolov5s.onnx"
    engine_path = "yolov5s.engine"

    print("=== Bắt đầu benchmark ===")

    onnx_time, ort_session, input_name = benchmark_onnx(onnx_path)
    trt_time, engine, context, d_input, d_output, bindings, output_shape = benchmark_tensorrt(engine_path)

    speedup = onnx_time / trt_time
    print(f"TensorRT nhanh hơn {speedup:.2f} lần so với ONNX")

    print("=== So sánh output ===")
    compare_outputs(ort_session, input_name, engine, context, d_input, d_output, bindings, output_shape)
