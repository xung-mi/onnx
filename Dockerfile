# nvidia-smi => cuda driver 12.2
# nvcc --version => CUDA Toolkit (nvcc) 12.1
# cat /etc/os-release => ubuntu version 22.04
# uname -m => architecture : x86_64

# Base image: TensorRT official
FROM nvcr.io/nvidia/tensorrt:23.06-py3

# Environment variables for non-interactive install
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Install system dependencies & NVIDIA key
RUN apt-get update && apt-get install -y \
    wget gnupg2 lsb-release && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    rm -f cuda-keyring_1.1-1_all.deb

# Remove cuDNN 8 cleanly if exists
RUN (apt-get purge -y libcudnn8 libcudnn8-dev || true) && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install cuDNN 9 (CUDA 12)
RUN apt-get update && apt-get install -y cudnn-cuda-12 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip first
RUN pip install --upgrade pip

# Install core production dependencies first (stable packages)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Install dev/test packages (less cache priority)
COPY requirements-dev.txt /tmp/requirements-dev.txt
RUN pip install --no-cache-dir -r /tmp/requirements-dev.txt

# App layer (keep it last to leverage Docker layer caching)
WORKDIR /app
COPY . /app

CMD ["python3"]

