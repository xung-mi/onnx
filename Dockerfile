# nvidia-smi => cuda driver 12.2
# nvcc --version => CUDA Toolkit (nvcc) 12.1
# cat /etc/os-release => ubuntu version 22.04
# uname -m => architecture : x86_64

# Base image: TensorRT 10.12 + CUDA 12.4
FROM nvcr.io/nvidia/tensorrt:24.04-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Cài các dev package để build python bindings
RUN apt-get update && apt-get install -y \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    cmake \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Build TensorRT python bindings (nếu SDK cung cấp source)
WORKDIR /usr/src/tensorrt/python
ENV PYTHON_VERSION=3.10
RUN ./python_setup.sh --python-version ${PYTHON_VERSION} --use-pip

# Cài đặt core production dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Cài đặt dev tools
COPY requirements-dev.txt /tmp/requirements-dev.txt
RUN pip install --no-cache-dir -r /tmp/requirements-dev.txt

# App layer
WORKDIR /app
COPY . /app

CMD ["python3"]
