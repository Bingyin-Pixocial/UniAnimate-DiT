# Use NVIDIA CUDA base image with Python 3.10
FROM nvidia/cuda:12.1-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libgomp1 \
    libgcc-s1 \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install PyTorch with CUDA 12.1 support
RUN pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121

# Install additional system dependencies for attention implementations
RUN apt-get update && apt-get install -y \
    build-essential \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install -r requirements.txt

# Install additional training dependencies
RUN pip install peft lightning pandas deepspeed

# Install optional attention implementations for better performance
RUN pip install flash-attn --no-build-isolation

# Install xfuser for Unified Sequence Parallel (USP)
RUN pip install xfuser

# Install huggingface hub and modelscope for model downloading
RUN pip install "huggingface_hub[cli]" modelscope

# Install onnxruntime-gpu for faster pose extraction
RUN pip install onnxruntime-gpu==1.18.1

# Copy the project files
COPY . .

# Install the project in development mode
RUN pip install -e .

# Create necessary directories
RUN mkdir -p outputs checkpoints data/saved_pose

# Set default command
CMD ["/bin/bash"] 