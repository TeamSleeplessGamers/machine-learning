# ✅ Use a base image that matches your CUDA 12.1 + Python needs
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3-pip \
    git \
    curl \
    netcat \
    && apt-get clean

# Set python3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Set working directory
WORKDIR /workspace

# Install PyTorch with CUDA 12.1 (explicit install)
RUN pip install --upgrade pip
RUN pip install --no-cache-dir torch==2.3.1+cu121 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Copy and run your setup script
COPY scripts/gpu_startup.sh /opt/init/run_gpu_setup.sh
RUN chmod +x /opt/init/run_gpu_setup.sh

# Run setup, then keep container alive if it fails
CMD ["/bin/bash", "-c", "/opt/init/run_gpu_setup.sh || tail -f /dev/null"]
