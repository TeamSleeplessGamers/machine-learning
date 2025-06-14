# Use a base image with Python + CUDA
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Set working directory
WORKDIR /workspace

# Install OS-level dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    netcat \
    python3.13 \
    python3-pip \
    python3-venv \
    && apt-get clean

# Copy your setup script into the container
COPY scripts/gpu_startup.sh /opt/init/run_gpu_setup.sh
RUN chmod +x /opt/init/run_gpu_setup.sh

CMD ["/bin/bash", "-c", "/opt/init/run_gpu_setup.sh || tail -f /dev/null"]
