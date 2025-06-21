FROM nvidia/cuda:12.4.1-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_PREFER_BINARY=1 \
    PYTHONUNBUFFERED=1

# Install system packages
RUN apt-get update && \
    apt-get install -y python3-pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install lmdeploy runpod huggingface-hub timm

# Download model using huggingface-hub
RUN python3 -c "\
from huggingface_hub import snapshot_download; \
snapshot_download( \
    'OpenGVLab/InternVL3-14B', \
    local_dir='/workspace/models/InternVL3-14B' \
)"

# Create temp directory for images
RUN mkdir -p /tmp

# Copy handler
WORKDIR /src
COPY src/handler.py .

CMD ["python3", "handler.py"]