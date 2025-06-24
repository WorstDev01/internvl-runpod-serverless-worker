FROM nvidia/cuda:12.4.1-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_PREFER_BINARY=1 \
    PYTHONUNBUFFERED=1

# Install system packages
RUN apt-get update && \
    apt-get install -y python3-pip git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 && \
    pip install transformers>=4.37.0 && \
    pip install pillow requests && \
    pip install runpod psutil && \
    pip install huggingface-hub timm && \
    pip install sentencepiece protobuf

# Download model using huggingface-hub
# Change this to match your model - using InternVL3-1B as in the handler
RUN python3 -c "\
from huggingface_hub import snapshot_download; \
snapshot_download( \
    'OpenGVLab/InternVL3-1B', \
    local_dir='/workspace/models/InternVL3-1B' \
)"

# Create temp directory for images
RUN mkdir -p /tmp

# Copy handler
WORKDIR /src
COPY src/handler.py .

CMD ["python3", "handler.py"]