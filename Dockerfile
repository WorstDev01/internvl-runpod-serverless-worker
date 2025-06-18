FROM nvidia/cuda:12.4.1-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TORCHDYNAMO_DISABLE=1 \
    PIP_PREFER_BINARY=1 \
    PYTHONUNBUFFERED=1

# Install system packages
RUN apt-get update && \
    apt-get install -y python3-pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install vllm runpod pillow huggingface-hub

# Download model using huggingface-hub (more efficient than git)
RUN python3 -c "\
from huggingface_hub import snapshot_download; \
snapshot_download( \
    'OpenGVLab/InternVL3-14B', \
    local_dir='/workspace/models/InternVL3-14B' \
)"

# Copy handler
WORKDIR /src
COPY src/handler.py .

# Set default environment variables
ENV VLLM_MODEL=/workspace/models/InternVL3-14B \
    VLLM_TRUST_REMOTE_CODE=true \
    VLLM_ENFORCE_EAGER=true \
    VLLM_LIMIT_MM_PER_PROMPT="{\"image\": 1, \"video\": 0}"

CMD ["python3", "handler.py"]