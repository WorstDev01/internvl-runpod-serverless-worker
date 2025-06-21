import runpod
from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig, ChatTemplateConfig
from lmdeploy.vl import load_image
import time
import subprocess
import psutil
import gc
import torch

# Global pipeline variable
pipe = None


def get_gpu_memory():
    """Get current GPU memory usage"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'],
                                capture_output=True, text=True)
        if result.returncode == 0:
            used, total = result.stdout.strip().split(', ')
            return int(used), int(total)
    except:
        pass
    return None, None


def get_system_memory():
    """Get system RAM usage"""
    memory = psutil.virtual_memory()
    return memory.used // 1024 // 1024, memory.total // 1024 // 1024  # MB


def log_memory_usage(stage):
    """Log detailed memory usage at different stages"""
    gpu_used, gpu_total = get_gpu_memory()
    ram_used, ram_total = get_system_memory()

    print(f"\n=== MEMORY USAGE - {stage} ===")
    if gpu_used and gpu_total:
        print(f"GPU VRAM: {gpu_used}MB / {gpu_total}MB ({gpu_used / gpu_total * 100:.1f}%)")
    print(f"System RAM: {ram_used}MB / {ram_total}MB ({ram_used / ram_total * 100:.1f}%)")

    # PyTorch memory if available
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() // 1024 // 1024
        cached = torch.cuda.memory_reserved() // 1024 // 1024
        print(f"PyTorch Allocated: {allocated}MB")
        print(f"PyTorch Cached: {cached}MB")
    print("=" * 40)


def initialize_pipeline():
    """Initialize the LMDeploy pipeline once"""
    global pipe

    if not pipe:
        log_memory_usage("BEFORE MODEL LOAD")

        model_path = "/workspace/models/InternVL3-14B"

        pipe = pipeline(
            model_path,
            backend_config=TurbomindEngineConfig(session_len=16384, tp=1),
            chat_template_config=ChatTemplateConfig(model_name='internvl2_5')
        )

        log_memory_usage("AFTER MODEL LOAD")


def process_batch_requests(batch_data):
    """Convert batch requests to LMDeploy format (text, image) tuples"""

    # First, collect all image URLs
    image_urls = []
    texts = []

    for request in batch_data:
        text = ""
        image_url = ""
        content = request["messages"][0]["content"]

        for item in content:
            if item["type"] == "text":
                text = item["text"]
            elif item["type"] == "image_url":
                image_url = item["image_url"]["url"]

        texts.append(text)
        image_urls.append(image_url)

    # Load all images in parallel/batch
    images = [load_image(url) if url else None for url in image_urls]

    # Create prompts
    prompts = []
    for text, image in zip(texts, images):
        if image:
            prompts.append((text, image))
        else:
            prompts.append(text)

    return prompts


def create_generation_config(batch_data):
    """Create generation config from request parameters"""
    first_request = batch_data[0] if batch_data else {}

    return GenerationConfig(
        max_new_tokens=first_request.get("max_tokens", 512),
        temperature=first_request.get("temperature", 0.1),
        top_p=first_request.get("top_p", 1.0),
        top_k=first_request.get("top_k", 50)
    )


def handler(job):
    """Main RunPod handler function"""
    try:
        input_data = job["input"]

        log_memory_usage("HANDLER START")

        # Initialize pipeline
        initialize_pipeline()

        # Handle prewarm
        if "prewarm" in input_data:
            log_memory_usage("AFTER PREWARM")
            return {"warm": True}

        # Get batch requests
        batch_requests = input_data.get("batch", [])

        if not batch_requests:
            return {"error": "Batch requests list is empty"}

        print(f"\nProcessing {len(batch_requests)} prompts...")
        start_time = time.time()

        # Process batch requests into (text, image) tuples
        preprocessing_start = time.time()
        prompts = process_batch_requests(batch_requests)
        preprocessing_end = time.time()

        print(f"Preprocessing time: {preprocessing_end - preprocessing_start:.2f}s")

        if not prompts:
            return {"error": "No valid prompts found"}

        # Create generation config
        gen_config = create_generation_config(batch_requests)

        log_memory_usage("BEFORE INFERENCE")

        # Run batch inference
        inference_start = time.time()
        print(f"Starting inference with {len(prompts)} prompts...")
        responses = pipe(prompts, gen_config=gen_config)
        inference_end = time.time()

        log_memory_usage("AFTER INFERENCE")

        # Format results
        results = []
        for i, response in enumerate(responses):
            results.append({
                "index": i,
                "text": response.text if hasattr(response, 'text') else str(response),
                "finish_reason": "stop"
            })

        end_time = time.time()
        total_time = end_time - start_time
        inference_time = inference_end - inference_start
        preprocessing_time = preprocessing_end - preprocessing_start

        print(f"\n=== PERFORMANCE SUMMARY ===")
        print(f"Preprocessing time: {preprocessing_time:.2f}s")
        print(f"Pure inference time: {inference_time:.2f}s")
        print(f"Total time: {total_time:.2f}s")
        print(f"Prompts per second: {len(batch_requests) / inference_time:.2f}")
        print("=" * 30)

        # Clean up
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        log_memory_usage("AFTER CLEANUP")

        return {"results": results}

    except Exception as e:
        import traceback
        traceback.print_exc()
        log_memory_usage("ERROR STATE")
        return {"error": str(e)}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})