import runpod
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import time
import subprocess
import psutil
import gc
import requests
from io import BytesIO

# Global model variables
model = None
tokenizer = None

# Constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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


def build_transform(input_size=448):
    """Build image transformation pipeline"""
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform


def load_image_from_url(url):
    """Load image from URL"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert('RGB')
    except Exception as e:
        print(f"Error loading image from {url}: {e}")
        return None


def prepare_images_batch(image_urls, input_size=448):
    """Load and transform multiple images for batch processing"""
    transform = build_transform(input_size)

    pixel_values_list = []
    num_patches_list = []
    valid_indices = []

    for idx, url in enumerate(image_urls):
        if url:
            image = load_image_from_url(url)
            if image:
                pixel_values = transform(image).unsqueeze(0)
                pixel_values_list.append(pixel_values)
                num_patches_list.append(1)
                valid_indices.append(idx)
        else:
            # No image URL provided
            valid_indices.append(idx)

    if pixel_values_list:
        pixel_values = torch.cat(pixel_values_list, dim=0)
        return pixel_values, num_patches_list, valid_indices

    return None, [], valid_indices


def initialize_model():
    """Initialize the model and tokenizer once"""
    global model, tokenizer

    if model is None:
        log_memory_usage("BEFORE MODEL LOAD")

        model_path = 'OpenGVLab/InternVL3-1B'  # Or use your local path

        print(f"Loading model from {model_path}...")
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map='auto'
        ).eval()

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False
        )

        print("Model loaded successfully!")
        log_memory_usage("AFTER MODEL LOAD")


def process_batch_requests(batch_data):
    """Extract texts and image URLs from batch requests"""
    texts = []
    image_urls = []

    for request in batch_data:
        text = ""
        image_url = ""

        # Handle OpenAI-style format
        if "messages" in request:
            content = request["messages"][0]["content"]

            for item in content:
                if item["type"] == "text":
                    text = item["text"]
                elif item["type"] == "image_url":
                    image_url = item["image_url"]["url"]
        else:
            # Handle simple format
            text = request.get("text", "")
            image_url = request.get("image_url", "")

        texts.append(text)
        image_urls.append(image_url)

    return texts, image_urls


def create_generation_config(batch_data):
    """Create generation config from request parameters"""
    first_request = batch_data[0] if batch_data else {}

    return {
        "max_new_tokens": first_request.get("max_tokens", 256),
        "temperature": first_request.get("temperature", 0.0),
        "top_p": first_request.get("top_p", 1.0),
        "top_k": first_request.get("top_k", 50),
        "do_sample": first_request.get("temperature", 0.0) > 0
    }


def handler(job):
    """Main RunPod handler function"""
    try:
        input_data = job["input"]

        log_memory_usage("HANDLER START")

        # Initialize model
        initialize_model()

        # Handle prewarm
        if "prewarm" in input_data:
            log_memory_usage("AFTER PREWARM")
            return {"warm": True}

        # Get batch requests
        batch_requests = input_data.get("batch", [])

        if not batch_requests:
            return {"error": "Batch requests list is empty"}

        print(f"\nProcessing {len(batch_requests)} requests...")
        start_time = time.time()

        # Process batch requests
        preprocessing_start = time.time()
        texts, image_urls = process_batch_requests(batch_requests)

        # Prepare images if any
        pixel_values = None
        num_patches_list = []
        valid_indices = list(range(len(texts)))

        if any(image_urls):
            pixel_values, num_patches_list, valid_indices = prepare_images_batch(image_urls)
            if pixel_values is not None:
                pixel_values = pixel_values.to(torch.bfloat16).cuda()

        preprocessing_end = time.time()
        print(f"Preprocessing time: {preprocessing_end - preprocessing_start:.2f}s")

        # Create generation config
        gen_config = create_generation_config(batch_requests)

        log_memory_usage("BEFORE INFERENCE")

        # Prepare questions for batch inference
        questions = []
        for idx, (text, image_url) in enumerate(zip(texts, image_urls)):
            if image_url and pixel_values is not None:
                # Format with image placeholder
                question = f"<image>\n{text}" if text else "<image>\nDescribe this image."
            else:
                # Text-only question
                question = text
            questions.append(question)

        # Run batch inference
        inference_start = time.time()
        print(f"Starting inference with {len(questions)} prompts...")

        if pixel_values is not None and len(num_patches_list) > 0:
            # Image + text batch inference
            responses = model.batch_chat(
                tokenizer,
                pixel_values,
                num_patches_list=num_patches_list,
                questions=[questions[i] for i in valid_indices if i < len(questions)],
                generation_config=gen_config
            )
        else:
            # Text-only batch inference
            # Note: You might need to implement text-only batch processing
            # For now, we'll process text-only requests one by one
            responses = []
            for question in questions:
                response = model.chat(
                    tokenizer,
                    pixel_values=None,
                    question=question,
                    generation_config=gen_config
                )
                responses.append(response)

        inference_end = time.time()

        log_memory_usage("AFTER INFERENCE")

        # Format results
        results = []
        response_idx = 0

        for idx in range(len(batch_requests)):
            if idx in valid_indices and response_idx < len(responses):
                results.append({
                    "index": idx,
                    "text": responses[response_idx],
                    "finish_reason": "stop"
                })
                response_idx += 1
            else:
                results.append({
                    "index": idx,
                    "text": "Error: Failed to process this request",
                    "finish_reason": "error"
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