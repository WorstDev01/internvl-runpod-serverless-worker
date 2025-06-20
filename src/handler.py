import os
import time
import base64
import runpod
import requests
import torch

# LMDeploy imports
from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig, ChatTemplateConfig
from lmdeploy.vl import load_image
from lmdeploy.vl.constants import IMAGE_TOKEN

# Global variables
pipe = None


def initialize_lmdeploy(input_data):
    global pipe

    if not pipe:
        print("Initializing LMDeploy...")
        start_time = time.time()

        model_path = "/workspace/models/InternVL3-14B"

        # LMDeploy backend configuration (adjusted for smaller model)
        backend_config = TurbomindEngineConfig(
            session_len=8192,
            tp=1,  # Tensor parallelism (1 for single GPU)
            cache_max_entry_count=0.8,  # KV cache usage
        )

        # Chat template configuration for InternVL3
        chat_template_config = ChatTemplateConfig(
            model_name='internvl2_5'  # InternVL3 uses internvl2_5 template
        )

        print(f"Model path: {model_path}")
        print(f"Backend config: session_len={backend_config.session_len}, tp={backend_config.tp}")

        # Initialize LMDeploy pipeline
        pipe = pipeline(
            model_path,
            backend_config=backend_config,
            chat_template_config=chat_template_config
        )

        print('─' * 20,
              f"--- LMDeploy initialization took {time.time() - start_time:.2f} seconds ---",
              '─' * 20, "*Unfortunately, this time is being billed.", sep=os.linesep)


def download_image_from_url(url):
    """Download image from URL and return PIL Image"""
    try:
        download_start = time.time()
        print(f"Downloading image from URL: {url}")
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        # Save to temporary file for LMDeploy
        temp_path = f"/tmp/image_{hash(url)}.jpg"
        with open(temp_path, 'wb') as f:
            f.write(response.content)

        # Load with LMDeploy's load_image function
        image = load_image(temp_path)
        download_time = time.time() - download_start
        print(f"Downloaded and loaded image in {download_time:.3f}s from: {url}")
        return image
    except Exception as e:
        print(f"Error downloading image from URL {url}: {e}")
        return None


def decode_base64_image(base64_string):
    """Decode base64 image string and save for LMDeploy"""
    try:
        decode_start = time.time()
        # Remove data URL prefix if present
        if base64_string.startswith('data:'):
            base64_string = base64_string.split(',')[1]

        # Decode base64
        image_data = base64.b64decode(base64_string)

        # Save to temporary file for LMDeploy
        temp_path = f"/tmp/image_{hash(base64_string[:100])}.jpg"
        with open(temp_path, 'wb') as f:
            f.write(image_data)

        # Load with LMDeploy's load_image function
        image = load_image(temp_path)
        decode_time = time.time() - decode_start
        print(f"Decoded and loaded base64 image in {decode_time:.3f}s")
        return image
    except Exception as e:
        print(f"Error decoding base64 image: {e}")
        return None


def process_batch_requests(batch_data):
    """Convert batch requests to LMDeploy format"""
    print(f"🔍 DEBUG: Starting batch processing for {len(batch_data)} requests")
    batch_start_time = time.time()

    processed_prompts = []
    image_count = 0
    text_only_count = 0
    total_image_processing_time = 0

    for i, request in enumerate(batch_data):
        request_start = time.time()

        if "messages" in request:
            # Handle chat format with multimodal support
            messages = request["messages"]

            # Process each message to extract text and images
            text_parts = []
            images = []

            for message in messages:
                role = message.get("role", "")
                content = message.get("content", "")

                if isinstance(content, list):
                    # Handle multimodal content
                    for item in content:
                        if item.get("type") == "text":
                            text_parts.append(item["text"])
                        elif item.get("type") == "image_url":
                            # Extract image data
                            image_url = item.get("image_url", {})
                            url = image_url.get("url", "")

                            image_process_start = time.time()
                            if url.startswith("data:"):
                                # Handle base64 encoded image
                                image = decode_base64_image(url)
                                if image:
                                    images.append(image)
                            elif url.startswith(("http://", "https://")):
                                # Handle HTTP/HTTPS URLs
                                image = download_image_from_url(url)
                                if image:
                                    images.append(image)
                            else:
                                print(f"Unsupported image URL format: {url}")

                            image_process_time = time.time() - image_process_start
                            total_image_processing_time += image_process_time
                            print(f"📷 Image {i} processing took: {image_process_time:.3f}s")
                else:
                    # Simple text content
                    text_parts.append(str(content))

            # Combine text parts
            combined_text = " ".join(text_parts)

            # For LMDeploy, create prompt with images
            if images:
                # Use first image (LMDeploy format)
                prompt_tuple = (combined_text, images[0])
                processed_prompts.append(prompt_tuple)
                image_count += 1
                print(f"Created multimodal prompt {i} with image")
            else:
                # Text-only request
                processed_prompts.append(combined_text)
                text_only_count += 1

        elif "prompt" in request:
            # Handle simple prompt format
            processed_prompts.append(str(request["prompt"]))
            text_only_count += 1
        else:
            # Fallback
            processed_prompts.append(str(request))
            text_only_count += 1

        request_time = time.time() - request_start
        print(f"⏱️  Request {i} total processing: {request_time:.3f}s")

    batch_processing_time = time.time() - batch_start_time

    print(f"🔍 DEBUG: Batch processing summary:")
    print(f"  📊 Total requests: {len(batch_data)}")
    print(f"  📷 Image requests: {image_count}")
    print(f"  📝 Text-only requests: {text_only_count}")
    print(f"  ⏱️  Total batch processing time: {batch_processing_time:.3f}s")
    print(f"  📷 Total image processing time: {total_image_processing_time:.3f}s")
    print(f"  ⚡ Average per request: {batch_processing_time / len(batch_data):.3f}s")
    if image_count > 0:
        print(f"  📷 Average per image: {total_image_processing_time / image_count:.3f}s")

    return processed_prompts


def create_generation_config(batch_data):
    """Create LMDeploy generation configuration from first request"""
    first_request = batch_data[0] if batch_data else {}

    # Default generation config (adjusted for smaller 1B model)
    config_params = {
        "max_new_tokens": 512,  # Reduced for 1B model
        "temperature": 0.1,
    }

    # Override with provided parameters
    if "max_tokens" in first_request:
        config_params["max_new_tokens"] = first_request["max_tokens"]
    if "temperature" in first_request:
        config_params["temperature"] = first_request["temperature"]
    if "top_p" in first_request:
        config_params["top_p"] = first_request["top_p"]
    if "top_k" in first_request:
        config_params["top_k"] = first_request["top_k"]

    print(f"Generation config: {config_params}")

    return GenerationConfig(**config_params)


def handler(job):
    try:
        handler_start_time = time.time()
        input_data = job["input"]
        print(f"🚀 Handler started - Received input data: {input_data}")

        # Initialize LMDeploy
        initialize_lmdeploy(input_data)

        # Handle prewarm
        if "prewarm" in input_data:
            return {"warm": True}

        # Get batch requests
        if "batch" not in input_data:
            return {"error": "Expected 'batch' key with list of requests"}

        batch_requests = input_data["batch"]

        if not batch_requests:
            return {"error": "Batch requests list is empty"}

        print(f"🔥 Processing {len(batch_requests)} batch requests with LMDeploy")

        # GPU Memory before processing
        if torch.cuda.is_available():
            memory_before = torch.cuda.memory_allocated() / 1024 ** 3  # GB
            memory_reserved_before = torch.cuda.memory_reserved() / 1024 ** 3  # GB
            print(
                f"🧠 GPU Memory before processing: {memory_before:.2f}GB allocated, {memory_reserved_before:.2f}GB reserved")

        # Process requests into LMDeploy format
        preprocessing_start = time.time()
        processed_prompts = process_batch_requests(batch_requests)
        preprocessing_time = time.time() - preprocessing_start

        print(f"✅ Processed {len(processed_prompts)} prompts for LMDeploy")
        print(f"⏱️  PREPROCESSING TOTAL TIME: {preprocessing_time:.3f}s")
        print(f"⚡ Preprocessing per request: {preprocessing_time / len(processed_prompts):.3f}s")

        # Log prompt types
        multimodal_count = sum(1 for prompt in processed_prompts if isinstance(prompt, tuple))
        text_only_count = len(processed_prompts) - multimodal_count
        print(f"📊 Prompt breakdown: {multimodal_count} multimodal, {text_only_count} text-only")

        for i, prompt in enumerate(processed_prompts[:3]):  # Show first 3 only
            if isinstance(prompt, tuple):
                print(f"Prompt {i}: Multimodal - {prompt[0][:100]}...")
            else:
                print(f"Prompt {i}: Text only - {str(prompt)[:100]}...")

        # Create generation configuration
        gen_config = create_generation_config(batch_requests)

        # GPU Memory before inference
        if torch.cuda.is_available():
            memory_before_inference = torch.cuda.memory_allocated() / 1024 ** 3  # GB
            print(f"🧠 GPU Memory before inference: {memory_before_inference:.2f}GB allocated")

        # Generate responses using LMDeploy batch inference
        print(f"🔥 Starting LMDeploy batch inference...")
        inference_start_time = time.time()

        # LMDeploy batch inference
        responses = pipe(processed_prompts, gen_config=gen_config)

        inference_time = time.time() - inference_start_time
        print(f"✅ LMDeploy batch generation completed!")
        print(f"⏱️  INFERENCE TOTAL TIME: {inference_time:.3f}s")
        print(f"⚡ Inference per request: {inference_time / len(responses):.3f}s")

        # GPU Memory after inference
        if torch.cuda.is_available():
            memory_after = torch.cuda.memory_allocated() / 1024 ** 3  # GB
            memory_peak = torch.cuda.max_memory_allocated() / 1024 ** 3  # GB
            print(f"🧠 GPU Memory after inference: {memory_after:.2f}GB allocated")
            print(f"🧠 GPU Memory peak during inference: {memory_peak:.2f}GB")

        # Format results
        results_start = time.time()
        results = []
        for i, response in enumerate(responses):
            # LMDeploy response format
            if hasattr(response, 'text'):
                generated_text = response.text
            else:
                generated_text = str(response)

            if i < 3:  # Show first 3 only
                print(f"Generated text {i}: '{generated_text[:100]}...'")

            result = {
                "index": i,
                "text": generated_text,
                "finish_reason": "stop",  # LMDeploy doesn't provide detailed finish reasons
                "prompt_tokens": None,  # Would need to calculate separately
                "completion_tokens": None  # Would need to calculate separately
            }
            results.append(result)

        results_time = time.time() - results_start
        total_handler_time = time.time() - handler_start_time

        # FINAL PERFORMANCE SUMMARY
        print(f"\n" + "=" * 60)
        print(f"🎯 PERFORMANCE SUMMARY")
        print(f"=" * 60)
        print(f"📊 Batch size: {len(results)} requests")
        print(f"📷 Multimodal requests: {multimodal_count}")
        print(f"📝 Text-only requests: {text_only_count}")
        print(f"")
        print(f"⏱️  TIMING BREAKDOWN:")
        print(f"  🔄 Preprocessing: {preprocessing_time:.3f}s ({preprocessing_time / total_handler_time * 100:.1f}%)")
        print(f"  🧠 Model inference: {inference_time:.3f}s ({inference_time / total_handler_time * 100:.1f}%)")
        print(f"  📝 Results formatting: {results_time:.3f}s ({results_time / total_handler_time * 100:.1f}%)")
        print(f"  🚀 Total handler time: {total_handler_time:.3f}s")
        print(f"")
        print(f"⚡ THROUGHPUT:")
        print(f"  📈 Requests per second: {len(results) / total_handler_time:.2f}")
        print(f"  ⏱️  Average per request: {total_handler_time / len(results):.4f}s")
        print(f"  🔥 Pure inference RPS: {len(results) / inference_time:.2f}")
        print(f"  📷 Preprocessing overhead per request: {preprocessing_time / len(results):.4f}s")
        print(f"")
        if multimodal_count > 0:
            print(f"🖼️  IMAGE PROCESSING:")
            print(f"  📷 Average preprocessing per multimodal request: {preprocessing_time / multimodal_count:.4f}s")
        print(f"=" * 60)

        return {"results": results}

    except Exception as e:
        print(f"❌ Error in LMDeploy handler: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})