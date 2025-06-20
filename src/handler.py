import os
import time
import base64
import runpod
import requests

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
        print(f"Downloading image from URL: {url}")
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        # Save to temporary file for LMDeploy
        temp_path = f"/tmp/image_{hash(url)}.jpg"
        with open(temp_path, 'wb') as f:
            f.write(response.content)

        # Load with LMDeploy's load_image function
        image = load_image(temp_path)
        print(f"Downloaded and loaded image from: {url}")
        return image
    except Exception as e:
        print(f"Error downloading image from URL {url}: {e}")
        return None


def decode_base64_image(base64_string):
    """Decode base64 image string and save for LMDeploy"""
    try:
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
        print(f"Decoded and loaded base64 image")
        return image
    except Exception as e:
        print(f"Error decoding base64 image: {e}")
        return None


def process_batch_requests(batch_data):
    """Convert batch requests to LMDeploy format"""
    processed_prompts = []

    for request in batch_data:
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
                print(f"Created multimodal prompt with image")
            else:
                # Text-only request
                processed_prompts.append(combined_text)

        elif "prompt" in request:
            # Handle simple prompt format
            processed_prompts.append(str(request["prompt"]))
        else:
            # Fallback
            processed_prompts.append(str(request))

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
        input_data = job["input"]
        print(f"Received input data: {input_data}")

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

        print(f"Processing {len(batch_requests)} batch requests with LMDeploy")

        # Process requests into LMDeploy format
        processed_prompts = process_batch_requests(batch_requests)

        print(f"Processed {len(processed_prompts)} prompts for LMDeploy")
        for i, prompt in enumerate(processed_prompts):
            if isinstance(prompt, tuple):
                print(f"Prompt {i}: Multimodal - {prompt[0][:100]}...")
            else:
                print(f"Prompt {i}: Text only - {str(prompt)[:100]}...")

        # Create generation configuration
        gen_config = create_generation_config(batch_requests)

        # Generate responses using LMDeploy batch inference
        print(f"Generating responses with LMDeploy batch inference...")
        start_time = time.time()

        # LMDeploy batch inference
        responses = pipe(processed_prompts, gen_config=gen_config)

        generation_time = time.time() - start_time
        print(f"LMDeploy batch generation completed in {generation_time:.2f} seconds")

        # Format results
        results = []
        for i, response in enumerate(responses):
            # LMDeploy response format
            if hasattr(response, 'text'):
                generated_text = response.text
            else:
                generated_text = str(response)

            print(f"Generated text {i}: '{generated_text[:100]}...'")

            result = {
                "index": i,
                "text": generated_text,
                "finish_reason": "stop",  # LMDeploy doesn't provide detailed finish reasons
                "prompt_tokens": None,  # Would need to calculate separately
                "completion_tokens": None  # Would need to calculate separately
            }
            results.append(result)

        print(f"Successfully generated {len(results)} results with LMDeploy")
        print(f"Average time per request: {generation_time / len(results):.4f} seconds")
        print(f"Requests per second: {len(results) / generation_time:.2f}")

        return {"results": results}

    except Exception as e:
        print(f"Error in LMDeploy handler: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})