import os
import time
import base64
import runpod
import requests
from vllm import LLM, SamplingParams
from PIL import Image
import io
print(f"vLLM version: {vllm.__version__}")
# Global variables
llm = None


def initialize_llm(input_data):
    global llm

    if not llm:
        print("Initializing VLLM...")
        start_time = time.time()

        # Get engine args from environment or input
        engine_args = {}

        # Load from environment variables
        for key, value in os.environ.items():
            if key.startswith("VLLM_"):
                param_name = key.replace("VLLM_", "").lower()

                # Convert boolean strings
                if value.lower() == "true":
                    engine_args[param_name] = True
                elif value.lower() == "false":
                    engine_args[param_name] = False
                # Special handling for JSON values
                elif param_name == "limit_mm_per_prompt":
                    import json
                    try:
                        engine_args[param_name] = json.loads(value)
                        print(f"Parsed {param_name}: {engine_args[param_name]}")
                    except json.JSONDecodeError:
                        print(f"Warning: Could not parse {key} as JSON: {value}")
                        engine_args[param_name] = value
                else:
                    engine_args[param_name] = value

        print(f"Final engine args: {engine_args}")

        # Override with input args if provided
        if "engine_args" in input_data:
            engine_args.update(input_data["engine_args"])

        llm = LLM(**engine_args)
        print('─' * 20,
              "--- VLLM engine and model cold start initialization took %s seconds ---" % (time.time() - start_time),
              '─' * 20, "*Unfortunately, this time is being billed.", sep=os.linesep)


def download_image_from_url(url):
    """Download image from URL and return PIL Image at full quality"""
    try:
        print(f"Downloading image from URL: {url}")
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        # Create PIL Image directly from response content without any compression
        image = Image.open(io.BytesIO(response.content))
        print(f"Downloaded image size: {image.size}, mode: {image.mode}")
        return image
    except Exception as e:
        print(f"Error downloading image from URL {url}: {e}")
        return None


def decode_base64_image(base64_string):
    """Decode base64 image string to PIL Image"""
    try:
        # Remove data URL prefix if present
        if base64_string.startswith('data:'):
            base64_string = base64_string.split(',')[1]

        # Decode base64
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        return image
    except Exception as e:
        print(f"Error decoding base64 image: {e}")
        return None


def process_batch_requests(batch_data):
    """Convert batch requests to proper format for InternVL3"""
    processed_inputs = []

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

            # For InternVL3, if there are images, add <image> token
            if images:
                # InternVL3 expects <image> token at the beginning
                final_prompt = f"<image>\n{combined_text}"
                processed_inputs.append({
                    "prompt": final_prompt,
                    "multi_modal_data": {"image": images[0]}  # Use first image
                })
                print(f"Created multimodal input with image size: {images[0].size}")
            else:
                # Text-only request
                processed_inputs.append({"prompt": combined_text})

        elif "prompt" in request:
            # Handle simple prompt format
            processed_inputs.append({"prompt": str(request["prompt"])})
        else:
            # Fallback
            processed_inputs.append({"prompt": str(request)})

    return processed_inputs


def create_sampling_params(batch_data):
    """Create sampling parameters from first request - only use provided params"""
    first_request = batch_data[0] if batch_data else {}

    # Only include parameters that are explicitly provided
    params = {}

    if "max_tokens" in first_request:
        params["max_tokens"] = first_request["max_tokens"]
    if "temperature" in first_request:
        params["temperature"] = first_request["temperature"]
    if "top_p" in first_request:
        params["top_p"] = first_request["top_p"]
    if "top_k" in first_request:
        params["top_k"] = first_request["top_k"]
    if "repetition_penalty" in first_request:
        params["repetition_penalty"] = first_request["repetition_penalty"]
    if "stop" in first_request:
        params["stop"] = first_request["stop"]

    print(f"Sampling params (only provided): {params}")  # Debug print

    return SamplingParams(**params) if params else SamplingParams()


def handler(job):
    try:
        input_data = job["input"]

        print(f"Received input data: {input_data}")  # Debug print

        # Initialize LLM
        initialize_llm(input_data)

        # Handle prewarm
        if "prewarm" in input_data:
            return {"warm": True}

        # Get batch requests
        if "batch" not in input_data:
            return {"error": "Expected 'batch' key with list of requests"}

        batch_requests = input_data["batch"]

        if not batch_requests:
            return {"error": "Batch requests list is empty"}

        print(f"Processing {len(batch_requests)} batch requests")

        # Process requests
        processed_inputs = process_batch_requests(batch_requests)

        # Debug: Print processed inputs
        print(f"Processed {len(processed_inputs)} inputs")
        for i, inp in enumerate(processed_inputs):
            if "multi_modal_data" in inp:
                print(f"Input {i}: Multimodal - {inp['prompt'][:100]}...")
            else:
                print(f"Input {i}: Text only - {inp.get('prompt', '')[:100]}...")

        # Create sampling parameters
        sampling_params = create_sampling_params(batch_requests)

        # Generate responses
        print(f"Generating responses with sampling params: {sampling_params}")

        # For InternVL3, we pass the processed inputs directly
        outputs = llm.generate(processed_inputs, sampling_params)

        # Format results
        results = []
        for i, output in enumerate(outputs):
            generated_text = output.outputs[0].text
            print(f"Generated text {i}: '{generated_text}'")  # Debug print

            result = {
                "index": i,
                "text": generated_text,
                "finish_reason": output.outputs[0].finish_reason,
                "prompt_tokens": len(output.prompt_token_ids) if hasattr(output, 'prompt_token_ids') else None,
                "completion_tokens": len(output.outputs[0].token_ids) if hasattr(output.outputs[0],
                                                                                 'token_ids') else None
            }
            results.append(result)

        print(f"Successfully generated {len(results)} results")
        return {"results": results}

    except Exception as e:
        print(f"Error in handler: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})