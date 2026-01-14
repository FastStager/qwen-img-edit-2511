# Qwen Camera Edit 2511 API Documentation

This documentation describes the interface and parameters for the Qwen Camera Edit 2511 serverless endpoint. This model supports instruction-based image editing, camera angle manipulation, and automated prompt refinement via an integrated vision-language rewriter.

## Endpoint Overview

The endpoint accepts base64 encoded images and text instructions. It utilizes the Qwen-Image-Edit-2511 base model enhanced with Lightning and Multi-Angle LoRA adapters for high-speed, high-quality generation.

## Request Parameters

The `input` object in your JSON payload supports the following fields:

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `image` | String | null | Base64 encoded string of the input image. |
| `images` | Array | [] | A list of base64 encoded strings for batch processing or multi-image tasks. |
| `prompt` | String/Array | "" | The editing instruction. Can be a list of strings if processing a batch. |
| `rewrite_prompt` | Boolean | false | If true, uses Qwen3-VL to optimize the instruction for better visual results. |
| `azimuth` | Integer | null | Camera horizontal angle. Snaps to: 0, 45, 90, 135, 180, 225, 270, 315. |
| `elevation` | Integer | null | Camera vertical angle. Snaps to: -30 (low), 0 (eye), 30 (elevated), 60 (high). |
| `distance` | Float | null | Camera distance/zoom. Snaps to: 0.6 (close), 1.0 (medium), 1.4, 1.8 (wide). |
| `seed` | Integer | 42 | Random seed for generation reproducibility. |
| `num_inference_steps`| Integer | 4 | Number of diffusion steps. Recommended 4-8 for Lightning adapter. |
| `guidance_scale` | Float | 1.0 | Classifier-free guidance scale. Values 1.0 to 1.5 are recommended. |
| `width` | Integer | null | Target width. If null, automatically calculated to 1MP area. |
| `height` | Integer | null | Target height. If null, automatically calculated to 1MP area. |

---

## Usage Modes

### 1. Camera Control

By providing `azimuth`, `elevation`, and `distance`, the system injects specialized camera tokens (`<sks>`) into the prompt. This allows you to shift the perspective of the original scene.

### 2. Instruction Rewriting

Setting `rewrite_prompt: true` activates a Vision-Language Model (Qwen3-VL) that analyzes the input image and your prompt. It transforms vague instructions (e.g., "make it better") into precise visual descriptions (e.g., "Enhance the lighting, add high-contrast shadows, and sharpen the textures").

### 3. Batch Processing

If `images` is a list and `prompt` is a list of the same length, the endpoint processes them as a batch, applying the corresponding prompt to each image respectively.

---

## Examples

### Shell (cURL)

This script encodes a local image and sends it to the Synchronous RunPod API.

```bash
export RUNPOD_API_KEY="your_api_key"
export ENDPOINT_ID="your_endpoint_id"
export IMAGE_PATH="./input_scene.png"

# Encode image to base64
BASE64_IMAGE=$(base64 -i "$IMAGE_PATH")

# Construct payload
cat <<EOF > payload.json
{
  "input": {
    "image": "$BASE64_IMAGE",
    "prompt": "Change the time of day to sunset with warm orange lighting",
    "azimuth": 45,
    "elevation": 30,
    "rewrite_prompt": true,
    "num_inference_steps": 4,
    "guidance_scale": 1.2,
    "seed": 1337
  }
}
EOF

# Send request
curl -s -X POST \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -d @payload.json \
  "https://api.runpod.ai/v2/${ENDPOINT_ID}/runsync" \
  | jq -r '.output.images[0]' | base64 --decode > output_result.png
```

### Python

Utilizing the `requests` library for a more structured implementation.

```python
import requests
import base64
import json

def edit_image(image_path, prompt, azimuth=0):
    with open(image_path, "rb") as f:
        img_str = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "input": {
            "image": img_str,
            "prompt": prompt,
            "azimuth": azimuth,
            "rewrite_prompt": True,
            "num_inference_steps": 6,
            "guidance_scale": 1.0
        }
    }

    headers = {
        "Authorization": "Bearer YOUR_RUNPOD_KEY",
        "Content-Type": "application/json"
    }

    response = requests.post(
        "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync",
        headers=headers,
        json=payload
    )

    result = response.json()
    if "output" in result:
        output_b64 = result["output"]["images"][0]
        with open("output.png", "wb") as f:
            f.write(base64.b64decode(output_b64))
        print("Image processed successfully.")
    else:
        print(f"Error: {result}")

edit_image("room.jpg", "Add a modern sofa and a coffee table", azimuth=90)
```

# Qwen Camera Edit 2511 API Documentation

This documentation describes the interface and parameters for the Qwen Camera Edit 2511 serverless endpoint. This model supports instruction-based image editing, camera angle manipulation, and automated prompt refinement.

## Endpoint Parameters

The `input` object in your JSON payload supports the following fields:

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `image` | String | null | Base64 encoded string of the input image. |
| `images` | Array | [] | A list of base64 encoded strings for batching or multi-image references. |
| `prompt` | String/Array | "" | The editing instruction. Use an Array for batch processing. |
| `rewrite_prompt` | Boolean | false | Uses Qwen3-VL to analyze images and optimize instructions. |
| `azimuth` | Integer | 0 | Camera horizontal angle (0, 45, 90, 135, 180, 225, 270, 315). |
| `elevation` | Integer | 0 | Camera vertical angle (-30, 0, 30, 60). |
| `distance` | Float | 1.0 | Camera zoom level (0.6, 1.0, 1.4, 1.8). |
| `seed` | Integer | 42 | Random seed for generation. |
| `num_inference_steps`| Integer | 4 | Recommended 4-8 steps for the Lightning adapter. |
| `guidance_scale` | Float | 1.0 | Recommended 1.0-1.5. |

---

## Batch Processing

To process multiple independent images in a single request, provide an array of base64 strings in the `images` field and an array of strings in the `prompt` field. The lengths of both arrays must match.

### Batch Request Example
```json
{
  "input": {
    "images": ["base64_img_1...", "base64_img_2..."],
    "prompt": [
      "Add a red hat to the person",
      "Change the background to a snowy mountain"
    ],
    "num_inference_steps": 4,
    "seed": 42
  }
}
```

---

## Multi-Image Reference (Stylization/Context)

When you provide multiple images but only a **single string** as the prompt, the system treats the images as a collective context. This is specifically powerful when `rewrite_prompt` is set to `true`, as the rewriter will analyze both images to understand style or subject migration.

### Style Reference Example
```json
{
  "input": {
    "images": ["base64_source_content...", "base64_style_reference..."],
    "prompt": "Apply the artistic style and color palette of image 2 to image 1",
    "rewrite_prompt": true,
    "guidance_scale": 1.2
  }
}
```

---

## Camera Perspective Manipulation

The endpoint uses specialized tokens to alter the camera view of the input image. You can combine these with standard prompts.

### Camera Control Example
```json
{
  "input": {
    "image": "base64_encoded_image...",
    "prompt": "A futuristic city street",
    "azimuth": 270,
    "elevation": 60,
    "distance": 1.8,
    "num_inference_steps": 6
  }
}
```

---

## Python Implementation Snippet

```python
import requests
import base64

def run_batch_edit(image_paths, prompts):
    encoded_images = []
    for path in image_paths:
        with open(path, "rb") as f:
            encoded_images.append(base64.b64encode(f.read()).decode("utf-8"))

    payload = {
        "input": {
            "images": encoded_images,
            "prompt": prompts,
            "num_inference_steps": 4,
            "guidance_scale": 1.0
        }
    }

    response = requests.post(
        "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync",
        headers={"Authorization": "Bearer YOUR_API_KEY"},
        json=payload
    )
    
    return response.json()["output"]["images"]

# Usage for batch
results = run_batch_edit(["img1.jpg", "img2.jpg"], ["Make it sunset", "Make it rainy"])
```

## Technical Notes

- **Dimension Handling:** If `width` and `height` are not provided, the model automatically scales the input to a 1-megapixel area (approx. 1024x1024) while preserving aspect ratio.
- **Rewriter Logic:** When `rewrite_prompt` is active, the Vision-Language model prepends a professional instruction set to ensure the base model understands complex modifications like object deletion or subtle facial edits.
- **Performance:** The Lightning LoRA is fused/loaded to allow high-quality edits in as few as 4 steps, significantly reducing latency.
