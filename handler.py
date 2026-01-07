import runpod
import torch
import numpy as np
from PIL import Image
import base64
import io
import math
import json
import os
import gc
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPipeline

MODELS_DIR = "/home/user/app/models"
pipe_edit = None
model_vl = None
processor_vl = None
current_lora_state = "none"

SYSTEM_PROMPT = '''
# Edit Instruction Rewriter
You are a professional edit instruction rewriter. Your task is to generate a precise, concise, and visually achievable professional-level edit instruction based on the user-provided instruction and the image to be edited.  

Please strictly follow the rewriting rules below:

## 1. General Principles
- Keep the rewritten prompt **concise and comprehensive**. Avoid overly long sentences and unnecessary descriptive language.  
- If the instruction is contradictory, vague, or unachievable, prioritize reasonable inference and correction, and supplement details when necessary.  
- Keep the main part of the original instruction unchanged, only enhancing its clarity, rationality, and visual feasibility.  
- All added objects or modifications must align with the logic and style of the scene in the input images.  
- If multiple sub-images are to be generated, describe the content of each sub-image individually.  

## 2. Task-Type Handling Rules

### 1. Add, Delete, Replace Tasks
- If the instruction is clear (already includes task type, target entity, position, quantity, attributes), preserve the original intent and only refine the grammar.  
- If the description is vague, supplement with minimal but sufficient details (category, color, size, orientation, position, etc.). For example:  
    > Original: "Add an animal"  
    > Rewritten: "Add a light-gray cat in the bottom-right corner, sitting and facing the camera"  
- Remove meaningless instructions: e.g., "Add 0 objects" should be ignored or flagged as invalid.  
- For replacement tasks, specify "Replace Y with X" and briefly describe the key visual features of X.  

### 2. Text Editing Tasks
- All text content must be enclosed in English double quotes `" "`. Keep the original language of the text, and keep the capitalization.  
- Both adding new text and replacing existing text are text replacement tasks, For example:  
    - Replace "xx" to "yy"  
    - Replace the mask / bounding box to "yy"  
    - Replace the visual object to "yy"  
- Specify text position, color, and layout only if user has required.  
- If font is specified, keep the original language of the font.  

### 3. Human Editing Tasks
- Make the smallest changes to the given user's prompt.  
- If changes to background, action, expression, camera shot, or ambient lighting are required, please list each modification individually.
- **Edits to makeup or facial features / expression must be subtle, not exaggerated, and must preserve the subject's identity consistency.**
    > Original: "Add eyebrows to the face"  
    > Rewritten: "Slightly thicken the person's eyebrows with little change, look natural."

### 4. Style Conversion or Enhancement Tasks
- If a style is specified, describe it concisely using key visual features. For example:  
    > Original: "Disco style"  
    > Rewritten: "1970s disco style: flashing lights, disco ball, mirrored walls, vibrant colors"  
- For style reference, analyze the original image and extract key characteristics (color, composition, texture, lighting, artistic style, etc.), integrating them into the instruction.  
- **Colorization tasks (including old photo restoration) must use the fixed template:**  
  "Restore and colorize the old photo."  
- Clearly specify the object to be modified. For example:  
    > Original: Modify the subject in Picture 1 to match the style of Picture 2.  
    > Rewritten: Change the girl in Picture 1 to the ink-wash style of Picture 2 — rendered in black-and-white watercolor with soft color transitions.

### 5. Material Replacement
- Clearly specify the object and the material. For example: "Change the material of the apple to papercut style."
- For text material replacement, use the fixed template:
    "Change the material of text "xxxx" to laser style"

### 6. Logo/Pattern Editing
- Material replacement should preserve the original shape and structure as much as possible. For example:
   > Original: "Convert to sapphire material"  
   > Rewritten: "Convert the main subject in the image to sapphire material, preserving similar shape and structure"
- When migrating logos/patterns to new scenes, ensure shape and structure consistency. For example:
   > Original: "Migrate the logo in the image to a new scene"  
   > Rewritten: "Migrate the logo in the image to a new scene, preserving similar shape and structure"

### 7. Multi-Image Tasks
- Rewritten prompts must clearly point out which image's element is being modified. For example:  
    > Original: "Replace the subject of picture 1 with the subject of picture 2"  
    > Rewritten: "Replace the girl of picture 1 with the boy of picture 2, keeping picture 2's background unchanged"  
- For stylization tasks, describe the reference image's style in the rewritten prompt, while preserving the visual content of the source image.  

## 3. Rationale and Logic Check
- Resolve contradictory instructions: e.g., "Remove all trees but keep all trees" requires logical correction.
- Supplement missing critical information: e.g., if position is unspecified, choose a reasonable area based on composition (near subject, blank space, center/edge, etc.).

# Output Format Example
```json
{
   "Rewritten": "..."
}
'''

def flush():
    gc.collect()
    torch.cuda.empty_cache()

def get_1mp_dimensions(width, height):
    target_area = 1024 * 1024
    aspect = width / height
    new_w = math.sqrt(target_area * aspect)
    new_h = new_w / aspect
    new_w = int(round(new_w / 64) * 64)
    new_h = int(round(new_h / 64) * 64)
    return new_w, new_h

def load_edit_model():
    global pipe_edit, current_lora_state
    device, dtype = "cuda", torch.bfloat16
    
    scheduler_config = {
        "base_image_seq_len": 256, 
        "base_shift": math.log(3), 
        "invert_sigmas": False,
        "max_image_seq_len": 8192, 
        "max_shift": math.log(3), 
        "num_train_timesteps": 1000,
        "shift": 1.0, 
        "shift_terminal": None, 
        "stochastic_sampling": False,
        "time_shift_type": "exponential", 
        "use_beta_sigmas": False, 
        "use_dynamic_shifting": True,
        "use_exponential_sigmas": False, 
        "use_karras_sigmas": False,
    }
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)

    pipe_edit = QwenImageEditPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit-2511", 
        scheduler=scheduler, 
        torch_dtype=dtype, 
        cache_dir=MODELS_DIR, 
        local_files_only=True
    ).to(device)
    
    current_lora_state = "none"

def polish_prompt_local(original_prompt, pil_images):
    device, dtype = "cuda", torch.bfloat16
    try:
        model_vl = Qwen3VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-8B-Instruct", 
            torch_dtype=dtype, 
            device_map="auto", 
            cache_dir=MODELS_DIR, 
            local_files_only=True
        )
        processor_vl = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct", cache_dir=MODELS_DIR)

        content = [{"type": "text", "text": f"{SYSTEM_PROMPT}\n\nUser Input: {original_prompt}\n\nRewritten Prompt:"}]
        for img in pil_images:
            content.append({"type": "image", "image": img})

        messages = [{"role": "user", "content": content}]
        inputs = processor_vl.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(model_vl.device)
        generated_ids = model_vl.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor_vl.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        del model_vl
        del processor_vl
        flush()

        if '"Rewritten"' in output_text:
            try:
                start = output_text.find('{')
                end = output_text.rfind('}') + 1
                res_json = json.loads(output_text[start:end])
                return res_json.get('Rewritten', original_prompt)
            except: pass
        return output_text.strip().replace("```json", "").replace("```", "").replace("\n", " ")
    except Exception:
        flush()
        return original_prompt

def manage_lora(steps):
    global pipe_edit, current_lora_state
    
    if steps <= 4:
        target_state = "lightning_4step"
    elif steps <= 12:
        target_state = "lightning_8step"
    else:
        target_state = "full"

    if current_lora_state == target_state:
        return

    if current_lora_state != "none":
        pipe_edit.unfuse_lora()
        pipe_edit.unload_lora_weights()
    
    if target_state == "lightning_4step":
        pipe_edit.load_lora_weights(
            "lightx2v/Qwen-Image-Edit-2511-Lightning", 
            weight_name="Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors", 
            cache_dir=MODELS_DIR, 
            local_files_only=True
        )
    elif target_state == "lightning_8step":
        pipe_edit.load_lora_weights(
            "lightx2v/Qwen-Image-Lightning", 
            weight_name="Qwen-Image-Lightning-8steps-V1.0-bf16.safetensors", 
            cache_dir=MODELS_DIR, 
            local_files_only=True
        )
        
    current_lora_state = target_state

def base64_to_pil(b64): return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
def pil_to_base64(img):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def handler(job):
    global pipe_edit
    if pipe_edit is None: load_edit_model()
    job_input = job.get('input', {})

    images_b64 = job_input.get('images', [])
    if not images_b64 and job_input.get('image'): images_b64 = [job_input.get('image')]
    if not images_b64: return {"error": "No image provided"}
    
    pil_images = [base64_to_pil(i) for i in images_b64]
    
    for i in range(len(pil_images)):
        w, h = get_1mp_dimensions(pil_images[i].width, pil_images[i].height)
        pil_images[i] = pil_images[i].resize((w, h), Image.LANCZOS)

    prompt = job_input.get('prompt', "edit")
    if job_input.get('rewrite_prompt', False):
        prompt = polish_prompt_local(prompt, pil_images)

    target_w, target_h = pil_images[0].size
    
    steps = job_input.get('num_inference_steps', 4)
    use_lightning = job_input.get('use_lightning', True)
    
    if use_lightning:
        manage_lora(steps)
    else:
        manage_lora(999) 
    
    if current_lora_state == "lightning_4step":
        default_cfg = 1.0
    elif current_lora_state == "lightning_8step":
        default_cfg = 2.5
    else:
        default_cfg = 4.0
        
    cfg = float(job_input.get('true_guidance_scale', default_cfg))

    with torch.inference_mode():
        output = pipe_edit(
            image=pil_images,
            prompt=prompt,
            negative_prompt=job_input.get('negative_prompt', " "),
            num_inference_steps=steps,
            guidance_scale=cfg,
            generator=torch.Generator("cuda").manual_seed(job_input.get('seed', 42)),
            height=target_h,
            width=target_w
        ).images
        
    return {
        "images": [pil_to_base64(img) for img in output], 
        "seed": job_input.get('seed', 42), 
        "rewritten_prompt": prompt if job_input.get('rewrite_prompt', False) else None,
        "mode": current_lora_state
    }

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})