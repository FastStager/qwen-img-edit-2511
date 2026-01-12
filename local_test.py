import torch
import numpy as np
from PIL import Image
import math
import json
import gc
import os
import sys
import time  

IMAGE_PATH = "/workspace/image14.png"
OUTPUT_PATH = "output_plus_test.png"
RAW_PROMPT = "Add furnishings and accessories to this room as an interior designer would do for a real estate staging. The generated image shall have the exact same dimensions as the original image and architectural details. Respect doorways and windows and make sure they are consistent with the source image and not blocked by furniture. Use cute accessories and with appropriate wall space, add smart simple graphic paintings. Use neutral colors with light colored accents to match the colors of the room. Give the area an attractive glow."
USE_REWRITER = False  
SEED = 42
STEPS = 4
CFG = 1.0

try:
    from qwenimage.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline
except ImportError:
    try:
        from diffusers import QwenImageEditPlusPipeline
    except ImportError:
        print("CRITICAL ERROR: Could not import 'QwenImageEditPlusPipeline'.")
        print("Ensure the 'qwenimage' folder is in this directory.")
        exit(1)

try:
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    from diffusers import FlowMatchEulerDiscreteScheduler
except ImportError:
    print("Missing libs. Run: pip install torch torchvision diffusers transformers accelerate safetensors sentencepiece peft qwen_vl_utils protobuf")
    exit(1)

SYSTEM_PROMPT = '''
# Edit Instruction Rewriter
You are a professional edit instruction rewriter. Your task is to generate a precise, concise, and visually achievable professional-level edit instruction based on the user-provided instruction and the image to be edited.  
... (Prompt Truncated for brevity, kept same as original) ...
'''

def get_1mp_dimensions(width, height):
    """Calculates 1MP dimensions (1024x1024 equivalent), snapped to 64px."""
    target_area = 1024 * 1024
    aspect = width / height
    new_w = math.sqrt(target_area * aspect)
    new_h = new_w / aspect
    new_w = int(round(new_w / 64) * 64)
    new_h = int(round(new_h / 64) * 64)
    return new_w, new_h

def flush():
    gc.collect()
    torch.cuda.empty_cache()

def main():
    total_start_time = time.time()
    
    dtype = torch.bfloat16
    print(f"Running Local Verification (Plus Pipeline).")

    if not os.path.exists(IMAGE_PATH):
        print(f"Error: Image not found at {IMAGE_PATH}")
        return
        
    original_image = Image.open(IMAGE_PATH).convert("RGB")
    
    w, h = get_1mp_dimensions(original_image.width, original_image.height)
    print(f"Resizing input to {w}x{h}")
    pil_image = original_image.resize((w, h), Image.LANCZOS)

    final_prompt = RAW_PROMPT

    if USE_REWRITER:
        print("\n[Phase 1] Loading Qwen3-VL (Rewriter)...")
        try:
            model_vl = Qwen3VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen3-VL-8B-Instruct", torch_dtype=dtype, device_map="auto"
            )
            processor_vl = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
            
            prompt_text = f"{SYSTEM_PROMPT}\n\nUser Input: {RAW_PROMPT}\n\nRewritten Prompt:"
            messages = [{"role": "user", "content": [{"type": "image", "image": pil_image}, {"type": "text", "text": prompt_text}]}]
            
            inputs = processor_vl.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(model_vl.device)
            
            generated_ids = model_vl.generate(**inputs, max_new_tokens=512)
            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            output_text = processor_vl.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            
            if '"Rewritten"' in output_text:
                try:
                    start = output_text.find('{')
                    end = output_text.rfind('}') + 1
                    res_json = json.loads(output_text[start:end])
                    final_prompt = res_json.get('Rewritten', RAW_PROMPT)
                except: pass
            else:
                final_prompt = output_text.strip().replace("```json", "").replace("```", "").replace("\n", " ")
            
            print(f"Rewritten: {final_prompt}")
            
            del model_vl
            del processor_vl
            flush()
            print("VLM Unloaded.")
            
        except Exception as e:
            print(f"Rewrite failed: {e}")

    print("\n[Phase 2] Loading Qwen-Image-Edit-2511...")
    
    load_start_time = time.time()

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

    pipe = QwenImageEditPlusPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit-2511",
        scheduler=scheduler,
        torch_dtype=dtype,
        low_cpu_mem_usage=True
    )

    print("Loading Lightning LoRA...")
    pipe.load_lora_weights(
        "lightx2v/Qwen-Image-Edit-2511-Lightning", 
        weight_name="Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
    )
    
    try:
        pipe.fuse_lora()
    except Exception as e:
        print(f"Fusion skipped (Low RAM): {e}")

    print("Enabling CPU Offload...")
    pipe.enable_model_cpu_offload()

    torch.cuda.synchronize()
    load_end_time = time.time()
    load_duration = load_end_time - load_start_time

    print(f"Running Inference (Seed: {SEED})...")
    
    torch.cuda.synchronize() 
    inf_start_time = time.time()
    
    output = pipe(
        image=[pil_image],
        prompt=final_prompt,
        negative_prompt=" ",
        num_inference_steps=STEPS,
        true_cfg_scale=CFG,        
        guidance_scale=CFG,        
        generator=torch.Generator("cuda").manual_seed(SEED),
        height=h,
        width=w
    ).images[0]

    torch.cuda.synchronize()
    inf_end_time = time.time()
    inf_duration = inf_end_time - inf_start_time

    print(f"Saving to {OUTPUT_PATH}")
    output.save(OUTPUT_PATH)
    
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    
    print("\n" + "="*40)
    print(f"BENCHMARK REPORT ({torch.cuda.get_device_name(0)})")
    print("="*40)
    print(f"Model Load Time : {load_duration:.4f} sec")
    print(f"Inference Time  : {inf_duration:.4f} sec")
    print(f"Total Execution : {total_duration:.4f} sec")
    print("="*40 + "\n")
    print("Done.")

if __name__ == "__main__":
    main()