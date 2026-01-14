import runpod
import torch
from PIL import Image
import base64
import io
import math
import gc
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPlusPipeline

import rewriter

BAKED_MODEL_PATH = "/home/user/app/models/baked_model"
LORA_DIR = "/home/user/app/models/lora"
LIGHTNING_LORA = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
ANGLES_LORA = "qwen-image-edit-2511-multiple-angles-lora.safetensors"

pipe = None

AZIMUTH_MAP = {0: "front view", 45: "front-right quarter view", 90: "right side view", 135: "back-right quarter view", 180: "back view", 225: "back-left quarter view", 270: "left side view", 315: "front-left quarter view"}
ELEVATION_MAP = {-30: "low-angle shot", 0: "eye-level shot", 30: "elevated shot", 60: "high-angle shot"}
DISTANCE_MAP = {0.6: "close-up", 1.0: "medium shot", 1.4: "medium shot", 1.8: "wide shot"}

def snap_to_nearest(value, options):
    return min(options, key=lambda x: abs(x - value))

def build_camera_prompt(azimuth, elevation, distance):
    if azimuth is None and elevation is None and distance is None:
        return ""
    az = snap_to_nearest(float(azimuth or 0), list(AZIMUTH_MAP.keys()))
    el = snap_to_nearest(float(elevation or 0), list(ELEVATION_MAP.keys()))
    di = snap_to_nearest(float(distance or 1.0), list(DISTANCE_MAP.keys()))
    return f"<sks> {AZIMUTH_MAP[az]} {ELEVATION_MAP[el]} {DISTANCE_MAP[di]}"

def get_1mp_dimensions(width, height):
    target_area = 1024 * 1024
    aspect = width / height
    new_w = math.sqrt(target_area * aspect)
    new_h = new_w / aspect
    new_w = int(round(new_w / 64) * 64)
    new_h = int(round(new_h / 64) * 64)
    return new_w, new_h

def load_edit_model():
    global pipe
    if pipe is not None: return

    print("--- Loading Edit Model & Adapters ---")
    dtype = torch.bfloat16
    
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
        BAKED_MODEL_PATH,
        scheduler=scheduler,
        torch_dtype=dtype,
        local_files_only=True
    ).to("cuda")

    pipe.load_lora_weights(LORA_DIR, weight_name=LIGHTNING_LORA, adapter_name="lightning")
    pipe.load_lora_weights(LORA_DIR, weight_name=ANGLES_LORA, adapter_name="angles")
    pipe.set_adapters(["lightning", "angles"], adapter_weights=[1.0, 1.0])
    
    print("--- Models Loaded ---")

def base64_to_pil(b64): 
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")

def pil_to_base64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def handler(job):
    gc.collect()
    torch.cuda.empty_cache()
    
    global pipe
    if pipe is None: load_edit_model()
    
    job_input = job.get('input', {})
    
    images_in = job_input.get('images', [])
    if not images_in and job_input.get('image'):
        images_in = [job_input.get('image')]
        
    if not images_in: return {"error": "No images provided"}
    
    try:
        pil_images = [base64_to_pil(i) for i in images_in]
    except Exception as e:
        return {"error": f"Image decode failed: {str(e)}"}

    raw_prompt = job_input.get('prompt', "")
    do_rewrite = job_input.get('rewrite_prompt', False)
    cam_prompt = build_camera_prompt(job_input.get('azimuth'), job_input.get('elevation'), job_input.get('distance'))

    prompts = []
    rewritten_log = []
    
    is_batch = isinstance(raw_prompt, list)
    
    if is_batch:
        if len(raw_prompt) != len(pil_images):
            return {"error": "Batch prompt length mismatch"}
        for idx, p in enumerate(raw_prompt):
            final_p = p
            if do_rewrite:
                final_p = rewriter.polish_prompt(p, [pil_images[idx]])
                rewritten_log.append(final_p)
            prompts.append(f"{cam_prompt} {final_p}".strip())
    else:
        final_p = raw_prompt
        if do_rewrite:
            final_p = rewriter.polish_prompt(raw_prompt, pil_images)
            rewritten_log.append(final_p)
        prompts = [f"{cam_prompt} {final_p}".strip()]

    w, h = get_1mp_dimensions(pil_images[0].width, pil_images[0].height)
    if job_input.get('height') and job_input.get('width'):
        h = int(job_input['height'])
        w = int(job_input['width'])

    proc_images = [img.resize((w, h), Image.LANCZOS) for img in pil_images]

    seed = job_input.get('seed', 42)
    steps = int(job_input.get('num_inference_steps', 4))
    cfg = float(job_input.get('guidance_scale', 1.0))
    generator = torch.Generator("cuda").manual_seed(seed)
    
    output_images = []
    try:
        with torch.no_grad():
            if is_batch:
                output_images = pipe(
                    image=proc_images, prompt=prompts, num_inference_steps=steps,
                    guidance_scale=cfg, generator=generator, height=h, width=w
                ).images
            else:
                output_images = pipe(
                    image=proc_images, prompt=prompts[0], num_inference_steps=steps,
                    guidance_scale=cfg, generator=generator, height=h, width=w
                ).images
    except Exception as e:
        return {"error": f"Inference failed: {str(e)}"}
    finally:
        gc.collect()
        torch.cuda.empty_cache()

    return {
        "images": [pil_to_base64(img) for img in output_images],
        "seed": seed,
        "final_prompts": prompts,
        "rewritten_log": rewritten_log if do_rewrite else None
    }

if __name__ == "__main__":
    load_edit_model()
    runpod.serverless.start({"handler": handler})