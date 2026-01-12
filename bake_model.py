import torch
import math
import os
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPlusPipeline

OUTPUT_DIR = "baked_model"

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

print("Loading Base Model...")
scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
pipe = QwenImageEditPlusPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2511",
    scheduler=scheduler,
    torch_dtype=torch.bfloat16
)

print("Loading LoRA...")
pipe.load_lora_weights(
    "lightx2v/Qwen-Image-Edit-2511-Lightning", 
    weight_name="Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
)

print("Fusing...")
pipe.fuse_lora()

print(f"Saving to {OUTPUT_DIR}...")
pipe.save_pretrained(OUTPUT_DIR, safe_serialization=True)
print("Done.")