import torch
import math
import os
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPlusPipeline
from huggingface_hub import snapshot_download, hf_hub_download

BASE_MODEL_DIR = "/home/user/app/models/baked_model"
LORA_DIR = "/home/user/app/models/lora"
VL_DIR = "/home/user/app/models/Qwen3-VL-8B-Instruct"

def bake():
    os.makedirs(BASE_MODEL_DIR, exist_ok=True)
    os.makedirs(LORA_DIR, exist_ok=True)
    os.makedirs(VL_DIR, exist_ok=True)

    print("--- [BAKE] Downloading Qwen-VL (Rewriter) ... ---")
    snapshot_download("Qwen/Qwen3-VL-8B-Instruct", local_dir=VL_DIR)

    print("--- [BAKE] Loading Base Edit Model... ---")
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
        torch_dtype=torch.bfloat16
    )

    print("--- [BAKE] Saving Optimized Base Model... ---")
    pipe.save_pretrained(BASE_MODEL_DIR, safe_serialization=True, max_shard_size="100GB")

    print("--- [BAKE] Downloading Lightning LoRA... ---")
    hf_hub_download(
        repo_id="lightx2v/Qwen-Image-Edit-2511-Lightning",
        filename="Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
        local_dir=LORA_DIR
    )

    print("--- [BAKE] Downloading Multi-Angle LoRA... ---")
    hf_hub_download(
        repo_id="fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA",
        filename="qwen-image-edit-2511-multiple-angles-lora.safetensors",
        local_dir=LORA_DIR
    )

    print("--- [BAKE] Done. ---")

if __name__ == "__main__":
    bake()