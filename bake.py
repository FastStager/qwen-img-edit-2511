import torch
import math
import os
import shutil
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPlusPipeline
from huggingface_hub import snapshot_download, hf_hub_download

BASE_MODEL_DIR = "/home/user/app/models/baked_model"
LORA_DIR = "/home/user/app/models/lora"
VL_DIR = "/home/user/app/models/Qwen3-VL-8B-Instruct"

SCHEDULER_CONFIG = {
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

LIGHTNING_LORA = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
ANGLES_LORA = "qwen-image-edit-2511-multiple-angles-lora.safetensors"

def bake():
    os.makedirs(BASE_MODEL_DIR, exist_ok=True)
    os.makedirs(LORA_DIR, exist_ok=True)
    os.makedirs(VL_DIR, exist_ok=True)

    print("--- [BAKE] Downloading Qwen-VL (Rewriter) ... ---")
    snapshot_download("Qwen/Qwen3-VL-8B-Instruct", local_dir=VL_DIR)

    print("--- [BAKE] Loading Base Edit Model... ---")
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(SCHEDULER_CONFIG)

    pipe = QwenImageEditPlusPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit-2511",
        scheduler=scheduler,
        torch_dtype=torch.bfloat16
    )

    # --- Download LoRAs ---
    print("--- [BAKE] Downloading Lightning LoRA... ---")
    hf_hub_download(
        repo_id="lightx2v/Qwen-Image-Edit-2511-Lightning",
        filename=LIGHTNING_LORA,
        local_dir=LORA_DIR
    )

    print("--- [BAKE] Downloading Multi-Angle LoRA... ---")
    hf_hub_download(
        repo_id="fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA",
        filename=ANGLES_LORA,
        local_dir=LORA_DIR
    )

    # --- Fuse LoRAs permanently into base weights ---
    # This eliminates all per-step LoRA overhead at inference time.
    # The saved model IS the fused model — no LoRA loading needed at runtime.
    print("--- [BAKE] Fusing LoRAs into base weights... ---")
    pipe.load_lora_weights(LORA_DIR, weight_name=LIGHTNING_LORA, adapter_name="lightning")
    pipe.load_lora_weights(LORA_DIR, weight_name=ANGLES_LORA, adapter_name="angles")
    pipe.set_adapters(["lightning", "angles"], adapter_weights=[1.0, 1.0])
    pipe.fuse_lora()
    pipe.unload_lora_weights()

    print("--- [BAKE] Saving Fused Model... ---")
    pipe.save_pretrained(BASE_MODEL_DIR, safe_serialization=True, max_shard_size="100GB")

    # LoRA files no longer needed — weights are baked in
    shutil.rmtree(LORA_DIR, ignore_errors=True)

    print("--- [BAKE] Done. ---")

if __name__ == "__main__":
    bake()
