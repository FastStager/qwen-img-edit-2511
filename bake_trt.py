"""
Two-stage model preparation for Vast.ai H100 pod.

Stage 1: Download base model + LoRAs + rewriter, fuse LoRAs
Stage 2: Compile transformer to TensorRT engine (requires GPU)

Usage:
    python bake_trt.py                # Full pipeline (download + fuse + TRT)
    python bake_trt.py --download     # Stage 1 only (no GPU needed)
    python bake_trt.py --trt          # Stage 2 only (needs GPU, models already downloaded)

Paths (Vast.ai network storage):
    /workspace/models/baked_model/          Fused LoRA model
    /workspace/models/trt_engine/           Compiled TRT transformer
    /workspace/models/Qwen3-VL-8B-Instruct/ Rewriter model
"""

import torch
import math
import os
import sys
import json
import time
import gc
import shutil
from PIL import Image
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPlusPipeline
from huggingface_hub import snapshot_download, hf_hub_download

# ─── Paths ───────────────────────────────────────────────
MODELS_DIR      = os.environ.get("MODELS_DIR", "/workspace/models")
BAKED_MODEL_DIR = f"{MODELS_DIR}/baked_model"
LORA_DIR        = f"{MODELS_DIR}/lora"
VL_DIR          = f"{MODELS_DIR}/Qwen3-VL-8B-Instruct"
TRT_DIR         = f"{MODELS_DIR}/trt_engine"

# ─── LoRA files ──────────────────────────────────────────
LIGHTNING_LORA = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
ANGLES_LORA    = "qwen-image-edit-2511-multiple-angles-lora.safetensors"

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


# ═════════════════════════════════════════════════════════
# STAGE 1: Download + Fuse (CPU ok)
# ═════════════════════════════════════════════════════════

def stage_download_and_fuse():
    for d in [BAKED_MODEL_DIR, LORA_DIR, VL_DIR]:
        os.makedirs(d, exist_ok=True)

    print("=" * 60)
    print("STAGE 1: Download & Fuse LoRAs")
    print("=" * 60)

    # ── Rewriter ──
    if os.path.exists(f"{VL_DIR}/config.json"):
        print("\n--- Qwen3-VL already downloaded, skipping ---")
    else:
        print("\n--- Downloading Qwen3-VL-8B-Instruct (rewriter) ---")
        snapshot_download("Qwen/Qwen3-VL-8B-Instruct", local_dir=VL_DIR)

    # ── Check if already fused ──
    if os.path.exists(f"{BAKED_MODEL_DIR}/model_index.json"):
        print("\n--- Fused model already exists, skipping ---")
        return

    # ── Base model ──
    print("\n--- Loading Qwen-Image-Edit-2511 ---")
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(SCHEDULER_CONFIG)
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit-2511",
        scheduler=scheduler,
        torch_dtype=torch.bfloat16,
    )

    # ── LoRAs ──
    print("\n--- Downloading Lightning LoRA ---")
    hf_hub_download(
        repo_id="lightx2v/Qwen-Image-Edit-2511-Lightning",
        filename=LIGHTNING_LORA,
        local_dir=LORA_DIR,
    )
    print("--- Downloading Multi-Angle LoRA ---")
    hf_hub_download(
        repo_id="fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA",
        filename=ANGLES_LORA,
        local_dir=LORA_DIR,
    )

    # ── Fuse ──
    print("\n--- Fusing LoRAs into base weights ---")
    pipe.load_lora_weights(LORA_DIR, weight_name=LIGHTNING_LORA, adapter_name="lightning")
    pipe.load_lora_weights(LORA_DIR, weight_name=ANGLES_LORA, adapter_name="angles")
    pipe.set_adapters(["lightning", "angles"], adapter_weights=[1.0, 1.0])
    pipe.fuse_lora()
    pipe.unload_lora_weights()

    # ── Save ──
    print("\n--- Saving fused model ---")
    pipe.save_pretrained(BAKED_MODEL_DIR, safe_serialization=True, max_shard_size="100GB")

    shutil.rmtree(LORA_DIR, ignore_errors=True)
    del pipe
    gc.collect()

    print("\nSTAGE 1 complete.")


# ═════════════════════════════════════════════════════════
# STAGE 2: Compile TensorRT (requires H100 GPU)
# ═════════════════════════════════════════════════════════

def stage_compile_trt():
    import torch_tensorrt

    os.makedirs(TRT_DIR, exist_ok=True)

    gpu_name = torch.cuda.get_device_name(0)
    cc = torch.cuda.get_device_capability()

    print("=" * 60)
    print("STAGE 2: Compile TensorRT Engine")
    print(f"  GPU:     {gpu_name}")
    print(f"  CC:      {cc[0]}.{cc[1]}")
    print(f"  TRT ver: {torch_tensorrt.__version__}")
    print("=" * 60)

    if cc[0] < 8:
        print("ERROR: Need Ampere (cc 8.x) or Hopper (cc 9.x)")
        sys.exit(1)

    # ── Load fused pipeline ──
    print("\n--- Loading fused model onto GPU ---")
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(SCHEDULER_CONFIG)
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        BAKED_MODEL_DIR,
        scheduler=scheduler,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).to("cuda")

    # ── Save transformer config (needed at runtime) ──
    config_dict = {k: v for k, v in dict(pipe.transformer.config).items()}
    with open(f"{TRT_DIR}/transformer_config.json", "w") as f:
        json.dump(config_dict, f, indent=2, default=str)
    print("--- Saved transformer config ---")

    # ── FP8 quantization (Hopper only) ──
    use_fp8 = False
    if cc[0] >= 9:
        try:
            import nvidia_modelopt.torch.quantization as mtq
            print("\n--- Applying ModelOpt FP8 quantization (Hopper) ---")
            pipe.transformer = mtq.diffusers.quantize_diffusers_module(
                pipe, "transformer", mtq.FP8_DEFAULT_CONFIG
            )
            use_fp8 = True
            print("--- FP8 applied ---")
        except ImportError:
            print("--- nvidia-modelopt not installed, using FP16 ---")
        except Exception as e:
            print(f"--- FP8 failed ({e}), using FP16 ---")
    else:
        print(f"\n--- Ampere GPU (cc {cc[0]}.{cc[1]}), using FP16 ---")

    # ── Wrap transformer in TensorRT ──
    print("\n--- Creating MutableTorchTensorRTModule ---")
    enabled_precisions = {torch.float16}
    if use_fp8:
        enabled_precisions.add(torch.float8_e4m3fn)

    pipe.transformer = torch_tensorrt.MutableTorchTensorRTModule(
        pipe.transformer,
        strict=False,
        allow_complex_guards_as_runtime_asserts=True,
        enabled_precisions=enabled_precisions,
        truncate_double=True,
        immutable_weights=False,
        offload_module_to_cpu=True,
    )

    # ── Compile via warmup (this is the slow part) ──
    print("\n--- Compiling TRT engine via warmup ---")
    print("--- This takes 20-30 min on H100. Go get coffee. ---")
    dummy = Image.new("RGB", (1024, 1024), (128, 128, 128))

    t0 = time.time()
    with torch.inference_mode():
        pipe(
            image=[dummy],
            prompt="warmup for tensorrt compilation",
            num_inference_steps=4,
            height=1024,
            width=1024,
        )
    compile_time = time.time() - t0
    print(f"--- Compilation: {compile_time:.1f}s ({compile_time/60:.1f} min) ---")

    # ── Benchmark ──
    print("\n--- Benchmarking (3 runs) ---")
    times = []
    for i in range(3):
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.inference_mode():
            pipe(
                image=[dummy],
                prompt="benchmark run",
                num_inference_steps=4,
                height=1024,
                width=1024,
            )
        torch.cuda.synchronize()
        t = time.time() - t0
        times.append(t)
        print(f"  Run {i+1}: {t:.3f}s")

    avg = sum(times) / len(times)
    print(f"  Average: {avg:.3f}s")

    # ── Save engine ──
    engine_path = f"{TRT_DIR}/transformer.pt2"
    print(f"\n--- Saving TRT engine to {engine_path} ---")
    torch_tensorrt.MutableTorchTensorRTModule.save(pipe.transformer, engine_path)

    engine_size = os.path.getsize(engine_path)

    # ── Save metadata ──
    metadata = {
        "gpu": gpu_name,
        "compute_capability": f"{cc[0]}.{cc[1]}",
        "fp8": use_fp8,
        "enabled_precisions": [str(p) for p in enabled_precisions],
        "compile_time_s": round(compile_time, 1),
        "benchmark_avg_s": round(avg, 3),
        "compiled_resolution": "1024x1024",
        "inference_steps": 4,
        "engine_size_gb": round(engine_size / 1e9, 2),
        "torch_tensorrt_version": torch_tensorrt.__version__,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    with open(f"{TRT_DIR}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    del pipe, dummy
    torch.cuda.empty_cache()
    gc.collect()

    print(f"\n{'='*60}")
    print(f"STAGE 2 complete.")
    print(f"  Engine:  {engine_path} ({engine_size/1e9:.2f} GB)")
    print(f"  FP8:     {use_fp8}")
    print(f"  Avg inf: {avg:.3f}s (4 steps @ 1024x1024)")
    print(f"{'='*60}")


# ═════════════════════════════════════════════════════════

if __name__ == "__main__":
    if "--trt" in sys.argv:
        stage_compile_trt()
    elif "--download" in sys.argv:
        stage_download_and_fuse()
    else:
        stage_download_and_fuse()
        stage_compile_trt()
