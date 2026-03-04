#!/bin/bash
# ─── Vast.ai H100 Pod Setup ───
# Run on PyTorch template pod before bake_trt.py
# Usage: bash setup_vast.sh
set -e

echo "=== [1/3] Installing ML dependencies ==="
pip install --no-cache-dir \
    git+https://github.com/huggingface/diffusers.git \
    'transformers>=4.49.0' \
    torchvision \
    accelerate \
    safetensors \
    sentencepiece \
    peft \
    protobuf \
    pyvips \
    'torchao==0.7.0' \
    qwen_vl_utils \
    huggingface_hub \
    runpod

echo "=== [2/3] Installing TensorRT stack ==="
pip install --no-cache-dir \
    torch-tensorrt \
    tensorrt

# ModelOpt for FP8 quantization (Hopper only, optional)
echo "=== [3/3] Installing ModelOpt (FP8, optional) ==="
pip install --no-cache-dir nvidia-modelopt 2>/dev/null \
    && echo "--- nvidia-modelopt installed ---" \
    || echo "--- nvidia-modelopt failed (FP8 will be skipped) ---"

echo ""
echo "=== Setup complete ==="
echo "Next: python bake_trt.py"
echo ""
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader
