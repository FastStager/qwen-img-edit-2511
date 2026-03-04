# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Serverless image editing API powered by Qwen vision-language models, deployed on RunPod. Supports instruction-based image editing, 3D camera angle manipulation via `<sks>` tokens, and optional prompt refinement using Qwen3-VL. Includes a Gradio web UI with Three.js 3D camera visualization.

## Architecture

**Production pipeline (Docker → RunPod):**
1. `bake.py` runs at Docker build time: downloads base model (`Qwen/Qwen-Image-Edit-2511`), Lightning LoRA (4-step), and Multi-Angle LoRA, permanently fuses both LoRAs into base weights, saves to `/home/user/app/models/baked_model`, then downloads `Qwen3-VL-8B-Instruct` for rewriting
2. `handler.py` is the RunPod serverless handler: loads the pre-fused model, accepts base64-encoded images + prompts, returns base64 outputs
3. `rewriter.py` provides optional prompt enhancement via Qwen3-VL — loaded on demand, moved CPU↔CUDA to share VRAM with the edit model

**Key files:**
- `handler.py` — Production serverless handler (RunPod). All optimization logic lives here.
- `rewriter.py` — Vision-language prompt polisher (Qwen3-VL-8B-Instruct). Contains the 89-line system prompt defining rewrite rules.
- `bake.py` — Build-time model preparation. Fuses LoRAs into base weights so runtime has zero LoRA overhead.
- `app.py` — Gradio web UI with Three.js 3D camera control widget. Uses `enable_model_cpu_offload()` instead of baked model.
- `local_test.py` — Standalone benchmark script for GPU testing without RunPod.

## Commands

```bash
# Build Docker image (downloads ~30GB of models, runs bake.py)
docker build -t qwen-edit .

# Run Gradio UI locally (requires GPU, downloads models from HuggingFace)
python app.py

# Run local benchmark (requires GPU, edit IMAGE_PATH in file)
python local_test.py
```

## Key Design Patterns

**Image I/O:** Uses PyVIPS (C library) for decode+resize in a single pipeline with zero intermediate copies, and for output encoding — bypasses PIL's slow encoder. Batch I/O is parallelized via `ThreadPoolExecutor(max_workers=4)`.

**GPU optimizations in handler.py** (applied in order during model load):
- TF32 matmul/cuDNN, Flash SDPA + mem-efficient SDPA (math SDPA disabled)
- QKV projection fusion (`pipe.fuse_qkv_projections()`)
- VAE: channels-last memory format, tiling, slicing
- FP8 weight-only quantization via TorchAO on transformer, VAE, and text encoder (Ampere+ GPUs only, cc >= 8.0)
- Warmup inference during init to prime CUDA kernels

**Camera control:** `build_camera_prompt()` maps azimuth/elevation/distance values to `<sks>` token strings that the Multi-Angle LoRA was trained on. Values snap to nearest valid discrete steps.

**Rewriter memory management:** Qwen3-VL is loaded to CPU initially, moved to CUDA for inference, then moved back to CPU with `torch.cuda.empty_cache()` + `gc.collect()` so the edit model retains VRAM.

## Differences Between Deployment Modes

| Aspect | handler.py (production) | app.py (Gradio) | local_test.py |
|--------|------------------------|-----------------|---------------|
| Model source | Pre-fused baked model | HuggingFace + runtime LoRA loading | HuggingFace + runtime LoRA |
| LoRA handling | Already fused at bake time | `set_adapters()` with CPU offload | `fuse_lora()` at runtime |
| Image I/O | PyVIPS (C pipeline) | PIL directly | PIL directly |
| Quantization | FP8 via TorchAO | None | None |
| Rewriter | Imported from rewriter.py | Not included | Inline (simplified) |

## API Request/Response Format

**Input** (via RunPod `/runsync` or `/run`):
- `image` (string) or `images` (array): base64-encoded images
- `prompt` (string or array): editing instruction(s)
- `rewrite_prompt` (bool): enable Qwen3-VL refinement
- `azimuth`, `elevation`, `distance`: camera control values (snapped to discrete steps)
- `seed`, `num_inference_steps` (default 4), `guidance_scale` (default 1.0)
- `width`/`height`: auto-calculated to 1MP if null
- `output_format`: JPEG (default) or PNG, `output_quality`: 95

**Output**: `{ images: [base64...], seed, final_prompts, rewritten_log }`

**Batch modes**: array of prompts + array of images (1:1 mapping), or multiple images + single prompt (multi-image reference/stylization).
