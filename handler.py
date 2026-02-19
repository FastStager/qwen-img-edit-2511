import runpod
import torch
import numpy as np
from PIL import Image
import pyvips
import base64
import io
import math
import time
from concurrent.futures import ThreadPoolExecutor
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPlusPipeline

import rewriter

# --- Tier 3: torchao FP8 quantization ---
try:
    from torchao.quantization import quantize_, float8_weight_only
    HAS_TORCHAO_FP8 = True
    print("--- torchao FP8 available ---")
except ImportError:
    HAS_TORCHAO_FP8 = False
    print("--- torchao FP8 not found, skipping quantization ---")

# Thread pool for parallel image decode+resize / encode (Kestrel-style)
_io_pool = ThreadPoolExecutor(max_workers=4)

# --- Global perf flags (free speedup on Ampere+ GPUs) ---
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# --- Force fast SDPA backends, disable slow math fallback ---
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(False)

BAKED_MODEL_PATH = "/home/user/app/models/baked_model"

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

# --- pyvips: decode + resize in one C pipeline, zero intermediate copies ---
def vips_decode_resize(b64, target_w, target_h):
    """base64 → vips decode → Lanczos resize → PIL (for pipe). All in C."""
    raw = base64.b64decode(b64)
    img = pyvips.Image.new_from_buffer(raw, "")
    # Force 3-band sRGB (drop alpha if present)
    if img.bands == 4:
        img = img[:3]
    if img.bands == 1:
        img = img.bandjoin([img, img])
    img = img.resize(target_w / img.width, vscale=target_h / img.height, kernel="lanczos3")
    # vips → numpy → PIL (one copy, no intermediate files)
    mem = img.write_to_memory()
    arr = np.frombuffer(mem, dtype=np.uint8).reshape(img.height, img.width, img.bands)
    return Image.fromarray(arr)

def vips_decode_only(b64):
    """base64 → vips decode → PIL (no resize). For getting original dimensions."""
    raw = base64.b64decode(b64)
    img = pyvips.Image.new_from_buffer(raw, "")
    return img.width, img.height

# --- pyvips: encode output PIL → JPEG/PNG → base64, skip PIL encoder ---
def vips_encode_b64(pil_img, fmt="JPEG", quality=95):
    """PIL → numpy → vips → encode → base64. Bypasses PIL's slow encoder."""
    arr = np.asarray(pil_img)
    vi = pyvips.Image.new_from_memory(arr.data, arr.shape[1], arr.shape[0], arr.shape[2], "uchar")
    if fmt == "JPEG":
        buf = vi.jpegsave_buffer(Q=quality)
    else:
        buf = vi.pngsave_buffer()
    return base64.b64encode(buf).decode("utf-8")

def load_edit_model():
    global pipe
    if pipe is not None: return

    print("--- Loading Fused Edit Model ---")
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

    # LoRAs are already fused into weights at bake time — no loading needed

    # Fuse Q/K/V attention projections into a single matmul
    try:
        pipe.fuse_qkv_projections()
        print("--- QKV projections fused ---")
    except Exception as e:
        print(f"--- QKV fusion not supported: {e} ---")

    # Optimal memory layout for conv layers (VAE)
    try:
        pipe.vae.to(memory_format=torch.channels_last)
        print("--- VAE channels_last enabled ---")
    except Exception:
        pass

    # --- VAE tiling + slicing (reduces peak VRAM, enables larger batches) ---
    try:
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()
        print("--- VAE tiling + slicing enabled ---")
    except Exception:
        pass

    # --- FP8 quantize: transformer + VAE + text encoder ---
    if HAS_TORCHAO_FP8:
        try:
            cc = torch.cuda.get_device_capability()
            if cc[0] >= 8:
                quantize_(pipe.transformer, float8_weight_only())
                print(f"--- FP8: transformer quantized (cc {cc[0]}.{cc[1]}) ---")
                try:
                    quantize_(pipe.vae, float8_weight_only())
                    print("--- FP8: VAE quantized ---")
                except Exception as e:
                    print(f"--- FP8: VAE quantization failed (non-fatal): {e} ---")
                if hasattr(pipe, 'text_encoder'):
                    try:
                        quantize_(pipe.text_encoder, float8_weight_only())
                        print("--- FP8: text_encoder quantized ---")
                    except Exception as e:
                        print(f"--- FP8: text_encoder quantization failed (non-fatal): {e} ---")
            else:
                print(f"--- FP8 skipped: GPU cc {cc[0]}.{cc[1]} < 8.0 ---")
        except Exception as e:
            print(f"--- FP8 quantization failed: {e} ---")

    # Warmup: prime CUDA kernels so first real request isn't slow
    print("--- Warmup inference ---")
    dummy = Image.new("RGB", (512, 512), (128, 128, 128))
    with torch.inference_mode():
        pipe(image=[dummy], prompt="warmup", num_inference_steps=1, height=512, width=512)
    del dummy
    torch.cuda.empty_cache()

    print("--- Model Ready ---")

def handler(job):
    global pipe
    if pipe is None: load_edit_model()
    t0 = time.perf_counter()

    job_input = job.get('input', {})

    images_in = job_input.get('images', [])
    if not images_in and job_input.get('image'):
        images_in = [job_input.get('image')]

    if not images_in: return {"error": "No images provided"}

    # Get target dimensions from first image (peek via vips, no full decode)
    try:
        orig_w, orig_h = vips_decode_only(images_in[0])
    except Exception as e:
        return {"error": f"Image probe failed: {str(e)}"}

    w, h = get_1mp_dimensions(orig_w, orig_h)
    if job_input.get('height') and job_input.get('width'):
        h = int(job_input['height'])
        w = int(job_input['width'])

    # --- pyvips: decode + resize in one shot, parallel for batch ---
    try:
        def _decode_resize(b64):
            return vips_decode_resize(b64, w, h)

        if len(images_in) > 1:
            pil_images = list(_io_pool.map(_decode_resize, images_in))
        else:
            pil_images = [_decode_resize(images_in[0])]
    except Exception as e:
        return {"error": f"Image decode failed: {str(e)}"}

    t_decode = time.perf_counter()

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

    seed = job_input.get('seed', 42)
    steps = int(job_input.get('num_inference_steps', 4))
    cfg = float(job_input.get('guidance_scale', 1.0))
    generator = torch.Generator("cuda").manual_seed(seed)

    out_fmt = job_input.get('output_format', 'JPEG').upper()
    out_quality = int(job_input.get('output_quality', 95))

    t_preprocess = time.perf_counter()

    output_images = []
    try:
        with torch.inference_mode():
            if is_batch:
                output_images = pipe(
                    image=pil_images, prompt=prompts, num_inference_steps=steps,
                    guidance_scale=cfg, generator=generator, height=h, width=w
                ).images
            else:
                output_images = pipe(
                    image=pil_images, prompt=prompts[0], num_inference_steps=steps,
                    guidance_scale=cfg, generator=generator, height=h, width=w
                ).images
    except Exception as e:
        return {"error": f"Inference failed: {str(e)}"}
    finally:
        torch.cuda.empty_cache()

    t_inference = time.perf_counter()

    # --- pyvips: parallel output encoding, bypasses PIL encoder ---
    def _encode(img):
        return vips_encode_b64(img, fmt=out_fmt, quality=out_quality)

    if len(output_images) > 1:
        encoded = list(_io_pool.map(_encode, output_images))
    else:
        encoded = [_encode(output_images[0])]

    t_encode = time.perf_counter()

    print(f"[TIMING] decode+resize={t_decode-t0:.3f}s prompt={t_preprocess-t_decode:.3f}s "
          f"inference={t_inference-t_preprocess:.3f}s encode={t_encode-t_inference:.3f}s "
          f"total={t_encode-t0:.3f}s")

    return {
        "images": encoded,
        "seed": seed,
        "final_prompts": prompts,
        "rewritten_log": rewritten_log if do_rewrite else None
    }

if __name__ == "__main__":
    load_edit_model()
    runpod.serverless.start({"handler": handler})
