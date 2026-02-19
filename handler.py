import runpod
import torch
from PIL import Image
import base64
import io
import math
from concurrent.futures import ThreadPoolExecutor
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPlusPipeline
import torchvision.transforms.functional as TVF

import rewriter

# --- Tier 2: turbojpeg for fast image I/O ---
try:
    from turbojpeg import TurboJPEG
    _tjpeg = TurboJPEG()
    HAS_TURBOJPEG = True
    print("--- turbojpeg available ---")
except ImportError:
    HAS_TURBOJPEG = False
    print("--- turbojpeg not found, falling back to PIL ---")

# --- Tier 3: torchao FP8 quantization ---
try:
    from torchao.quantization import quantize_, float8_weight_only
    HAS_TORCHAO_FP8 = True
    print("--- torchao FP8 available ---")
except ImportError:
    HAS_TORCHAO_FP8 = False
    print("--- torchao FP8 not found, skipping quantization ---")

# Thread pool for parallel image decode + encode (Kestrel-style)
_io_pool = ThreadPoolExecutor(max_workers=4)

# --- Global perf flags (free speedup on Ampere+ GPUs) ---
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# --- Tier 3: force fast SDPA backends, disable slow math fallback ---
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

    # --- Tier 2: VAE tiling + slicing (reduces peak VRAM, enables larger batches) ---
    try:
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()
        print("--- VAE tiling + slicing enabled ---")
    except Exception:
        pass

    # --- Tier 3: FP8 quantization on transformer (30-40% faster on Ada/Hopper GPUs) ---
    if HAS_TORCHAO_FP8:
        try:
            cc = torch.cuda.get_device_capability()
            if cc[0] >= 8:  # Ampere (8.x), Ada (8.9), Hopper (9.0)
                quantize_(pipe.transformer, float8_weight_only())
                print(f"--- FP8 quantization applied (compute capability {cc[0]}.{cc[1]}) ---")
            else:
                print(f"--- FP8 skipped: GPU compute capability {cc[0]}.{cc[1]} < 8.0 ---")
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

def base64_to_pil(b64):
    raw = base64.b64decode(b64)
    if HAS_TURBOJPEG:
        try:
            import numpy as np
            arr = _tjpeg.decode(raw)  # BGR numpy array
            return Image.fromarray(arr[:, :, ::-1])  # BGR -> RGB
        except Exception:
            pass  # not JPEG or decode error, fall back to PIL
    return Image.open(io.BytesIO(raw)).convert("RGB")

def pil_to_base64(img, fmt="JPEG", quality=95):
    if fmt == "JPEG" and HAS_TURBOJPEG:
        import numpy as np
        arr = np.array(img)[:, :, ::-1]  # RGB -> BGR
        jpeg_bytes = _tjpeg.encode(arr, quality=quality)
        return base64.b64encode(jpeg_bytes).decode("utf-8")
    buf = io.BytesIO()
    if fmt == "JPEG":
        img.save(buf, format="JPEG", quality=quality)
    else:
        img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def handler(job):
    global pipe
    if pipe is None: load_edit_model()

    job_input = job.get('input', {})

    images_in = job_input.get('images', [])
    if not images_in and job_input.get('image'):
        images_in = [job_input.get('image')]

    if not images_in: return {"error": "No images provided"}

    try:
        # Tier 2: parallel image decode via thread pool (Kestrel-style)
        if len(images_in) > 1:
            pil_images = list(_io_pool.map(base64_to_pil, images_in))
        else:
            pil_images = [base64_to_pil(images_in[0])]
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

    # --- Tier 3: GPU-side resize (skip PIL CPU LANCZOS) ---
    proc_images = []
    for img in pil_images:
        t = TVF.to_tensor(img).unsqueeze(0).to("cuda")          # [1,3,H,W] on GPU
        t = TVF.resize(t, [h, w], antialias=True).squeeze(0)    # [3,h,w] GPU resize
        proc_images.append(TVF.to_pil_image(t.cpu()))            # back to PIL for pipe
    del t

    seed = job_input.get('seed', 42)
    steps = int(job_input.get('num_inference_steps', 4))
    cfg = float(job_input.get('guidance_scale', 1.0))
    generator = torch.Generator("cuda").manual_seed(seed)

    # Output format: JPEG (fast, ~10x faster encoding) or PNG (lossless)
    out_fmt = job_input.get('output_format', 'JPEG').upper()
    out_quality = int(job_input.get('output_quality', 95))

    output_images = []
    try:
        # --- Tier 3: inference_mode > no_grad (disables more autograd tracking) ---
        with torch.inference_mode():
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
        torch.cuda.empty_cache()

    # --- Tier 3: parallel output encoding via thread pool ---
    def _encode(img):
        return pil_to_base64(img, fmt=out_fmt, quality=out_quality)

    if len(output_images) > 1:
        encoded = list(_io_pool.map(_encode, output_images))
    else:
        encoded = [_encode(output_images[0])]

    return {
        "images": encoded,
        "seed": seed,
        "final_prompts": prompts,
        "rewritten_log": rewritten_log if do_rewrite else None
    }

if __name__ == "__main__":
    load_edit_model()
    runpod.serverless.start({"handler": handler})
