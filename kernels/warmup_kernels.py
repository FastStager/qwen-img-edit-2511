"""
AOT kernel cache pre-population script.

Run this on an H100 (same GPU arch as production) to compile all Triton kernels
at all relevant shapes. The resulting cache in TRITON_CACHE_DIR is then copied
into the Docker image for zero JIT cost at cold start.

Usage:
    TRITON_CACHE_DIR=/home/user/app/.triton_cache python kernels/warmup_kernels.py
"""

import os
import sys
import time
import torch

# Ensure triton cache dir is set
CACHE_DIR = os.environ.get("TRITON_CACHE_DIR", os.path.expanduser("~/.triton/cache"))
os.environ["TRITON_CACHE_DIR"] = CACHE_DIR
print(f"Triton cache dir: {CACHE_DIR}")


def warmup_fused_adaln():
    """Compile fused AdaLN kernel at common hidden dimensions."""
    from kernels.fused_adaln import fused_adaln_forward

    device = "cuda"
    dtype = torch.bfloat16

    # Common hidden dims for Qwen-Image-Edit transformer
    hidden_dims = [1536, 3072, 4096, 5120]
    batch_seq_sizes = [1, 4, 16, 64, 256, 1024, 4096]

    compiled = 0
    for D in hidden_dims:
        for BS in batch_seq_sizes:
            try:
                x = torch.randn(BS, D, device=device, dtype=dtype)
                scale = torch.randn(BS, D, device=device, dtype=dtype)
                shift = torch.randn(BS, D, device=device, dtype=dtype)
                _ = fused_adaln_forward(x, scale, shift)
                compiled += 1
            except Exception as e:
                print(f"  AdaLN skip D={D} BS={BS}: {e}")

    print(f"  Fused AdaLN: {compiled} variants compiled")
    return compiled


def warmup_fused_groupnorm():
    """Compile fused GroupNorm kernel at common VAE tensor shapes."""
    from kernels.fused_groupnorm_silu import fused_groupnorm_silu, _fused_groupnorm_only

    device = "cuda"
    dtype = torch.bfloat16

    # Typical VAE shapes: (batch, channels, H, W)
    shapes = [
        (1, 128, 128, 128),   # early VAE layers at 1024x1024
        (1, 256, 64, 64),
        (1, 512, 32, 32),
        (1, 512, 64, 64),
        (1, 256, 128, 128),
        (1, 128, 256, 256),
        (1, 512, 16, 16),
        # Common smaller resolutions
        (1, 128, 64, 64),
        (1, 256, 32, 32),
        (1, 512, 16, 16),
    ]

    compiled = 0
    for shape in shapes:
        N, C, H, W = shape
        num_groups = 32
        if C % num_groups != 0:
            num_groups = 16 if C % 16 == 0 else 8

        try:
            x = torch.randn(shape, device=device, dtype=dtype)
            weight = torch.randn(C, device=device, dtype=dtype)
            bias = torch.randn(C, device=device, dtype=dtype)
            _ = fused_groupnorm_silu(x, weight, bias, num_groups=num_groups)
            _ = _fused_groupnorm_only(x, weight, bias, num_groups=num_groups)
            compiled += 1
        except Exception as e:
            print(f"  GroupNorm skip {shape}: {e}")

    print(f"  Fused GroupNorm: {compiled} shape variants compiled")
    return compiled


def main():
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. Run this on a GPU machine.")
        sys.exit(1)

    device_name = torch.cuda.get_device_name(0)
    cc = torch.cuda.get_device_capability(0)
    print(f"GPU: {device_name} (cc {cc[0]}.{cc[1]})")

    if cc[0] < 8:
        print("WARNING: GPU compute capability < 8.0. FP8 kernels won't work in production.")

    t0 = time.perf_counter()

    print("\n[1/2] Warming up Fused AdaLN kernels...")
    n1 = warmup_fused_adaln()

    print("\n[2/2] Warming up Fused GroupNorm kernels...")
    n2 = warmup_fused_groupnorm()

    elapsed = time.perf_counter() - t0
    total = n1 + n2
    print(f"\nDone: {total} kernel variants compiled in {elapsed:.1f}s")
    print(f"Cache at: {CACHE_DIR}")
    print("Copy this directory into your Docker image.")


if __name__ == "__main__":
    main()
